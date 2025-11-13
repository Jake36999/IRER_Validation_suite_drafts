"""
project_api.py
CLASSIFICATION: API Adapter Layer (Web UI Backend)
GOAL: Exposes core hunt functionality and the AI Debugging Assistant to the
      web control panel via simple functions.
"""

import sys
import os

# Ensure current directory is in sys.path for local module imports
sys.path.insert(0, os.getcwd())

import json
import subprocess
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import uuid

# Import centralized components
import settings
import ai_assistant_core
import log_scraper

# --- API Functions for UI Control ---

def _run_subprocess(script_name, *args):
    """Internal helper to run core Python scripts."""
    cmd = [sys.executable, script_name] + list(args)
    try:
        # NOTE: We set check=False for CLI scripts as failures are often part of the expected workflow
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr, "log_stream": (result.stdout + result.stderr).splitlines()}
    except FileNotFoundError:
        return {"success": False, "stderr": f"Error: Script '{script_name}' not found.", "log_stream": [f"Error: Script '{script_name}' not found."]}

def start_refined_hunt(seed_hash: Optional[str] = None) -> Dict[str, Any]:
    """
    Starts the full adaptive hunt orchestrator.
    """
    # ... logic for creating seed file ...
    result = _run_subprocess("adaptive_hunt_orchestrator.py")
    # ... logic for cleaning up seed file ...
    return result

def validate_final_candidate(config_hash: str) -> Dict[str, Any]:
    """
    Runs the final, definitive validation on a single hash (Phase 4).
    """
    result = _run_subprocess("validation_pipeline.py", config_hash)
    final_status = "CERTIFIED" if "DUAL MANDATE PASSED" in result.get('stdout', '') else "FAILED_VALIDATION"
    result['final_status'] = final_status
    return result

def get_ledger_summary() -> Dict[str, Any]:
    """
    Loads the ledger and computes key performance indicators (KPIs) for the UI dashboard.
    """
    try:
        df = pd.read_csv(settings.LEDGER_FILE)

        # 1. Best Scientific Result
        df_successful = df[df['log_prime_sse'] < 900].copy()
        if df_successful.empty:
            return {"status": "NO_DATA", "best_sse": 999.0, "best_fitness": 0.0, "best_hash": "N/A", "fail_rate_pct": 100}

        best_run = df.loc[df['fitness'].idxmax()]

        # 2. Geometric Stability Check (Assumed PPN is 1.0 for now)
        sse_pass = best_run['log_prime_sse'] <= settings.SCIENTIFIC_PASS_SSE

        # 3. Fail Rate
        fail_count = len(df[df['log_prime_sse'] >= 900])
        fail_rate_pct = round((fail_count / len(df)) * 100, 1)

        # 4. Convergence Data for Plotting (Example)
        sse_history = df_successful.groupby('generation')['log_prime_sse'].min().tolist()

        return {
            "status": "CONVERGED" if len(df) > 10 else "EXPLORING",
            "best_sse": round(best_run['log_prime_sse'], 6),
            "best_fitness": round(best_run['fitness'], 4),
            "best_hash": best_run['config_hash'],
            "fail_rate_pct": fail_rate_pct,
            "is_scientific_pass": sse_pass,
            "sse_history": sse_history
        }

    except FileNotFoundError:
        return {"status": "INITIALIZING", "best_sse": 999.0, "best_fitness": 0.0, "best_hash": "N/A", "fail_rate_pct": 0.0}
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "best_sse": 999.0, "best_fitness": 0.0, "best_hash": "N/A", "fail_rate_pct": 0.0}

def run_deconvolution_test():
    return _run_subprocess("deconvolution_validator.py")

def run_tda_taxonomy(config_hash: str):
    return _run_subprocess("tda_taxonomy_validator.py", config_hash)


# --- AGNOSTIC AI DEBUGGING ASSISTANT API ---

def ai_scan_logs(roots: List[str] = [os.getcwd()]) -> Dict[str, Any]:
    # Placeholder implementation
    return {"success": True, "final_status": "MOCK_SCAN_COMPLETE", "log_stream": ["Mock log scan performed."]}

def ai_query_debug(
    prompt: str,
    bug_instances: Optional[List[str]] = None,
    transcripts: Optional[List[str]] = None,
    artifacts_summary: Optional[str] = None
) -> Dict[str, Any]:
    # Placeholder implementation
    report_markdown = ai_assistant_core.generate_debugging_report(
        bug_instances=bug_instances or [],
        transcripts=transcripts or [],
        artifacts_summary=artifacts_summary,
        extra_context={"query_origin": "cli"}
    )
    return {
        "success": True,
        "final_status": "AI_RESPONSE_READY",
        "report_markdown": report_markdown,
        "log_stream": [f"Analysis complete. Heuristic Core responded."]
    }

if __name__ == "__main__":
    # Ensure correct indentation for CLI block (fixed structure)
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print(json.dumps(get_ledger_summary(), indent=2))
    elif len(sys.argv) > 2 and sys.argv[1] == "validate":
        validate_result = validate_final_candidate(sys.argv[2])
        print(json.dumps(validate_result, indent=2))
        if not validate_result["success"]:
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "hunt":
        hunt_result = start_refined_hunt()
        print(json.dumps(hunt_result, indent=2))
        if not hunt_result["success"]:
            sys.exit(1)
    elif len(sys.argv) > 2 and sys.argv[1] == "tda":
        tda_result = run_tda_taxonomy(sys.argv[2])
        print(json.dumps(tda_result, indent=2))
        if not tda_result["success"]:
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "ai":
        if len(sys.argv) > 2 and sys.argv[2] == "scan":
            roots = sys.argv[3:] if len(sys.argv) > 3 else [os.getcwd()]
            report = ai_scan_logs(roots)
        else:
            prompt = sys.argv[2] if len(sys.argv) > 2 else "Please analyze the last failure."
            report = ai_query_debug(prompt)
        print(json.dumps(report, indent=2))
        if not report["success"]:
            sys.exit(1)
    else:
        print("Usage: python project_api.py [summary|hunt|validate <hash>|tda <config_hash>|ai [query <prompt>|scan <root>]]")
