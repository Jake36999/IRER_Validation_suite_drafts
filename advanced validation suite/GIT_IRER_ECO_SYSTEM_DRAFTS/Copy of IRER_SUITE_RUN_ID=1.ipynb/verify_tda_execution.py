import subprocess
import sys
import json
import os
import shutil
import pandas as pd
import time # Ensure time is imported

# --- Configuration ---
LEDGER_PATH = os.path.join(os.getcwd(), "runs_ledger", "run_log.csv")
DIRS_TO_CLEAN = ["runs_ledger", "configs", "data", "provenance_reports"]


def clean_workspace():
    """Wipes artifacts from previous runs for a clean test."""
    print("[Verification] Cleaning workspace...")
    for d in DIRS_TO_CLEAN:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Re-create necessary directories
    for path in ["gravity", "configs", "data", "provenance_reports", os.path.join("runs_ledger")]:
        os.makedirs(path, exist_ok=True)


def run_adaptive_hunt():
    """Runs the Adaptive Hunt Orchestrator to populate data."""
    print("\n--- A. Running Adaptive Hunt to generate data and CSVs ---")
    hunt_command = [sys.executable, "project_api.py", "hunt"]

    try:
        result_hunt = subprocess.run(hunt_command, capture_output=True, text=True, check=True, timeout=120)
        print("Hunt Orchestrator executed successfully.")

        # Print the full stdout from project_api.py for confirmation and debugging
        print("\n--- project_api.py Hunt STDOUT ---")
        print(result_hunt.stdout)

        # Parse the JSON output from project_api.py to check its internal success status
        api_output = json.loads(result_hunt.stdout)
        if not api_output.get("success", False):
            print("\n--- project_api.py Hunt reported internal failure ---", file=sys.stderr)
            if "log_stream" in api_output:
                for line in api_output["log_stream"]:
                    print(f"API LOG: {line}", file=sys.stderr)
            raise Exception(f"project_api.py Hunt command reported internal failure: {api_output.get('stderr', 'No specific stderr reported by API.')}")

    except subprocess.CalledProcessError as e:
        print(f"\n--- CRITICAL ERROR: project_api.py Hunt failed ---", file=sys.stderr)
        print(f"RETURN CODE: {e.returncode}", file=sys.stderr)
        print(f"STDOUT: {e.stdout}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        raise Exception(f"Command '{' '.join(hunt_command)}' returned non-zero exit status {e.returncode}. Check output above for details.")
    except json.JSONDecodeError as e:
        print(f"\n--- CRITICAL ERROR: Could not parse JSON output from project_api.py ---", file=sys.stderr)
        print(f"JSON Decode Error: {e}", file=sys.stderr)
        print(f"Received STDOUT: {result_hunt.stdout}", file=sys.stderr)
        raise Exception(f"project_api.py did not return valid JSON: {e}")


def retrieve_best_candidate_hash():
    """Retrieves the hash of the best-performing candidate from the ledger."""
    if os.path.exists(LEDGER_PATH):
        df = pd.read_csv(LEDGER_PATH)
        if not df.empty and 'config_hash' in df.columns and 'fitness' in df.columns:
            # Ensure 'fitness' column is not all NaN before calling idxmax
            if df['fitness'].isnull().all():
                print("\nFATAL: 'fitness' column is all NaN. No valid runs for TDA.", file=sys.stderr)
                sys.exit(1)

            # Find the hash with the highest fitness
            best_config_hash = df.loc[df['fitness'].idxmax()]['config_hash']
            print(f"\nFound Best Candidate Hash: {best_config_hash}")
            return best_config_hash

    print("\nFATAL: Ledger file not found or empty after hunt. Cannot proceed with TDA.", file=sys.stderr)
    sys.exit(1)


def run_tda_validation(config_hash: str):
    """Executes the final TDA validation command."""
    print(f"\n--- B. Running TDA Taxonomy Validation on Hash: {config_hash[:10]}... ---")

    tda_command = [sys.executable, "project_api.py", "tda", config_hash]

    try:
        result_tda = subprocess.run(tda_command, capture_output=True, text=True, check=True, timeout=30)

        print("\n--- TDA Validation STDOUT (from project_api.py) ---\n")
        print(result_tda.stdout)
        print("\n--- TDA Validation STDERR (from project_api.py) ---\n")
        print(result_tda.stderr)

        # Parse the JSON output from project_api.py to check its internal success status
        api_output = json.loads(result_tda.stdout)
        if "Taxonomy:" in result_tda.stdout or api_output.get("success", False):
            print("\n✅ TDA Structural Validation PASSED: Taxonomy result obtained.")
        else:
            raise Exception("TDA Validation failed: No taxonomy result found in output or API reported failure.")

    except subprocess.CalledProcessError as e:
        print(f"\n--- CRITICAL ERROR: project_api.py TDA failed ---", file=sys.stderr)
        print(f"RETURN CODE: {e.returncode}", file=sys.stderr)
        print(f"STDOUT: {e.stdout}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        raise Exception(f"Command '{' '.join(tda_command)}' returned non-zero exit status {e.returncode}. Check output above for details.")
    except json.JSONDecodeError as e:
        print(f"\n--- CRITICAL ERROR: Could not parse JSON output from project_api.py for TDA ---", file=sys.stderr)
        print(f"JSON Decode Error: {e}", file=sys.stderr)
        print(f"Received STDOUT: {result_tda.stdout}", file=sys.stderr)
        raise Exception(f"project_api.py TDA did not return valid JSON: {e}")


def main():
    try:
        clean_workspace()

        # 1. Run Hunt
        run_adaptive_hunt()

        # 2. Retrieve Hash
        best_hash = retrieve_best_candidate_hash()

        # 3. Run TDA
        run_tda_validation(best_hash)

        print("\n\n✅ END-TO-END SUITE VERIFICATION SUCCESSFUL.")

    except Exception as e:
        print(f"\n\n❌ CRITICAL PIPELINE FAILURE: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
