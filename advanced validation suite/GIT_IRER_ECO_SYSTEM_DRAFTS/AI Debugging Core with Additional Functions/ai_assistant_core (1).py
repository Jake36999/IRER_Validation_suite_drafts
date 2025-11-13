"""
ai_assistant_core.py
CLASSIFICATION: Adaptive Learning Engine (ASTE V10.0 - Falsifiability Bonus)
GOAL: Provides mock responses for AI debugging queries until the full AI model is integrated.
"""

from typing import Dict, Any, List, Optional
import os
import json
import re # Import re for parsing AI response

import google.generativeai as genai
import settings # Import settings to get AI_ASSISTANT_MODE and GEMINI_API_KEY

# Configure the generative AI model if API key is available
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

def generate_debugging_report(
    bug_instances: List[str],
    transcripts: List[str],
    artifacts_summary: Optional[Dict[str, Any]] = None,
    extra_context: Optional[Dict[str, Any]] = None
) -> str:
    """Generates a debugging report based on provided inputs.

    This function will use a real AI model (Gemini Pro) if AI_ASSISTANT_MODE
    is set to 'GEMINI_PRO', otherwise it falls back to a mock response.
    """
    if os.getenv('AI_ASSISTANT_MODE') == 'GEMINI_PRO' and settings.GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-pro')

            # Construct the prompt for the AI model
            prompt_parts = [
                "You are an expert software debugger and an AI assistant for a scientific simulation platform (ASTE).",
                "Your task is to analyze simulation job failures and generate a concise debugging report.",
                "Focus on identifying the root cause, potential solutions, and suggesting parameter adjustments or code fixes.",
                "Include specific line numbers or module names if identifiable from stack traces.",
                "\n--- Failure Details ---"
            ]
            if bug_instances:
                prompt_parts.append("\nBug Instances (stderr/exceptions):\n" + "\n".join(bug_instances))
            if transcripts:
                prompt_parts.append("\nTranscripts (stdout/logs):\n" + "\n".join(transcripts))
            if artifacts_summary:
                prompt_parts.append("\nArtifacts Summary:\n" + json.dumps(artifacts_summary, indent=2))
            if extra_context:
                prompt_parts.append("\nExtra Context:\n" + json.dumps(extra_context, indent=2))

            prompt = "\n".join(prompt_parts)

            # Generate content using the Gemini Pro model
            response = model.generate_content(prompt)
            return "# AI-Powered Debugging Report (Gemini Pro)\n\n" + response.text
        except Exception as e:
            print(f"[AI Assistant Error] Failed to generate report with Gemini Pro: {e}", file=sys.stderr)
            # Fallback to mock if AI call fails
            return _generate_mock_debugging_report(bug_instances, transcripts, artifacts_summary, extra_context, "AI call failed.")
    else:
        # Fallback to mock implementation
        return _generate_mock_debugging_report(bug_instances, transcripts, artifacts_summary, extra_context, "MOCK mode active or API key missing.")

def _generate_mock_debugging_report(
    bug_instances: List[str],
    transcripts: List[str],
    artifacts_summary: Optional[Dict[str, Any]],
    extra_context: Optional[Dict[str, Any]],
    status_message: str
) -> str:
    """Helper function for generating a mock debugging report."""
    report = f"# Mock AI Debugging Report\n\n"
    report += f"**Status:** Placeholder for AI Core functionality (Mode: Mmock, reason: {status_message}).\n\n"

    # 1. Core Summary
    report += "## Analysis Summary (Simulated)\n"
    report += "The Agnostic AI Assistant is operating in **mock mode**. It performs basic parsing of the input payload to ensure the pipeline contract is met, but provides a simulated, non-diagnostic response. This allows downstream processes (like the Control Panel) to proceed with expected data flow.\n\n"

    # 2. Bug Instances Check
    if bug_instances:
        report += f"## Analysis of {len(bug_instances)} Identified 'Bug Instances'\n"
        report += f"The input payload contained **{len(bug_instances)}** distinct error traces/exceptions.\n"

        # Display the first line of each bug instance as a brief overview
        for i, bug in enumerate(bug_instances[:3]): # Displaying a max of 3 for brevity
            first_line = bug.splitlines()[0].strip()
            report += f"- **Instance {i+1} Overview:** `{first_line}`\n"

        if len(bug_instances) > 3:
             report += f"- ...and **{len(bug_instances) - 3}** more instances.\n"

        report += "(Full stack trace analysis is omitted in mock mode.)\n\n"
    else:
        report += "## Input Payload Analysis\n"
        report += "No explicit `bug_instances` (stack traces or error messages) were found in the input payload, suggesting a possible successful run or a non-standard failure.\n\n"

    # 3. Artifacts and Context
    report += "## Supplemental Context Review\n"

    if transcripts:
        report += f"- **Transcripts:** **{len(transcripts)}** log snippets/code contexts were processed.\n"
    else:
        report += "- **Transcripts:** No supplementary log/code transcripts were provided.\n"

    if artifacts_summary:
        summary_str = artifacts_summary.get('summary', 'Artifacts received.') if isinstance(artifacts_summary, dict) else str(artifacts_summary)
        first_line_summary = summary_str.splitlines()[0].strip()
        report += f"- **Artifacts Summary:** {first_line_summary} (Details suppressed).\n"
    else:
        report += "- **Artifacts Summary:** No structured artifacts were submitted.\n"

    if extra_context:
        report += f"- **Extra Context Keys:** {', '.join(extra_context.keys())} received.\n"
    else:
        report += "- **Extra Context:** No additional context was provided.\n"

    report += "\n"

    # 4. Final Mock Recommendation
    report += "**Recommendation (Mock):**\n"
    report += "The system contract for `generate_debugging_report` has been satisfied. The next logical step is to:\n"
    report += "1.  Proceed with the **Post-Analysis Handler** to process this report.\n"
    report += "2.  For a real diagnosis, the `AI_ASSISTANT_MODE` variable must be updated to a value that loads a fully integrated AI model (e.g., `MODEL_GCP_VERTEX`).\n"

    return report

def analyze_code_snippet(
    code_snippet: str,
    associated_error: Optional[str] = None,
    context_files: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Generates a mock analysis and suggested fix for a code snippet."""

    mock_analysis = (
        "MOCK Analysis: The provided code snippet was analyzed against a simulated set of rules. "
        "The core issue appears to be a mismatch between the expected variable scope and its usage "
        "in the JAX environment. This is a common pattern observed in the ASTE V8.0 platform "
        "when using complex control flow constructs without explicit `jax.lax.scan` or `jax.jit` boundary management."
    )

    mock_fix = (
        f"# MOCK FIX for the issue associated with:\n# {code_snippet.splitlines()[0].strip()}\n"
        "### SUGGESTED ACTION (MOCK):\n"
        "1.  **Refactor:** Ensure all state updates (like counters or logs) are managed via `jax.lax.scan` or wrapped in an opaque function boundary if side effects are unavoidable.\n"
        "2.  **Explicit Scoping:** If a key is missing (e.g., 'JAX_CONFIG_HASH'), wrap the configuration access in a `try-except` block or ensure configuration loading is complete before execution.\n"
        "3.  **Example Correction:** The snippet needs structural correction, likely related to proper indentation or dependency import."
    )

    return {
        "analysis": mock_analysis,
        "suggested_fix": mock_fix,
        "status": "MOCK_SUCCESS",
        "original_snippet_start": code_snippet[:50] + "..."
    }

def propose_mitigation_strategy(
    report_summary: str,
    recent_history: List[Dict[str, Any]]
) -> List[str]:
    """Proposes high-level strategic steps to mitigate recurring system failures."""

    # Analyze the mock report summary and history to generate a contextually-appropriate mock response.
    # We'll use static mock data for consistency in this placeholder phase.

    mock_strategies = [
        "**Strategy 1: Formalize State Management (Configuration Mismatch Mitigation)**: Implement the UNIFIED_SCIENCE_CORE module to decouple all configuration state from the JAX execution context, preventing silent key errors and cross-run configuration contamination. (See: Agnostic AI Debugging Framework Design)",
        "**Strategy 2: Enforce Indentation & Linter Compliance**: The recurrence of structural errors (like `IndentationError`) suggests a need to integrate an aggressive pre-commit hook that enforces PEP 8 compliance across all JAX worker scripts. This will reduce Mean Time to Resolution (MTTR) on trivial structural issues.",
        "**Strategy 3: Upgrade TDA Validation Logging**: Enhance logging within `tda_taxonomy_validator.py` (a known critical component) to record the exact input parameters and the Ripser configuration. This allows for post-mortem analysis of non-convergent solutions.",
        "**Strategy 4: Parameter Space Decoupling**: Isolate the evolutionary parameter hunting loop from the core physics simulation. This aligns with the HPC_MODULARITY_SUITE mandate, improving system stability and enabling independent scaling."
    ]

    return mock_strategies

def get_ai_guidance_for_breeding(
    best_run: Optional[Dict[str, Any]],
    current_generation: int
) -> Dict[str, float]:
    """Provides AI-guided mutation parameters for breeding."""
    # Default to mock logic
    mutation_rate = max(0.05, 0.1 - current_generation * 0.005) if best_run and best_run.get("fitness", 0) > 0.001 else min(0.3, 0.1 + current_generation * 0.01)
    mutation_strength = max(0.05, 0.5 - current_generation * 0.01) if best_run and best_run.get("fitness", 0) > 0.001 else min(1.0, 0.5 + current_generation * 0.02)

    if os.getenv('AI_ASSISTANT_MODE') == 'GEMINI_PRO' and settings.GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-pro')

            prompt_parts = [
                "You are an AI assistant for an evolutionary algorithm in a scientific simulation platform (ASTE).",
                "Your goal is to provide optimal mutation_rate and mutation_strength values for the next generation.",
                "Consider the current best_run's fitness and parameters, and the current generation number.",
                "A higher fitness is better. Aim for exploitation if fitness is high, and exploration if fitness is low or early in the generations.",
                "Output ONLY the two numerical values for mutation_rate and mutation_strength, comma-separated, followed by a brief reasoning.",
                "Example: 0.15, 0.30 - Reasoning: Reduced mutation for fine-tuning.",
                f"\nCurrent Generation: {current_generation}"
            ]
            if best_run:
                prompt_parts.append(f"Best Run (Fitness: {best_run.get('fitness', 'N/A'):.4f}, Params: {{'D': {best_run.get('param_D', 'N/A')}, 'eta': {best_run.get('param_eta', 'N/A')}, 'rho_vac': {best_run.get('param_rho_vac', 'N/A')}, 'a_coupling': {best_run.get('param_a_coupling', 'N/A')}}})")
            else:
                prompt_parts.append("No best run found yet (or fitness is non-positive). Prioritize exploration.")

            prompt = "\n".join(prompt_parts)
            response = model.generate_content(prompt)
            response_text = response.text.split(' - Reasoning:')[0].strip()

            match = re.match(r'(\d+\.?\d*),\s*(\d+\.?\d*)', response_text)
            if match:
                ai_mutation_rate = float(match.group(1))
                ai_mutation_strength = float(match.group(2))
                # Basic validation/clipping of AI-suggested values
                mutation_rate = max(0.01, min(ai_mutation_rate, 0.5))
                mutation_strength = max(0.01, min(ai_mutation_strength, 2.0))
                print(f"[AI Guidance] Gemini Pro suggested: Mutation Rate={{mutation_rate:.2f}}, Strength={{mutation_strength:.2f}}", file=sys.stdout)
            else:
                print(f"[AI Guidance Error] Could not parse Gemini Pro response: {response_text}. Falling back to mock logic.", file=sys.stderr)

        except Exception as e:
            print(f"[AI Guidance Error] Gemini Pro API call failed: {e}. Falling back to mock logic.", file=sys.stderr)

    return {"mutation_rate": mutation_rate, "mutation_strength": mutation_strength}

def get_resource_allocation_guidance(
    current_generation: int,
    historical_performance_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Provides AI-guided resource allocation parameters."""
    # Default mock logic (fallback)
    if current_generation < 5:
        suggested_parallel_jobs = 1
        estimated_batch_time_seconds = 60 * 5
    elif current_generation < 10:
        suggested_parallel_jobs = 2
        estimated_batch_time_seconds = 60 * 3
    else:
        suggested_parallel_jobs = 4
        estimated_batch_time_seconds = 60 * 2

    if os.getenv('AI_ASSISTANT_MODE') == 'GEMINI_PRO' and settings.GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-pro')

            prompt_parts = [
                "You are an AI assistant for resource management in a scientific simulation platform (ASTE).",
                "Your task is to provide optimal resource allocation for the next simulation generation.",
                "Consider the current generation number and a summary of historical performance (if available).",
                "Output ONLY two numerical values: `suggested_parallel_jobs` and `estimated_batch_time_seconds`, comma-separated, followed by a brief reasoning.",
                "Example: 2, 180 - Reasoning: Increased parallelism due to stable performance.",
                f"\nCurrent Generation: {current_generation}"
            ]
            if historical_performance_summary:
                prompt_parts.append("Historical Performance Summary:\n" + json.dumps(historical_performance_summary, indent=2))
            else:
                prompt_parts.append("No detailed historical performance summary available.")

            prompt = "\n".join(prompt_parts)
            response = model.generate_content(prompt)
            response_text = response.text.split(' - Reasoning:')[0].strip()

            match = re.match(r'(\d+),\s*(\d+)', response_text)
            if match:
                ai_suggested_parallel_jobs = int(match.group(1))
                ai_estimated_batch_time_seconds = int(match.group(2))
                # Basic validation/clipping of AI-suggested values
                suggested_parallel_jobs = max(1, min(ai_suggested_parallel_jobs, 10)) # Max 10 parallel jobs
                estimated_batch_time_seconds = max(60, min(ai_estimated_batch_time_seconds, 60 * 60)) # Min 1 min, Max 1 hour
                print(f"[AI Resource Guidance] Gemini Pro suggested: Parallel Jobs={{suggested_parallel_jobs}}, Batch Time={{estimated_batch_time_seconds}}s", file=sys.stdout)
            else:
                print(f"[AI Resource Guidance Error] Could not parse Gemini Pro response: {response_text}. Falling back to mock logic.", file=sys.stderr)

        except Exception as e:
            print(f"[AI Resource Guidance Error] Gemini Pro API call failed: {e}. Falling back to mock logic.", file=sys.stderr)

    return {
        "suggested_parallel_jobs": suggested_parallel_jobs,
        "estimated_batch_time_seconds": estimated_batch_time_seconds,
        "mock_analysis_hint": "Based on current generation and (mock) historical performance, adjusted parallelism."
    }

# --- Example Usage (Optional, for self-testing) ---
if __name__ == '__main__':
    import sys # Import sys for example usage
    mock_bugs = [
        "KeyError: 'JAX_CONFIG_HASH' not found in config dictionary.",
        "IndentationError: unexpected indent in line 45 of worker_script.py",
    ]

    # Test generate_debugging_report with MOCK mode
    os.environ['AI_ASSISTANT_MODE'] = 'MOCK'
    report_mock = generate_debugging_report(mock_bugs, transcripts=['log line 1'], extra_context={'run_id': 'MOCK-1'})
    print(report_mock)

    # Test generate_debugging_report with GEMINI_PRO mode (requires GEMINI_API_KEY to be set)
    if not os.getenv('GEMINI_API_KEY'):
        os.environ['GEMINI_API_KEY'] = 'YOUR_GEMINI_API_KEY_HERE' # Placeholder
        print("Warning: GEMINI_API_KEY not set. Using placeholder for example.")

    os.environ['AI_ASSISTANT_MODE'] = 'GEMINI_PRO'
    # Temporarily disable stdout to avoid verbose model output in test if API key is not real
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    report_gemini = generate_debugging_report(mock_bugs, transcripts=['log line 1'], extra_context={'run_id': 'GEMINI-1'})
    sys.stdout.close()
    sys.stdout = original_stdout
    print("\n--- Report with GEMINI_PRO mode (mocked if API key invalid) ---")
    print(report_gemini)

    # Test new functions
    mock_code = "def worker_func():\n  if check_condition:\n    return result"
    analysis = analyze_code_snippet(mock_code, associated_error="KeyError: 'foo'")
    print("\n--- Code Analysis Mock ---")
    print(analysis['analysis'])

    strategies = propose_mitigation_strategy(report_mock, [])
    print("\n--- Mitigation Strategy Mock ---")
    for s in strategies:
        print(f"- {s}")

    # Test get_ai_guidance_for_breeding
    print("\n--- AI Guidance for Breeding ---")
    # Test with MOCK mode first
    os.environ['AI_ASSISTANT_MODE'] = 'MOCK'
    guidance_gen0_mock = get_ai_guidance_for_breeding(None, 0)
    print(f"Generation 0 (MOCK, no best run): {guidance_gen0_mock}")
    guidance_gen10_good_mock = get_ai_guidance_for_breeding({"fitness": 0.01, "param_D": 1.0, "param_eta": 0.5, "param_rho_vac": 1.0, "param_a_coupling": 1.0}, 10)
    print(f"Generation 10 (MOCK, good best run): {guidance_gen10_good_mock}")

    # Test with GEMINI_PRO mode (requires GEMINI_API_KEY)
    os.environ['AI_ASSISTANT_MODE'] = 'GEMINI_PRO'
    # Temporarily disable stdout to avoid verbose model output in test if API key is not real
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    guidance_gen0_gemini = get_ai_guidance_for_breeding(None, 0)
    sys.stdout.close()
    sys.stdout = original_stdout
    print(f"Generation 0 (GEMINI_PRO, no best run): {guidance_gen0_gemini}")

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    guidance_gen10_good_gemini = get_ai_guidance_for_breeding({"fitness": 0.01, "param_D": 1.0, "param_eta": 0.5, "param_rho_vac": 1.0, "param_a_coupling": 1.0}, 10)
    sys.stdout.close()
    sys.stdout = original_stdout
    print(f"Generation 10 (GEMINI_PRO, good best run): {guidance_gen10_good_gemini}")

    # Test get_resource_allocation_guidance
    print("\n--- AI Resource Allocation ---")
    # Test with MOCK mode first
    os.environ['AI_ASSISTANT_MODE'] = 'MOCK'
    resource_gen0_mock = get_resource_allocation_guidance(0)
    print(f"Generation 0 (MOCK): {resource_gen0_mock}")
    resource_gen7_mock = get_resource_allocation_guidance(7, {"avg_job_time": 150, "failure_rate": 0.1})
    print(f"Generation 7 (MOCK): {resource_gen7_mock}")

    # Test with GEMINI_PRO mode (requires GEMINI_API_KEY)
    os.environ['AI_ASSISTANT_MODE'] = 'GEMINI_PRO'
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    resource_gen0_gemini = get_resource_allocation_guidance(0)
    sys.stdout.close()
    sys.stdout = original_stdout
    print(f"Generation 0 (GEMINI_PRO): {resource_gen0_gemini}")

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    resource_gen7_gemini = get_resource_allocation_guidance(7, {"avg_job_time": 150, "failure_rate": 0.1})
    sys.stdout.close()
    sys.stdout = original_stdout
    print(f"Generation 7 (GEMINI_PRO): {resource_gen7_gemini}")
