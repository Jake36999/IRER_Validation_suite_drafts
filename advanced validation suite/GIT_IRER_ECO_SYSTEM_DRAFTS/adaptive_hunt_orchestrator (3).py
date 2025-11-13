import os
import subprocess
import pandas as pd
import time
import sys
import shlex
import glob
import argparse
from typing import Tuple, List, Any

print("--- [ORCHESTRATOR] ENGAGED (v11.1: Fixes Bootstrap and Python Executable) ---")

# --- 1. CLI Configuration ---
def parse_args():
    p = argparse.ArgumentParser(description="Adaptive hunt orchestrator v11.1")
    p.add_argument("--worker", default="worker_v7.py", help="Worker script (default: worker_v7.py)")
    p.add_argument("--hunter", default="aste_hunter.py", help="Hunter script (default: aste_hunter.py)")
    p.add_argument("--master_dir", default="sweep_runs", help="Top-level output dir")
    p.add_argument("--todo", default="ASTE_generation_todo.json", help="Shared TODO filename")
    p.add_argument("--hunts", type=int, default=1, help="How many hunts to run")
    p.add_argument("--offset", type=int, default=33, help="Hunt index offset (e.g., 33 -> HUNT_033)")
    p.add_argument("--goal_sse", type=float, default=0.10, help="SSE target threshold")
    p.add_argument("--goal_gens", type=int, default=3, help="Consecutive generations to meet goal")
    p.add_argument("--max_gens", type=int, default=6, help="Safety cap per hunt (small for 3D smoke test)")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds between generations")
    return p.parse_args()

# --- 2. Helper Functions ---
def run_command(parts: List[str]):
    """Run a command, stream stdout, return exit code. Uses sys.executable."""
    cmd_str = " ".join(shlex.quote(x) for x in parts)
    print(f"\nExecuting: {cmd_str}\n")

    proc = subprocess.Popen(
        parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    last = []
    while True:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None: break
        if line:
            line = line.rstrip("\n")
            print(line)
            last.append(line)
            if len(last) > 10: last.pop(0)

    rc = proc.poll() or 0
    if rc != 0:
        print(f"\n[ORCH] Command failed (exit {rc}). Last lines:")
        for l in last:
            print("  ", l)
    return rc

def get_best_sse(master_dir: str, hunt_id: str) -> Tuple[float, str]:
    """Find the lowest SSE across candidate ledgers."""
    cands = glob.glob(os.path.join(master_dir, hunt_id, f"ledger_{hunt_id}*.csv"))
    scored = []
    for f in cands:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "final_sse" in df.columns:
                v = df[df["final_sse"] < 90000]["final_sse"]
                if not v.empty:
                    scored.append((float(v.min()), os.path.getmtime(f), f))
        except Exception:
            pass
    if not scored: return float("inf"), ""
    scored.sort(key=lambda x: (x[0], -x[1]))
    best_sse, _, path = scored[0]
    return best_sse, path

def needs_bootstrap(todo_file: str, hunt_dir: str) -> bool:
    """True if we must call Hunter first (no TODO and no ledger present)."""
    if os.path.exists(todo_file): return False
    if not os.path.isdir(hunt_dir): return True
    # Check if any ledger file exists
    if len(glob.glob(os.path.join(hunt_dir, "ledger_*.csv"))) > 0: return False
    return True

# --- 3. Main Orchestrator Logic ---
def main():
    args = parse_args()

    # Check dependencies (simplified here, full check is in worker_v7.py's __main__)
    if not os.path.exists(args.worker) or not os.path.exists(args.hunter):
        print(f"--- [ORCH] CRITICAL: Worker ({args.worker}) or Hunter ({args.hunter}) not found.")
        sys.exit(1)

    os.makedirs(args.master_dir, exist_ok=True)

    for i in range(args.hunts):
        idx = i + args.offset
        HUNT_ID = f"SNCGL_ADAPTIVE_HUNT_{idx:03d}"
        hunt_dir = os.path.join(args.master_dir, HUNT_ID)
        os.makedirs(hunt_dir, exist_ok=True)

        print("\n" + "-" * 80)
        print(f"--- STARTING ADAPTIVE HUNT: {HUNT_ID} (3D Stable Exploration) ---")
        print("-" * 80)

        consecutive = 0
        gen = 0
        best_overall = float("inf")

        while True:
            # Command argument lists
            hunter_cmd = [sys.executable, args.hunter, HUNT_ID, args.todo]
            worker_cmd = [sys.executable, args.worker, HUNT_ID, args.todo]

            print(f"\n--- Hunt {HUNT_ID}, Generation {gen} ---")

            # --- Bootstrap Check: Run Hunter FIRST if necessary ---
            if needs_bootstrap(args.todo, hunt_dir):
                print("[ORCH] Bootstrap: Calling Hunter first to create initial TODO...")
                rc = run_command(hunter_cmd)
                if rc != 0: break # Exit loop on Hunter failure

            # --- Step 1: Run Worker (Consumes TODO, creates ledger row) ---
            rc = run_command(worker_cmd)
            if rc != 0: break # Exit loop on Worker failure

            # --- Step 2: Run Hunter (Consumes ledger row, writes next TODO) ---
            rc = run_command(hunter_cmd)
            if rc != 0: break # Exit loop on Hunter failure

            # --- Step 3: Monitor Termination Conditions ---
            current_best, _ = get_best_sse(args.master_dir, HUNT_ID)
            best_overall = min(best_overall, current_best)
            print(f"[ORCH] Best SSE now: {current_best:.12f} | Best overall: {best_overall:.12f}")

            if current_best <= args.goal_sse:
                consecutive += 1
                print(f"GOAL MET: {consecutive}/{args.goal_gens} consecutive")
            else:
                consecutive = 0
                print("GOAL NOT MET: consecutive reset")

            if consecutive >= args.goal_gens:
                print(f"\n--- Hunt {HUNT_ID} COMPLETED ---")
                break

            if gen >= args.max_gens:
                print(f"\n--- Hunt {HUNT_ID} STOPPED --- (hit max_gens={args.max_gens})")
                break

            gen += 1
            time.sleep(args.sleep)

        # Final cleanup
        if os.path.exists(args.todo):
            try: os.remove(args.todo); print(f"Cleaned up residual '{args.todo}'.")
            except Exception as e: print(f"Warning: couldn't remove '{args.todo}': {e}")

    print("\n" + "-" * 80)
    print("--- ORCHESTRATOR FINISHED ALL HUNTS ---")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

print("adaptive_hunt_orchestrator.py successfully written.")
