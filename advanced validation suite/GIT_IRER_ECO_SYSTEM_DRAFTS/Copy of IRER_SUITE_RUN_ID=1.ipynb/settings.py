# Minimal settings.py content
import os

# --- RUN CONFIGURATION ---
NUM_GENERATIONS = 1         # Reduced for faster testing
POPULATION_SIZE = 2         # Reduced for faster testing
RUN_ID = 3                  # Current project ID for archival

# --- EVOLUTIONARY ALGORITHM PARAMETERS ---
LAMBDA_FALSIFIABILITY = 100.0 # Weight for the fitness bonus
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.1

# --- FILE PATHS AND DIRECTORIES (MUST BE SYNCHRONIZED) ---
BASE_DIR = os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROVENANCE_DIR = os.path.join(BASE_DIR, "provenance_reports")
LEDGER_FILE = os.path.join(BASE_DIR, "runs_ledger", "run_log.csv")

WORKER_SCRIPT = "worker_unified.py"
PPN_TEST_SCRIPT = "ppn_gamma_test.py"

# --- SCIENTIFIC CRITERIA ---
SCIENTIFIC_PASS_SSE = 0.05
ULTRA_LOW_SSE = 0.001
