import os

# Base directory for all operations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory configurations
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROVENANCE_DIR = os.path.join(BASE_DIR, 'provenance')
LEDGER_FILE = os.path.join(BASE_DIR, 'ledger', 'hunt_ledger.csv')

# Simulation settings
NUM_GENERATIONS = 10
POPULATION_SIZE = 5
WORKER_SCRIPT = os.path.join(BASE_DIR, 'worker_script.py') # Placeholder name

# Evolutionary Algorithm Parameters
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
LAMBDA_FALSIFIABILITY = 100.0

# AI Assistant Settings
AI_ASSISTANT_MODE = 'MOCK' # Default to MOCK
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # For actual AI model integration
