from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = ["idle", "hello", "thank_you", "yes", "no", "help"]

SEQUENCE_LENGTH = 30
NUM_SEQUENCES_PER_CLASS = 30
NUM_FEATURES = 258

HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = len(ACTIONS)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 60
