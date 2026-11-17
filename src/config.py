from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
SPLITS_DIR = ROOT_DIR / "splits"
MODEL_DIR = ROOT_DIR / "model"
LOGS_DIR = ROOT_DIR / "logs"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
PRESENT_DIR = ROOT_DIR / "present"
METRICS_FILE = MODEL_DIR / "metrics.json"

for d in [DATASETS_DIR, SPLITS_DIR, MODEL_DIR, LOGS_DIR, CHECKPOINTS_DIR, PRESENT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/muril-base-cased"
LABELS = ["for", "against", "none"]
LABEL2ID = {"for": 0, "against": 1, "none": 2}
ID2LABEL = {0: "for", 1: "against", 2: "none"}

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 4
RANDOM_SEED = 42

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
