"""Project-wide constants: paths, class list, seed. Single source of truth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LABELS_CSV = DATA_DIR / "train_labels.csv"
CACHE_DIR = DATA_DIR / "cache"
EXPERIMENTS_DIR = ROOT / "experiments"
SUBMISSIONS_DIR = ROOT / "submissions"
RESULTS_XLSX = ROOT / "results" / "Strategies Computer Vision Project.xlsx"

CLASSES: list[str] = ["apple", "google", "whatsapp", "facebook", "samsung", "mozilla", "messenger"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

SEED = 42
VAL_RATIO = 0.2
AUTHOR = "Erdem SANVER"
