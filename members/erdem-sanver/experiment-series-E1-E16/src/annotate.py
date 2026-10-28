"""Add the human interpretation or the Kaggle public score to an experiment, then rebuild the Excel.

  python -m src.annotate E3_simple_cnn_white --explanation "..."
  python -m src.annotate E3_simple_cnn_white --kaggle 0.8123
"""
import argparse
import json

from src.config import EXPERIMENTS_DIR
from src.results import export_excel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    ap.add_argument("--explanation")
    ap.add_argument("--kaggle", type=float)
    a = ap.parse_args()
    path = EXPERIMENTS_DIR / f"{a.exp_id}.json"
    if not path.exists():
        raise SystemExit(f"No such experiment: {path.name}")
    rec = json.loads(path.read_text())
    if a.explanation is not None:
        rec["explanation"] = a.explanation
    if a.kaggle is not None:
        rec["kaggle_public"] = a.kaggle
    path.write_text(json.dumps(rec, indent=2))
    print(f"updated {path.name}; Excel rows: {export_excel()}")


if __name__ == "__main__":
    main()
