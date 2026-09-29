"""Flatten ``results/results.pkl`` and write ``results/results.xlsx``.

Run after ``experiments/run_experiments.py``:

    python experiments/summarize_results.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cimo.paths import RESULTS_DIR


def flatten_summaries(summaries):
    """Expand per-criterion cosine triples into their own columns."""
    rows = []
    for summary in summaries:
        row = {key: value for key, value in summary.items() if key != "cosine_similarity"}
        for criterion, (mean, std, median) in summary["cosine_similarity"].items():
            row[f"cosine_mean_{criterion}"] = mean
            row[f"cosine_std_{criterion}"] = std
            row[f"cosine_median_{criterion}"] = median
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Print the summary table and write it to Excel."""
    path = RESULTS_DIR / "results.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}. Run experiments/run_experiments.py first.")
    with path.open("rb") as handle:
        _results, summaries = pickle.load(handle)
    frame = flatten_summaries(summaries)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    print(frame)
    excel_path = RESULTS_DIR / "results.xlsx"
    frame.to_excel(excel_path, index=False)
    print(f"Wrote {excel_path}")


if __name__ == "__main__":
    main()
