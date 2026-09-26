"""
Person 4 — Threshold Tuning
Amazon ML Challenge 2026: Business Entity Resolution

Sweeps decision thresholds on the held-out validation predictions
and selects the threshold that maximizes the challenge-specific F0.5.
"""

import os
import sys
import pandas as pd

# Make project root available when run as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.evaluate import load_gt_map, score_f05_entity_level


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VALIDATION_FILE = "data/processed/validation_predictions.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
THRESHOLD_RESULTS_FILE = "data/processed/threshold_results.tsv"
BEST_THRESHOLD_FILE = "data/processed/best_threshold.txt"


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    validation_file: str = VALIDATION_FILE,
    gt_file: str = GT_FILE,
    start: float = 0.10,
    stop: float = 0.90,
    step: float = 0.01,
) -> dict:

    print("=" * 70)
    print("  PERSON 4 — THRESHOLD TUNING")
    print("=" * 70)

    # Load validation predictions
    print("\n[STEP 1] Loading validation predictions")

    df = pd.read_csv(validation_file, sep="\t", dtype=str)

    required_cols = {
        "source1_entity_id",
        "candidate_entity_id",
        "label",
        "prediction_probability",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Validation file is missing required columns: {missing}"
        )

    df["prediction_probability"] = pd.to_numeric(
        df["prediction_probability"],
        errors="coerce"
    ).fillna(0.0)

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    ).fillna(0).astype(int)

    print(f"  Loaded rows : {len(df):,}")
    print(f"  Entities    : {df['source1_entity_id'].nunique():,}")

    # Load ground truth
    print("\n[STEP 2] Loading ground truth")

    gt_map = load_gt_map(gt_file)

    print(f"  Ground-truth entities : {len(gt_map):,}")

    # Sweep thresholds
    print("\n[STEP 3] Sweeping thresholds")

    thresholds = []
    current = start

    while current <= stop + 1e-9:
        thresholds.append(round(current, 2))
        current += step

    results = []

    for threshold in thresholds:

        metrics = score_f05_entity_level(
            df=df,
            y_pred=df["prediction_probability"].values,
            gt_map=gt_map,
            threshold=threshold,
            label=f"Threshold {threshold:.2f}",
        )

        results.append({
            "threshold": threshold,
            "f05": metrics["f05"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "n_predicted": metrics["n_predicted"],
            "n_actual": metrics["n_actual"],
            "n_correct": metrics["n_correct"],
        })

    results_df = pd.DataFrame(results)

    # Find threshold with maximum F0.5
    best_idx = results_df["f05"].idxmax()
    best = results_df.loc[best_idx]

    # Save all threshold results
    os.makedirs(
        os.path.dirname(THRESHOLD_RESULTS_FILE),
        exist_ok=True
    )

    results_df.to_csv(
        THRESHOLD_RESULTS_FILE,
        sep="\t",
        index=False
    )

    # Save best threshold
    with open(BEST_THRESHOLD_FILE, "w", encoding="utf-8") as f:
        f.write(f"{best['threshold']:.2f}\n")

    # Final report
    print("\n" + "=" * 70)
    print("  THRESHOLD TUNING RESULT")
    print("=" * 70)

    print(f"\n  Best threshold : {best['threshold']:.2f}")
    print(f"  F0.5           : {best['f05']:.4f}")
    print(f"  Precision      : {best['precision']:.4f}")
    print(f"  Recall         : {best['recall']:.4f}")
    print(f"  Predicted      : {int(best['n_predicted']):,}")
    print(f"  Actual         : {int(best['n_actual']):,}")
    print(f"  Correct        : {int(best['n_correct']):,}")

    print("\n  Saved:")
    print(f"    Threshold results : {THRESHOLD_RESULTS_FILE}")
    print(f"    Best threshold    : {BEST_THRESHOLD_FILE}")

    print("=" * 70)

    return {
        "best_threshold": float(best["threshold"]),
        "f05": float(best["f05"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "results_file": THRESHOLD_RESULTS_FILE,
        "best_threshold_file": BEST_THRESHOLD_FILE,
    }


if __name__ == "__main__":
    tune_threshold()