"""
Person 3 — Step 3: Model Training & Validation
Amazon ML Challenge 2026: Business Entity Resolution

Workflow:
  1. Load merged feature matrix (data/processed/merged_features_train.tsv)
  2. Evaluate combined candidate recall ceiling
  3. Entity-level train/val split (no leakage)
  4. Train XGBoost binary classifier with class weighting
  5. Evaluate validation predictions at threshold 0.5 (baseline for Person 4)
  6. Save model artifact (output/entity_match_model.pkl)
  7. Save feature importance (data/processed/feature_importance.tsv)
  8. Save validation predictions (data/processed/validation_predictions.tsv)
  9. Print final summary report

Usage:
  python -m src.model.train
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make sure project root is on path when run as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.merge_features import (
    merge_candidates,
    ALL_FEATURE_COLS,
    NAME_FEATURE_COLS,
    ADDR_FEATURE_COLS_MERGED,
    MERGED_FEATS_OUT,
    MERGED_CANDS_OUT,
)
from src.model.evaluate import (
    entity_level_split,
    evaluate_combined_recall,
    score_f05_entity_level,
    load_gt_map,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GT_FILE = "dataset/train/train_ground_truth.tsv"
MERGED_FEATS_FILE = "data/processed/merged_features_train.tsv"
MERGED_CANDS_FILE = "data/processed/merged_candidates_train.tsv"
MODEL_OUT = "output/entity_match_model.pkl"
FEAT_IMPORTANCE_OUT = "data/processed/feature_importance.tsv"
VAL_PREDICTIONS_OUT = "data/processed/validation_predictions.tsv"

RANDOM_SEED = 42
VAL_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Feature column identification
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Identify all numeric feature columns, excluding IDs, provenance flags, and label.
    Raise an error if any expected feature column is missing.
    """
    non_feature = {
        "source1_entity_id",
        "candidate_entity_id",
        "from_name",
        "from_address",
        "label",
    }
    # Use the canonical ordered feature list (defines order for XGBoost)
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in non_feature and c not in feature_cols]
    if extra:
        print(f"  NOTE: Ignoring unexpected columns: {extra}")
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: Expected feature cols not found in data: {missing}")
    return feature_cols


def check_no_id_leakage(feature_cols: list[str]) -> None:
    """Assert that no entity ID columns are used as predictors."""
    id_indicators = {"entity_id", "source1", "candidate"}
    for col in feature_cols:
        for indicator in id_indicators:
            if indicator in col.lower():
                raise ValueError(
                    f"LEAKAGE DETECTED: Feature column '{col}' looks like an ID. "
                    "IDs must never be used as predictive features."
                )
    print("  [OK] No ID leakage detected in feature columns.")


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> object:
    """
    Train an XGBoost binary classifier.
    Uses scale_pos_weight to handle class imbalance.
    Uses early stopping on validation logloss.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "xgboost is required for training. Install it with: pip install xgboost"
        )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    print(f"\n  XGBoost parameters:")
    print(f"    scale_pos_weight  = {scale_pos_weight:.2f}  (neg/pos = {neg_count}/{pos_count})")
    print(f"    n_estimators      = 500 (with early stopping, patience=30)")
    print(f"    max_depth         = 6")
    print(f"    learning_rate     = 0.05")
    print(f"    subsample         = 0.8")
    print(f"    colsample_bytree  = 0.8")
    print(f"    random_state      = {RANDOM_SEED}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        early_stopping_rounds=30,
        verbosity=1,
    )

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"\n  Best iteration (early stopping): {best_iter}")

    return model


# ---------------------------------------------------------------------------
# Feature importance report
# ---------------------------------------------------------------------------

def report_feature_importance(
    model,
    feature_names: list[str],
    out_file: str = FEAT_IMPORTANCE_OUT,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract, save, and print feature importances from the trained model.
    Separates name-related vs address-related features.
    """
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df.to_csv(out_file, sep="\t", index=False)
    print(f"\n  Feature importance saved -> {out_file}")

    print(f"\n  TOP {min(top_n, len(fi_df))} FEATURES BY IMPORTANCE")
    print(f"  {'Rank':<5} {'Feature':<35} {'Importance':>12}  {'Track'}")
    print("  " + "-" * 65)
    for i, row in fi_df.head(top_n).iterrows():
        track = "addr " if row["feature"].startswith("addr_") else "name "
        print(f"  {i+1:<5} {row['feature']:<35} {row['importance']:>12.6f}  {track}")

    # Separated summary
    name_fi = fi_df[~fi_df["feature"].str.startswith("addr_")]
    addr_fi = fi_df[fi_df["feature"].str.startswith("addr_")]

    print(f"\n  === NAME-RELATED FEATURES (Person 1 feedback) ===")
    for _, row in name_fi.iterrows():
        print(f"    {row['feature']:<35}  {row['importance']:.6f}")

    print(f"\n  === ADDRESS-RELATED FEATURES (Person 2 feedback) ===")
    for _, row in addr_fi.iterrows():
        print(f"    {row['feature']:<35}  {row['importance']:.6f}")

    return fi_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    force_remerge: bool = False,
    gt_file: str = GT_FILE,
    merged_feats_file: str = MERGED_FEATS_FILE,
    merged_cands_file: str = MERGED_CANDS_FILE,
    model_out: str = MODEL_OUT,
    feat_importance_out: str = FEAT_IMPORTANCE_OUT,
    val_predictions_out: str = VAL_PREDICTIONS_OUT,
) -> dict:
    """
    Full Person 3 training pipeline. Returns a metrics dict.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("  PERSON 3 — ENTITY RESOLUTION MODEL TRAINING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1: Merge candidates and features (or load cached result)
    # ------------------------------------------------------------------ #
    print("\n[STEP 1] Candidate merge & feature construction")
    if force_remerge or not os.path.exists(merged_feats_file):
        print("  Running full merge (this may take a minute)...")
        merge_candidates()
    else:
        print(f"  Found existing merged features: {merged_feats_file}")
        print("  Skipping re-merge (use force_remerge=True to regenerate)")

    print("\n  Loading merged feature matrix...")
    df = pd.read_csv(merged_feats_file, sep="\t", dtype=str).fillna("0")
    df["label"] = df["label"].astype(int)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")

    # ------------------------------------------------------------------ #
    # Step 2: Identify feature columns & check for leakage
    # ------------------------------------------------------------------ #
    print("\n[STEP 2] Feature column identification")
    feature_cols = get_feature_cols(df)
    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")
    check_no_id_leakage(feature_cols)

    # Convert features to float
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------ #
    # Step 3: Combined recall ceiling
    # ------------------------------------------------------------------ #
    print("\n[STEP 3] Combined candidate recall ceiling")
    if os.path.exists(merged_cands_file):
        recall_metrics = evaluate_combined_recall(
            merged_cands_file=merged_cands_file,
            gt_file=gt_file,
        )
    else:
        print("  WARNING: merged_candidates file not found, skipping recall ceiling check")
        recall_metrics = {}

    # ------------------------------------------------------------------ #
    # Step 4: Entity-level train/val split
    # ------------------------------------------------------------------ #
    print("\n[STEP 4] Entity-level validation split")
    df_train, df_val = entity_level_split(df, val_frac=VAL_FRACTION, random_seed=RANDOM_SEED)

    # ------------------------------------------------------------------ #
    # Step 5: Prepare arrays
    # ------------------------------------------------------------------ #
    print("\n[STEP 5] Preparing training arrays")
    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["label"].values.astype(np.int32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["label"].values.astype(np.int32)

    pos_train = int(y_train.sum())
    neg_train = int((y_train == 0).sum())
    pos_val = int(y_val.sum())
    neg_val = int((y_val == 0).sum())
    print(f"  Train: {len(X_train):,} rows  | pos={pos_train:,}  neg={neg_train:,}  "
          f"ratio={neg_train/max(pos_train,1):.1f}:1")
    print(f"  Val  : {len(X_val):,} rows  | pos={pos_val:,}  neg={neg_val:,}  "
          f"ratio={neg_val/max(pos_val,1):.1f}:1")
    print(f"  Positive rate (train): {pos_train/max(len(y_train),1)*100:.2f}%")

    # ------------------------------------------------------------------ #
    # Step 6: Train XGBoost
    # ------------------------------------------------------------------ #
    print("\n[STEP 6] Training XGBoost classifier")
    model = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)

    # ------------------------------------------------------------------ #
    # Step 7: Validation predictions
    # ------------------------------------------------------------------ #
    print("\n[STEP 7] Generating validation predictions")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds_05 = (val_probs >= 0.5).astype(int)

    gt_map = load_gt_map(gt_file)
    val_metrics = score_f05_entity_level(
        df=df_val,
        y_pred=val_probs,
        gt_map=gt_map,
        threshold=0.5,
        label="Validation Set — Baseline Threshold 0.5",
    )

    # ------------------------------------------------------------------ #
    # Step 8: Save validation predictions for Person 4
    # ------------------------------------------------------------------ #
    print("\n[STEP 8] Saving validation predictions")
    val_out_df = df_val[["source1_entity_id", "candidate_entity_id", "label"]].copy()
    val_out_df["prediction_probability"] = val_probs
    val_out_df["prediction_at_0_5"] = val_preds_05
    os.makedirs(os.path.dirname(val_predictions_out), exist_ok=True)
    val_out_df.to_csv(val_predictions_out, sep="\t", index=False)
    print(f"  Saved -> {val_predictions_out}")

    # ------------------------------------------------------------------ #
    # Step 9: Feature importance
    # ------------------------------------------------------------------ #
    print("\n[STEP 9] Feature importance")
    os.makedirs(os.path.dirname(feat_importance_out), exist_ok=True)
    fi_df = report_feature_importance(model, feature_cols, feat_importance_out)

    # ------------------------------------------------------------------ #
    # Step 10: Save model
    # ------------------------------------------------------------------ #
    print("\n[STEP 10] Saving model artifact")
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "random_seed": RANDOM_SEED,
            "val_fraction": VAL_FRACTION,
        }, f)
    print(f"  Saved -> {model_out}")

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    combined_recall = recall_metrics.get("combined_recall", float("nan"))
    top10_feats = fi_df.head(10)[["feature", "importance"]].values.tolist()

    print("\n" + "=" * 70)
    print("  PERSON 3 — FINAL PIPELINE REPORT")
    print("=" * 70)
    print(f"\n  CANDIDATE RECALL")
    print(f"    Name-only recall       : {recall_metrics.get('name_recall', float('nan')):.4f}")
    print(f"    Address-only recall    : {recall_metrics.get('addr_recall', float('nan')):.4f}")
    print(f"    Combined recall ceil   : {combined_recall:.4f}")
    print(f"\n  TRAINING DATASET")
    print(f"    Total merged pairs     : {len(df):,}")
    print(f"    Positives              : {int(df['label'].sum()):,}")
    print(f"    Negatives              : {len(df) - int(df['label'].sum()):,}")
    print(f"    Number of features     : {len(feature_cols)}")
    print(f"\n  VALIDATION RESULTS (threshold=0.5)")
    print(f"    Entity-level F0.5      : {val_metrics['f05']:.4f}")
    print(f"    Precision              : {val_metrics['precision']:.4f}")
    print(f"    Recall                 : {val_metrics['recall']:.4f}")
    print(f"    Predicted matches      : {val_metrics['n_predicted']:,}")
    print(f"    Actual matches         : {val_metrics['n_actual']:,}")
    print(f"\n  TOP 10 FEATURES BY IMPORTANCE")
    for i, (feat, imp) in enumerate(top10_feats, 1):
        track = "addr" if feat.startswith("addr_") else "name"
        print(f"    {i:2d}. [{track}] {feat:<35} {imp:.6f}")
    print(f"\n  ARTIFACTS")
    print(f"    Model                  : {model_out}")
    print(f"    Feature importance     : {feat_importance_out}")
    print(f"    Validation predictions : {val_predictions_out}")
    print(f"\n  REPRODUCE WITH")
    print(f"    python -m src.model.train")
    print("=" * 70)

    return {
        "recall_metrics": recall_metrics,
        "val_metrics": val_metrics,
        "feature_cols": feature_cols,
        "n_train_rows": len(df_train),
        "n_val_rows": len(df_val),
        "n_positives": int(df["label"].sum()),
        "n_negatives": len(df) - int(df["label"].sum()),
        "model_path": model_out,
        "feat_importance_path": feat_importance_out,
        "val_predictions_path": val_predictions_out,
        "top10_features": top10_feats,
    }


if __name__ == "__main__":
    run_training_pipeline()
