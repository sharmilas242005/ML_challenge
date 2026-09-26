"""
Person 3 — Step 2: Evaluation Metrics
Amazon ML Challenge 2026: Business Entity Resolution

Implements:
  1. evaluate_combined_recall()   — Candidate recall ceiling for merged union set
  2. score_f05_entity_level()     — Challenge-specific macro-averaged entity-level F0.5
  3. entity_level_split()         — Reproducible Source-1-grouped train/val split

Challenge F0.5 Formula:
    F0.5 = (1 + 0.5²) × P × R / (0.5² × P + R)
         = 1.25 × P × R / (0.25 × P + R)

Precision and Recall per Source-1 entity:
  P = |predicted_matches ∩ true_matches| / |predicted_matches|
  R = |predicted_matches ∩ true_matches| / |true_matches|

Correctly predicted singletons (predict nothing, true matches = empty) score F0.5 = 1.0.
Empty predictions for a non-singleton score F0.5 = 0.0 (unless explicitly handled).
"""

import os
import sys
import csv
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ground-truth loader (shared utility)
# ---------------------------------------------------------------------------

def load_gt_map(gt_file: str) -> dict[str, set]:
    """
    Returns dict: source1_entity_id -> set of matched entity IDs.
    Singletons map to empty set.
    """
    gt_map: dict[str, set] = {}
    with open(gt_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        s1_idx = header.index("source1_entity_id")
        m_idx = header.index("matched_entity_ids")
        for row in reader:
            if not row or len(row) <= max(s1_idx, m_idx):
                continue
            s1_id = row[s1_idx].strip()
            m_str = row[m_idx].strip()
            gt_map[s1_id] = {m.strip() for m in m_str.split(",") if m.strip()} if m_str else set()
    return gt_map


# ---------------------------------------------------------------------------
# 1. Combined Candidate Recall Ceiling
# ---------------------------------------------------------------------------

def evaluate_combined_recall(
    merged_cands_file: str,
    gt_file: str = "dataset/train/train_ground_truth.tsv",
    name_cands_file: str = "data/processed/name_candidates_train.tsv",
    addr_cands_file: str = "data/processed/address_candidates_train.tsv",
) -> dict:
    """
    Evaluates:
      - Name-only recall
      - Address-only recall
      - Combined (union) recall ceiling

    The combined recall ceiling is the upper bound on what any classifier
    trained on these candidates can achieve — it cannot recover missed true pairs.

    Prints a formatted report and returns a metrics dict.
    """
    print("=" * 70)
    print("            COMBINED CANDIDATE RECALL EVALUATION")
    print("=" * 70)

    # Load ground truth
    gt_map = load_gt_map(gt_file)
    total_s1 = len(gt_map)
    gt_positive_pairs: set[tuple[str, str]] = set()
    for s1_id, match_set in gt_map.items():
        for m_id in match_set:
            gt_positive_pairs.add((s1_id, m_id))
    total_true_pairs = len(gt_positive_pairs)
    singletons = sum(1 for v in gt_map.values() if len(v) == 0)

    print(f"\n  Ground truth — Source-1 entities   : {total_s1:,}")
    print(f"  Ground truth — Singletons           : {singletons:,}")
    print(f"  Ground truth — True match pairs     : {total_true_pairs:,}")

    def _count_recall(cands_file: str, label: str) -> tuple[int, int, float]:
        """Returns (total_pairs, captured_true, recall)."""
        captured = set()
        total = 0
        with open(cands_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            s1_idx = header.index("source1_entity_id")
            cand_idx = header.index("candidate_entity_id")
            for row in reader:
                if not row or len(row) <= max(s1_idx, cand_idx):
                    continue
                s1_id = row[s1_idx].strip()
                cand_id = row[cand_idx].strip()
                pair = (s1_id, cand_id)
                if pair in gt_positive_pairs:
                    captured.add(pair)
                total += 1
        recall = len(captured) / max(total_true_pairs, 1)
        print(f"\n  [{label}] pairs: {total:,} | captured: {len(captured):,} | "
              f"recall: {recall:.4f} ({recall*100:.2f}%)")
        return total, len(captured), recall

    n_name_total, n_name_cap, r_name = _count_recall(name_cands_file, "Name   candidates")
    n_addr_total, n_addr_cap, r_addr = _count_recall(addr_cands_file, "Address candidates")

    # Combined recall from the union file
    print("\n  Computing combined (union) recall...")
    merged_captured = set()
    merged_total = 0
    name_only_captured = 0
    addr_only_captured = 0
    both_captured = 0
    missed_by_both: list[tuple[str, str]] = []

    df_merged = pd.read_csv(merged_cands_file, sep="\t", dtype=str).fillna("")
    merged_total = len(df_merged)

    for _, row in df_merged.iterrows():
        pair = (row["source1_entity_id"], row["candidate_entity_id"])
        if pair in gt_positive_pairs:
            merged_captured.add(pair)
            fn = int(row.get("from_name", 0))
            fa = int(row.get("from_address", 0))
            if fn and fa:
                both_captured += 1
            elif fn:
                name_only_captured += 1
            else:
                addr_only_captured += 1

    # Missed = true pairs not in any candidate set
    for pair in gt_positive_pairs:
        if pair not in merged_captured:
            missed_by_both.append(pair)

    n_combined_cap = len(merged_captured)
    n_missed = len(missed_by_both)
    recall_combined = n_combined_cap / max(total_true_pairs, 1)
    reduction_ratio = 1.0 - (merged_total / max(total_s1 * (merged_total // max(total_s1, 1) + 1), 1))

    print("\n" + "=" * 70)
    print("  COMBINED CANDIDATE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Source-1 entities              : {total_s1:,}")
    print(f"  Ground-truth matches (pairs)   : {total_true_pairs:,}")
    print(f"  Name candidate pairs           : {n_name_total:,}")
    print(f"  Address candidate pairs        : {n_addr_total:,}")
    print(f"  Merged candidate pairs (union) : {merged_total:,}")
    print(f"  Captured true matches          : {n_combined_cap:,}")
    print(f"  Missed true matches            : {n_missed:,}")
    print(f"")
    print(f"  Name-only recall               : {r_name:.4f}  ({r_name*100:.2f}%)")
    print(f"  Address-only recall            : {r_addr:.4f}  ({r_addr*100:.2f}%)")
    print(f"  Combined recall ceiling        : {recall_combined:.4f}  ({recall_combined*100:.2f}%)")
    print(f"")
    print(f"  Captured from both tracks      : {both_captured:,}")
    print(f"  Captured from name track only  : {name_only_captured:,}")
    print(f"  Captured from addr track only  : {addr_only_captured:,}")
    print("=" * 70)

    if missed_by_both:
        print(f"\n  WARNING: {n_missed} true match(es) not in any candidate set.")
        print("  These are irrecoverable — classifier cannot improve beyond ceiling.")
        for pair in missed_by_both[:5]:
            print(f"    Missed: {pair[0]} -> {pair[1]}")
        if n_missed > 5:
            print(f"    ... and {n_missed - 5} more")

    return {
        "total_s1": total_s1,
        "total_true_pairs": total_true_pairs,
        "merged_pairs": merged_total,
        "name_pairs": n_name_total,
        "addr_pairs": n_addr_total,
        "name_recall": r_name,
        "addr_recall": r_addr,
        "combined_recall": recall_combined,
        "captured_true": n_combined_cap,
        "missed_true": n_missed,
        "singletons": singletons,
    }


# ---------------------------------------------------------------------------
# 2. Entity-Level Grouped Train/Val Split
# ---------------------------------------------------------------------------

def entity_level_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the merged feature DataFrame by Source-1 entity ID, so all candidate rows
    for the same Source-1 entity stay in the same split.

    This prevents data leakage (a row from the same entity appearing in both
    train and validation).

    Returns:
        (df_train, df_val)
    """
    rng = np.random.default_rng(seed=random_seed)
    all_s1_ids = df["source1_entity_id"].unique()
    n_total = len(all_s1_ids)
    shuffled = rng.permutation(all_s1_ids)
    n_val = max(1, int(n_total * val_frac))
    val_ids = set(shuffled[:n_val])

    df_train = df[~df["source1_entity_id"].isin(val_ids)].copy()
    df_val = df[df["source1_entity_id"].isin(val_ids)].copy()

    print("\n" + "=" * 70)
    print("  ENTITY-LEVEL VALIDATION SPLIT")
    print("=" * 70)
    print(f"  Total Source-1 entities : {n_total:,}")
    print(f"  Train entities          : {n_total - n_val:,}")
    print(f"  Validation entities     : {n_val:,}")
    print(f"  Train rows              : {len(df_train):,}")
    print(f"  Validation rows         : {len(df_val):,}")
    pos_tr = df_train["label"].sum()
    pos_val = df_val["label"].sum()
    neg_tr = len(df_train) - pos_tr
    neg_val = len(df_val) - pos_val
    print(f"  Train  pos/neg          : {int(pos_tr):,} / {int(neg_tr):,}  "
          f"(ratio {neg_tr / max(pos_tr, 1):.1f}:1)")
    print(f"  Val    pos/neg          : {int(pos_val):,} / {int(neg_val):,}  "
          f"(ratio {neg_val / max(pos_val, 1):.1f}:1)")
    print("=" * 70)

    return df_train, df_val


# ---------------------------------------------------------------------------
# 3. Entity-Level F0.5 Scorer
# ---------------------------------------------------------------------------

def f05_score_single(precision: float, recall: float) -> float:
    """
    Compute F0.5 for a single (precision, recall) pair.
    F0.5 = (1 + 0.5²) × P × R / (0.5² × P + R)
          = 1.25 × P × R / (0.25 × P + R)
    Returns 1.0 if both P and R are undefined (0/0) — correctly predicted singleton.
    Returns 0.0 if only denominator is 0 due to P=0 and R=0 with actual positives.
    """
    numerator = 1.25 * precision * recall
    denominator = 0.25 * precision + recall
    if denominator == 0.0:
        # Both P and R are 0 — see below for singleton vs non-singleton handling
        return 0.0
    return numerator / denominator


def score_f05_entity_level(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    gt_map: dict[str, set],
    threshold: float = 0.5,
    label: str = "Evaluation",
) -> dict:
    """
    Compute the challenge-specific macro-averaged entity-level F0.5.

    For each Source-1 entity:
      - predicted_matches  = candidate IDs where prediction_probability >= threshold
      - true_matches       = gt_map[source1_entity_id]
      - precision = |predicted ∩ true| / |predicted|     (1.0 if predicted is empty)
      - recall    = |predicted ∩ true| / |true|          (1.0 if true is empty)
      - F0.5 per entity as above

    Singleton handling:
      - If true_matches is empty AND predicted_matches is empty -> F0.5 = 1.0
      - If true_matches is empty AND predicted_matches is NOT empty -> F0.5 = 0.0

    Macro average = mean F0.5 across all Source-1 entities represented in df.

    Parameters
    ----------
    df          : DataFrame with at least 'source1_entity_id', 'candidate_entity_id', 'label'
    y_pred      : 1-D array of prediction probabilities aligned with df rows
    gt_map      : full ground truth map (all S1 entities)
    threshold   : classification threshold for positive prediction
    label       : display label for printed report

    Returns dict with: f05, precision, recall, n_entities, n_predicted, n_actual
    """
    df = df.copy()
    df["pred_prob"] = y_pred
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)

    entity_f05_scores: list[float] = []
    entity_precision_scores: list[float] = []
    entity_recall_scores: list[float] = []
    total_predicted = 0
    total_actual = 0
    total_correct = 0

    for s1_id, group in df.groupby("source1_entity_id"):
        true_matches = gt_map.get(str(s1_id), set())
        predicted_ids = set(group.loc[group["pred_label"] == 1, "candidate_entity_id"].tolist())

        correct = predicted_ids & true_matches

        # Precision
        if len(predicted_ids) == 0:
            p = 1.0 if len(true_matches) == 0 else 0.0
        else:
            p = len(correct) / len(predicted_ids)

        # Recall
        if len(true_matches) == 0:
            r = 1.0  # singleton — no true matches to recall
        else:
            r = len(correct) / len(true_matches)

        # F0.5
        if len(true_matches) == 0 and len(predicted_ids) == 0:
            f = 1.0  # correct singleton
        elif len(true_matches) == 0 and len(predicted_ids) > 0:
            f = 0.0  # false positive on singleton
        else:
            f = f05_score_single(p, r)

        entity_f05_scores.append(f)
        entity_precision_scores.append(p)
        entity_recall_scores.append(r)
        total_predicted += len(predicted_ids)
        total_actual += len(true_matches)
        total_correct += len(correct)

    macro_f05 = float(np.mean(entity_f05_scores)) if entity_f05_scores else 0.0
    macro_p = float(np.mean(entity_precision_scores)) if entity_precision_scores else 0.0
    macro_r = float(np.mean(entity_recall_scores)) if entity_recall_scores else 0.0
    n_entities = len(entity_f05_scores)

    print(f"\n{'=' * 70}")
    print(f"  {label.upper()}")
    print(f"{'=' * 70}")
    print(f"  Source-1 entities evaluated : {n_entities:,}")
    print(f"  Threshold used              : {threshold:.2f}")
    print(f"  Total predicted matches     : {total_predicted:,}")
    print(f"  Total actual matches        : {total_actual:,}")
    print(f"  Total correct predictions   : {total_correct:,}")
    print(f"")
    print(f"  Macro-avg Precision (P)     : {macro_p:.4f}")
    print(f"  Macro-avg Recall    (R)     : {macro_r:.4f}")
    print(f"  Macro-avg F0.5              : {macro_f05:.4f}")
    print(f"{'=' * 70}")

    return {
        "f05": macro_f05,
        "precision": macro_p,
        "recall": macro_r,
        "n_entities": n_entities,
        "n_predicted": total_predicted,
        "n_actual": total_actual,
        "n_correct": total_correct,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    evaluate_combined_recall(
        merged_cands_file="data/processed/merged_candidates_train.tsv"
    )
