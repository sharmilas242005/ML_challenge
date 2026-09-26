"""
Person 3 — Step 1: Candidate Merge & Feature Matrix Construction
Amazon ML Challenge 2026: Business Entity Resolution

Responsibilities:
  1. Union name-based and address-based candidate pairs (deduplicated).
  2. Left-join name features onto every merged pair.
  3. Left-join address features onto every merged pair.
  4. Fill missing feature values with 0.0 (appropriate for similarity scores).
  5. Assign binary labels from train_ground_truth.tsv.
  6. Save:
       data/processed/merged_candidates_train.tsv   (union candidate list)
       data/processed/merged_features_train.tsv     (full feature matrix with label)

Output schema for merged_features_train.tsv:
  source1_entity_id, candidate_entity_id,
  [6 name features], [13 address features],
  from_name, from_address,   <- provenance flags
  label

Name features (prefix: kept as-is):
  levenshtein_sim, token_jaccard, char_ngram_cos,
  token_sort_ratio, token_set_ratio, exact_match

Address features (prefix: addr_*):
  addr_levenshtein_sim, addr_token_jaccard, addr_tfidf_cosine,
  addr_exact_pincode_match, addr_exact_city_match, addr_exact_country_match,
  addr_exact_address_match, addr_char_ngram_cos, addr_token_overlap_count,
  addr_address_len_ratio, addr_digit_overlap_ratio,
  addr_house_number_match, addr_landmark_sim
"""

import os
import sys
import csv
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# File paths (all relative to project root)
# ---------------------------------------------------------------------------
NAME_CANDS_FILE = "data/processed/name_candidates_train.tsv"
ADDR_CANDS_FILE = "data/processed/address_candidates_train.tsv"
NAME_FEATS_FILE = "data/processed/name_features_train.tsv"
ADDR_FEATS_FILE = "data/processed/address_features_train.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
MERGED_CANDS_OUT = "data/processed/merged_candidates_train.tsv"
MERGED_FEATS_OUT = "data/processed/merged_features_train.tsv"

# ---------------------------------------------------------------------------
# Feature column names as produced by Person 1 and Person 2
# ---------------------------------------------------------------------------
NAME_FEATURE_COLS = [
    "levenshtein_sim",
    "token_jaccard",
    "char_ngram_cos",
    "token_sort_ratio",
    "token_set_ratio",
    "exact_match",
]

# Address feature cols as written by Person 2 (no prefix in source file)
ADDR_FEATURE_COLS_SRC = [
    "levenshtein_sim",
    "token_jaccard",
    "tfidf_cosine",
    "exact_pincode_match",
    "exact_city_match",
    "exact_country_match",
    "exact_address_match",
    "char_ngram_cos",
    "token_overlap_count",
    "address_len_ratio",
    "digit_overlap_ratio",
    "house_number_match",
    "landmark_sim",
]
# Address features get an "addr_" prefix in the merged matrix to avoid
# collision with the two shared column names (levenshtein_sim, token_jaccard, char_ngram_cos)
ADDR_FEATURE_COLS_MERGED = ["addr_" + c for c in ADDR_FEATURE_COLS_SRC]

# All numeric feature columns in the merged matrix (what the model will see)
ALL_FEATURE_COLS = NAME_FEATURE_COLS + ADDR_FEATURE_COLS_MERGED

# Merged feature matrix header
MERGED_HEADER = (
    ["source1_entity_id", "candidate_entity_id"]
    + NAME_FEATURE_COLS
    + ADDR_FEATURE_COLS_MERGED
    + ["from_name", "from_address", "label"]
)


def load_ground_truth(gt_file: str) -> dict[str, set]:
    """
    Load train_ground_truth.tsv into a dict:
        source1_entity_id -> set of matched entity IDs

    Handles:
      - Singletons (empty matched_entity_ids -> empty set)
      - Comma-separated multi-match (e.g. 'S2-TR000004,S3-TR000005')
    """
    gt_map: dict[str, set] = {}
    with open(gt_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty ground truth file: {gt_file}")
        try:
            s1_idx = header.index("source1_entity_id")
            m_idx = header.index("matched_entity_ids")
        except ValueError as e:
            raise ValueError(f"Ground truth missing expected column: {e}") from e

        for row in reader:
            if not row or len(row) <= max(s1_idx, m_idx):
                continue
            s1_id = row[s1_idx].strip()
            m_str = row[m_idx].strip()
            if m_str:
                gt_map[s1_id] = {m.strip() for m in m_str.split(",") if m.strip()}
            else:
                gt_map[s1_id] = set()

    return gt_map


def load_candidates_long(filepath: str) -> pd.DataFrame:
    """
    Load a long-format candidate file into a DataFrame.
    Expected columns: source1_entity_id, candidate_entity_id, source, block_reason
    Returns only the pair columns; extra columns are preserved.
    """
    df = pd.read_csv(filepath, sep="\t", dtype=str).fillna("")
    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate file {filepath} missing columns: {missing}")
    return df


def load_features(filepath: str, id_cols: list, feat_cols: list) -> pd.DataFrame:
    """
    Load a feature file, keeping only the ID columns + the specified feature columns.
    Extra columns (e.g. 'label' already in the source file) are dropped here
    since labels will be re-assigned centrally from ground truth.
    """
    df = pd.read_csv(filepath, sep="\t", dtype=str).fillna("")
    keep_cols = [c for c in id_cols + feat_cols if c in df.columns]
    return df[keep_cols]


def merge_candidates(
    name_cands_file: str = NAME_CANDS_FILE,
    addr_cands_file: str = ADDR_CANDS_FILE,
    name_feats_file: str = NAME_FEATS_FILE,
    addr_feats_file: str = ADDR_FEATS_FILE,
    gt_file: str = GT_FILE,
    merged_cands_out: str = MERGED_CANDS_OUT,
    merged_feats_out: str = MERGED_FEATS_OUT,
) -> pd.DataFrame:
    """
    Core merge function.

    Returns the merged feature DataFrame (also saved to merged_feats_out).
    """
    os.makedirs(os.path.dirname(merged_cands_out), exist_ok=True)
    os.makedirs(os.path.dirname(merged_feats_out), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load candidate lists
    # ------------------------------------------------------------------
    print("[1/7] Loading candidate pairs...")
    name_cands = load_candidates_long(name_cands_file)
    addr_cands = load_candidates_long(addr_cands_file)

    print(f"  Name candidates : {len(name_cands):>10,} pairs")
    print(f"  Addr candidates : {len(addr_cands):>10,} pairs")

    # Tag provenance before union
    name_cands["from_name"] = 1
    addr_cands["from_address"] = 1

    # Keep only the pair key + provenance for the union step
    name_pairs = name_cands[["source1_entity_id", "candidate_entity_id", "from_name"]].copy()
    addr_pairs = addr_cands[["source1_entity_id", "candidate_entity_id", "from_address"]].copy()

    # ------------------------------------------------------------------
    # 2. Union and deduplicate candidate pairs
    # ------------------------------------------------------------------
    print("[2/7] Unioning and deduplicating candidate pairs...")
    merged_cands = pd.merge(
        name_pairs.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]),
        addr_pairs.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]),
        on=["source1_entity_id", "candidate_entity_id"],
        how="outer",
    )
    # Fill provenance flags
    merged_cands["from_name"] = merged_cands["from_name"].fillna(0).astype(int)
    merged_cands["from_address"] = merged_cands["from_address"].fillna(0).astype(int)

    total_pairs = len(merged_cands)
    name_only = (merged_cands["from_name"] == 1) & (merged_cands["from_address"] == 0)
    addr_only = (merged_cands["from_name"] == 0) & (merged_cands["from_address"] == 1)
    both = (merged_cands["from_name"] == 1) & (merged_cands["from_address"] == 1)

    print(f"  Merged (union)  : {total_pairs:>10,} unique pairs")
    print(f"    from name only: {name_only.sum():>10,}")
    print(f"    from addr only: {addr_only.sum():>10,}")
    print(f"    from both     : {both.sum():>10,}")

    # Save merged candidates (for debugging / evaluation)
    merged_cands.to_csv(merged_cands_out, sep="\t", index=False)
    print(f"  Saved merged candidates -> {merged_cands_out}")

    # ------------------------------------------------------------------
    # 3. Load feature files
    # ------------------------------------------------------------------
    print("[3/7] Loading feature files...")
    id_cols = ["source1_entity_id", "candidate_entity_id"]

    # Name features: keep only the 6 feature cols (drop 'label' if present)
    name_feat_cols_available = [c for c in NAME_FEATURE_COLS
                                if c in pd.read_csv(name_feats_file, sep="\t", nrows=0).columns]
    name_feats = load_features(name_feats_file, id_cols, name_feat_cols_available)
    print(f"  Name features   : {len(name_feats):>10,} rows x {len(name_feats.columns)} cols")

    # Address features: keep the 13 address feature cols (drop 'label')
    addr_feat_cols_available = [c for c in ADDR_FEATURE_COLS_SRC
                                if c in pd.read_csv(addr_feats_file, sep="\t", nrows=0).columns]
    addr_feats_raw = load_features(addr_feats_file, id_cols, addr_feat_cols_available)

    # Rename address feature cols to addr_* prefix
    rename_map = {src: merged for src, merged in zip(ADDR_FEATURE_COLS_SRC, ADDR_FEATURE_COLS_MERGED)
                  if src in addr_feats_raw.columns}
    addr_feats = addr_feats_raw.rename(columns=rename_map)
    print(f"  Addr features   : {len(addr_feats):>10,} rows x {len(addr_feats.columns)} cols")

    # ------------------------------------------------------------------
    # 4. Join features onto merged candidate pairs
    # ------------------------------------------------------------------
    print("[4/7] Joining features onto merged candidates...")

    # Deduplicate feature files (take first occurrence per pair — features are deterministic)
    name_feats = name_feats.drop_duplicates(subset=id_cols, keep="first")
    addr_feats = addr_feats.drop_duplicates(subset=id_cols, keep="first")

    merged_df = merged_cands.merge(name_feats, on=id_cols, how="left")
    merged_df = merged_df.merge(addr_feats, on=id_cols, how="left")

    print(f"  Rows after join : {len(merged_df):>10,}")

    # ------------------------------------------------------------------
    # 5. Fill missing features with 0.0
    # ------------------------------------------------------------------
    print("[5/7] Filling missing feature values with 0.0...")
    for col in ALL_FEATURE_COLS:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0.0)
        else:
            print(f"  WARNING: expected feature column '{col}' not found — inserting zeros")
            merged_df[col] = 0.0

    # ------------------------------------------------------------------
    # 6. Assign labels from ground truth
    # ------------------------------------------------------------------
    print("[6/7] Assigning labels from ground truth...")
    gt_map = load_ground_truth(gt_file)

    # Build a set of positive pairs for O(1) lookup
    gt_positive_pairs: set[tuple[str, str]] = set()
    for s1_id, match_set in gt_map.items():
        for m_id in match_set:
            gt_positive_pairs.add((s1_id, m_id))

    merged_df["label"] = [
        1 if (str(r.source1_entity_id), str(r.candidate_entity_id)) in gt_positive_pairs
        else 0
        for r in merged_df.itertuples(index=False)
    ]

    positives = int(merged_df["label"].sum())
    negatives = len(merged_df) - positives
    print(f"  Total pairs     : {len(merged_df):>10,}")
    print(f"  Positives       : {positives:>10,}")
    print(f"  Negatives       : {negatives:>10,}")
    imbalance_ratio = negatives / max(positives, 1)
    print(f"  Neg/Pos ratio   : {imbalance_ratio:>10.1f}:1")

    # ------------------------------------------------------------------
    # 7. Save merged feature matrix
    # ------------------------------------------------------------------
    print("[7/7] Saving merged feature matrix...")
    output_cols = id_cols + ALL_FEATURE_COLS + ["from_name", "from_address", "label"]
    # Ensure we only include cols that actually exist
    output_cols = [c for c in output_cols if c in merged_df.columns]
    merged_df[output_cols].to_csv(merged_feats_out, sep="\t", index=False)
    print(f"  Saved merged features -> {merged_feats_out}")
    print(f"  Columns: {len(output_cols)} total ({len(ALL_FEATURE_COLS)} numeric features)")

    return merged_df[output_cols]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    merge_candidates()
