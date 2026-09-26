"""
Step 4 — Name Similarity Features Module
ML Challenge 2026: Business Entity Resolution (Person 1 - Name Domain Track)

Computes 6 core name similarity features for every candidate pair:
  1. Levenshtein normalized similarity (1 - edit_dist / max_len)
  2. Token Jaccard similarity (set intersection over union)
  3. Character 3-gram cosine similarity (robust to typos and transliteration)
  4. Token-sort-ratio (handles word-order swaps)
  5. Token-set-ratio (handles substring / prefix / partial matches)
  6. Exact-match-after-normalization boolean flag (1 or 0)

Schema:
  source1_entity_id, candidate_entity_id, levenshtein_sim, token_jaccard,
  char_ngram_cos, token_sort_ratio, token_set_ratio, exact_match[, label]

Label column is populated for train split via join against train_ground_truth.tsv.
"""

import sys
import os
import csv
import math
import argparse
from collections import Counter
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


def get_char_ngrams(s: str, n: int = 3) -> list[str]:
    """Generate character n-grams from a string."""
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s) - n + 1)]


def char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """Compute cosine similarity of character n-gram frequencies."""
    if not s1 or not s2:
        return 0.0
    ngrams1 = get_char_ngrams(s1, n)
    ngrams2 = get_char_ngrams(s2, n)
    if not ngrams1 or not ngrams2:
        return 0.0
    c1 = Counter(ngrams1)
    c2 = Counter(ngrams2)
    dot = sum(c1[k] * c2[k] for k in c1 if k in c2)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot / (norm1 * norm2))


def compute_pair_features(name1: str, name2: str) -> tuple[float, float, float, float, float, int]:
    """
    Compute all 6 name similarity features for a pair of normalized names.
    Returns:
      (levenshtein_sim, token_jaccard, char_ngram_cos, token_sort_ratio, token_set_ratio, exact_match)
    """
    if not name1 or not name2:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # 1. Levenshtein normalized similarity [0.0, 1.0]
    lev_sim = float(Levenshtein.normalized_similarity(name1, name2))

    # 2. Token Jaccard
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    union_len = len(tokens1 | tokens2)
    token_jaccard = float(len(tokens1 & tokens2) / union_len) if union_len > 0 else 0.0

    # 3. Char 3-gram cosine similarity
    ngram_cos = char_ngram_cosine(name1, name2, n=3)

    # 4. Token sort ratio [0.0, 1.0]
    sort_ratio = float(fuzz.token_sort_ratio(name1, name2) / 100.0)

    # 5. Token set ratio [0.0, 1.0]
    set_ratio = float(fuzz.token_set_ratio(name1, name2) / 100.0)

    # 6. Exact match flag
    exact_match = 1 if name1 == name2 else 0

    return (
        round(lev_sim, 4),
        round(token_jaccard, 4),
        round(ngram_cos, 4),
        round(sort_ratio, 4),
        round(set_ratio, 4),
        exact_match
    )


def load_candidate_ids(candidates_file: str) -> set[str]:
    """Collect all unique entity IDs appearing in the candidate file."""
    print(f"Collecting active entity IDs from {candidates_file}...")
    active_ids = set()
    with open(candidates_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        s1_idx = header.index("source1_entity_id")
        cand_idx = header.index("candidate_entity_id")
        for row in reader:
            if row and len(row) > max(s1_idx, cand_idx):
                active_ids.add(row[s1_idx].strip())
                active_ids.add(row[cand_idx].strip())
    print(f"Total active entities in candidate pairs: {len(active_ids):,}")
    return active_ids


def load_name_lookup(normalized_file: str, filter_ids: set[str] | None = None) -> dict[str, str]:
    """Load entity_id -> name_norm map from normalized file (filtered by active IDs if given)."""
    print(f"Loading normalized names from {normalized_file}...")
    lookup = {}
    with open(normalized_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        eid_idx = header.index("entity_id")
        name_idx = header.index("name_norm")
        for row in reader:
            if row and len(row) > max(eid_idx, name_idx):
                eid = row[eid_idx].strip()
                if filter_ids is None or eid in filter_ids:
                    lookup[eid] = row[name_idx].strip()
    print(f"Loaded {len(lookup):,} active names into memory.")
    return lookup


def load_ground_truth(gt_file: str) -> set[tuple[str, str]]:
    """Load ground truth matches into a set of (source1_id, matched_id) pairs."""
    print(f"Loading ground truth from {gt_file}...")
    gt_pairs = set()
    with open(gt_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        s1_idx = header.index("source1_entity_id")
        m_idx = header.index("matched_entity_ids")
        for row in reader:
            if row and len(row) > max(s1_idx, m_idx):
                s1_id = row[s1_idx].strip()
                m_str = row[m_idx].strip()
                if m_str:
                    for m_id in m_str.split(","):
                        m_id = m_id.strip()
                        if m_id:
                            gt_pairs.add((s1_id, m_id))
    print(f"Loaded {len(gt_pairs):,} ground truth positive pairs.")
    return gt_pairs


def compute_name_features(
    candidates_file: str,
    normalized_file: str,
    output_features_file: str,
    ground_truth_file: str | None = None
) -> str:
    """
    Main feature extraction function.
    Reads candidate pairs, attaches the 6 name similarity features,
    and attaches label if ground_truth_file is provided.
    Streams output directly to output_features_file.
    """
    os.makedirs(os.path.dirname(output_features_file), exist_ok=True)
    print(f"\n=== Computing Name Features for {os.path.basename(candidates_file)} ===")
    
    # Filter active IDs to keep memory usage minimal (~100MB)
    active_ids = load_candidate_ids(candidates_file)
    name_lookup = load_name_lookup(normalized_file, filter_ids=active_ids)
    del active_ids
    
    gt_pairs = load_ground_truth(ground_truth_file) if ground_truth_file else None

    is_train = gt_pairs is not None
    feature_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "levenshtein_sim",
        "token_jaccard",
        "char_ngram_cos",
        "token_sort_ratio",
        "token_set_ratio",
        "exact_match"
    ]
    if is_train:
        feature_cols.append("label")

    pair_count = 0
    pos_count = 0

    with open(output_features_file, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(feature_cols)

        with open(candidates_file, "r", encoding="utf-8", errors="replace") as f_in:
            reader = csv.reader(f_in, delimiter="\t")
            header = next(reader, None)
            s1_idx = header.index("source1_entity_id")
            cand_idx = header.index("candidate_entity_id")

            for row in reader:
                if not row or len(row) <= max(s1_idx, cand_idx):
                    continue
                s1_id = row[s1_idx].strip()
                cand_id = row[cand_idx].strip()

                name1 = name_lookup.get(s1_id, "")
                name2 = name_lookup.get(cand_id, "")

                feats = compute_pair_features(name1, name2)
                row_out = [s1_id, cand_id, *feats]

                if is_train:
                    label = 1 if (s1_id, cand_id) in gt_pairs else 0
                    if label == 1:
                        pos_count += 1
                    row_out.append(label)

                writer.writerow(row_out)
                pair_count += 1

                if pair_count % 500000 == 0:
                    print(f"  Computed features for {pair_count:,} pairs...", flush=True)

    print(f"Successfully computed features for {pair_count:,} pairs.")
    if is_train:
        print(f"Positive matches in candidate set: {pos_count:,} / {len(gt_pairs):,} (Recall = {pos_count / max(len(gt_pairs), 1):.4f})")
    print(f"Output saved to: {output_features_file}")
    return output_features_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute name similarity features")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all", help="Which split to compute features for")
    args = parser.parse_args()

    train_cands = "data/processed/name_candidates_train.tsv"
    train_norm = "data/processed/name_normalized_train.tsv"
    train_gt = "dataset/train/train_ground_truth.tsv"
    train_feats = "data/processed/name_features_train.tsv"

    test_cands = "data/processed/name_candidates_test.tsv"
    test_norm = "data/processed/name_normalized_test.tsv"
    test_feats = "data/processed/name_features_test.tsv"

    if args.split in ("train", "all") and os.path.exists(train_cands):
        compute_name_features(train_cands, train_norm, train_feats, ground_truth_file=train_gt)

    if args.split in ("test", "all") and os.path.exists(test_cands):
        compute_name_features(test_cands, test_norm, test_feats, ground_truth_file=None)
