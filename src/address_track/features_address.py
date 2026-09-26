"""
Step 4 — Address Similarity Features Module
Amazon ML Challenge 2026: Business Entity Resolution (Person 2 - Address Domain Track)

Computes 13 address similarity features for every candidate pair:
  1. levenshtein_sim      : Normalized Levenshtein similarity (1 - edit_dist / max_len)
  2. token_jaccard        : Word token Jaccard similarity (set intersection / union)
  3. tfidf_cosine         : Character 3-gram cosine similarity on address text
  4. exact_pincode_match  : Boolean flag (1 if both non-empty and pin1 == pin2, else 0)
  5. exact_city_match     : Boolean flag (1 if both non-empty and city1 == city2, else 0)
  6. exact_country_match  : Boolean flag (1 if both non-empty and country1 == country2, else 0)
  7. exact_address_match  : Boolean flag (1 if address_norm1 == address_norm2, else 0)
  8. char_ngram_cos       : Character 3-gram cosine similarity (fast n-gram overlap)
  9. token_overlap_count  : Integer count of shared tokens
 10. address_len_ratio    : Ratio of string lengths (min_len / max_len)
 11. digit_overlap_ratio  : Jaccard similarity of extracted numbers/digits
 12. house_number_match   : Boolean flag (1 if both have leading street/house numbers and they match)
 13. landmark_sim         : Token Jaccard similarity between extracted landmark fields

Schema:
  source1_entity_id, candidate_entity_id,
  levenshtein_sim, token_jaccard, tfidf_cosine, exact_pincode_match,
  exact_city_match, exact_country_match, exact_address_match, char_ngram_cos,
  token_overlap_count, address_len_ratio, digit_overlap_ratio,
  house_number_match, landmark_sim [, label]

Label column is populated for train split via join against train_ground_truth.tsv.
"""

import os
import sys
import csv
import math
import re
import argparse
from collections import Counter
import pandas as pd


def fast_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein edit distance with O(min(len1, len2)) memory."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    current = list(range(len1 + 1))
    for i2, c2 in enumerate(s2):
        new_row = [i2 + 1] * (len1 + 1)
        for i1, c1 in enumerate(s1):
            cost = 0 if c1 == c2 else 1
            new_row[i1 + 1] = min(
                current[i1 + 1] + 1,      # deletion
                new_row[i1] + 1,          # insertion
                current[i1] + cost        # substitution
            )
        current = new_row
    return current[len1]


def compute_char_ngrams(s: str, n: int = 3) -> list[str]:
    """Generate character n-grams from a string."""
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s) - n + 1)]


def char_ngram_cosine_sim(s1: str, s2: str, n: int = 3) -> float:
    """Compute cosine similarity of character n-gram frequency vectors."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    ng1 = compute_char_ngrams(s1, n)
    ng2 = compute_char_ngrams(s2, n)
    if not ng1 or not ng2:
        return 0.0
    c1 = Counter(ng1)
    c2 = Counter(ng2)
    dot = sum(c1[k] * c2[k] for k in c1 if k in c2)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot / (norm1 * norm2))


def extract_numbers(s: str) -> set[str]:
    """Extract all numerical tokens from a string."""
    if not s:
        return set()
    return set(re.findall(r'\b\d+\b', s))


def extract_house_number(s: str) -> str:
    """Extract leading street or building number if present."""
    if not s:
        return ""
    m = re.match(r'^(?:unit\s+|apt\s+)?(\d+[a-zA-Z]?)', s.strip().lower())
    return m.group(1) if m else ""


def compute_address_pair_features(rec1: dict, rec2: dict, cache: dict | None = None) -> list:
    """
    Computes all 13 address similarity features for a pair of records.
    Uses optional cache for (addr1, addr2) text features.
    """
    a1 = rec1.get("address_norm", "")
    a2 = rec2.get("address_norm", "")
    pin1 = rec1.get("pincode", "")
    pin2 = rec2.get("pincode", "")
    city1 = rec1.get("city_norm", "")
    city2 = rec2.get("city_norm", "")
    c1 = rec1.get("country", "")
    c2 = rec2.get("country", "")
    lm1 = rec1.get("landmark", "")
    lm2 = rec2.get("landmark", "")

    # Component matching flags
    exact_pin = 1 if (pin1 and pin2 and pin1 == pin2) else 0
    exact_city = 1 if (city1 and city2 and city1.lower() == city2.lower()) else 0
    exact_country = 1 if (c1 and c2 and c1.upper() == c2.upper()) else 0
    exact_addr = 1 if (a1 and a2 and a1 == a2) else 0

    # House / building number match
    h1 = extract_house_number(a1)
    h2 = extract_house_number(a2)
    house_num_match = 1 if (h1 and h2 and h1 == h2) else 0

    # Landmark similarity
    if lm1 and lm2:
        tokens_lm1 = set(lm1.split())
        tokens_lm2 = set(lm2.split())
        u = tokens_lm1 | tokens_lm2
        lm_sim = round(len(tokens_lm1 & tokens_lm2) / len(u), 4) if u else 0.0
    else:
        lm_sim = 0.0

    # Cache lookup for text-level similarity
    cache_key = (a1, a2) if a1 <= a2 else (a2, a1)
    if cache is not None and cache_key in cache:
        lev_sim, jaccard, cos_sim, n_cos, tok_overlap, len_ratio, num_jaccard = cache[cache_key]
    else:
        if not a1 or not a2:
            lev_sim, jaccard, cos_sim, n_cos = 0.0, 0.0, 0.0, 0.0
            tok_overlap, len_ratio, num_jaccard = 0, 0.0, 0.0
        elif a1 == a2:
            lev_sim = 1.0
            jaccard = 1.0
            cos_sim = 1.0
            n_cos = 1.0
            tok_overlap = len(a1.split())
            len_ratio = 1.0
            nums = extract_numbers(a1)
            num_jaccard = 1.0 if nums else 0.0
        else:
            # 1. Levenshtein similarity
            max_len = max(len(a1), len(a2))
            dist = fast_levenshtein_distance(a1, a2)
            lev_sim = round(1.0 - (dist / max_len), 4)

            # 2. Token Jaccard & overlap
            t1 = set(a1.split())
            t2 = set(a2.split())
            intersection = t1 & t2
            union = t1 | t2
            jaccard = round(len(intersection) / len(union), 4) if union else 0.0
            tok_overlap = len(intersection)

            # 3. Char n-gram cosine
            cos_sim = round(char_ngram_cosine_sim(a1, a2, n=3), 4)
            n_cos = cos_sim

            # 4. Length ratio
            min_len = min(len(a1), len(a2))
            len_ratio = round(min_len / max_len, 4) if max_len > 0 else 0.0

            # 5. Digit overlap ratio
            nums1 = extract_numbers(a1)
            nums2 = extract_numbers(a2)
            if nums1 and nums2:
                num_u = nums1 | nums2
                num_jaccard = round(len(nums1 & nums2) / len(num_u), 4)
            elif not nums1 and not nums2:
                num_jaccard = 0.0
            else:
                num_jaccard = 0.0

        if cache is not None:
            cache[cache_key] = (lev_sim, jaccard, cos_sim, n_cos, tok_overlap, len_ratio, num_jaccard)

    return [
        lev_sim,
        jaccard,
        cos_sim,
        exact_pin,
        exact_city,
        exact_country,
        exact_addr,
        n_cos,
        tok_overlap,
        len_ratio,
        num_jaccard,
        house_num_match,
        lm_sim
    ]


def load_ground_truth(gt_file: str) -> set[tuple[str, str]]:
    """Load positive match pairs from ground truth TSV."""
    gt_pairs = set()
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
            if m_str:
                for mid in m_str.split(","):
                    mid = mid.strip()
                    if mid:
                        gt_pairs.add((s1_id, mid))
    return gt_pairs


def compute_address_features(
    candidates_file: str = "data/processed/address_candidates_train.tsv",
    normalized_file: str = "data/processed/address_normalized_train.tsv",
    output_features_file: str = "data/processed/address_features.tsv",
    ground_truth_file: str | None = "dataset/train/train_ground_truth.tsv"
) -> str:
    """
    Computes address similarity features for all candidate pairs and writes TSV output.
    If ground_truth_file is provided, appends 'label' column (1 or 0).
    """
    os.makedirs(os.path.dirname(output_features_file), exist_ok=True)
    print(f"\n=== Computing Address Features for {os.path.basename(candidates_file)} ===")

    # 1. Load normalized records into memory index by entity_id
    print(f"Loading normalized records from {normalized_file}...")
    rec_lookup = {}
    with open(normalized_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                rec_lookup[eid] = row
    print(f"Loaded {len(rec_lookup):,} normalized records.")

    # 2. Load Ground Truth if available (train mode)
    gt_pairs = load_ground_truth(ground_truth_file) if ground_truth_file and os.path.exists(ground_truth_file) else None
    is_train = gt_pairs is not None
    if is_train:
        print(f"Loaded {len(gt_pairs):,} ground truth positive pairs.")

    feature_cols = [
        "source1_entity_id",
        "candidate_entity_id",
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
        "landmark_sim"
    ]
    if is_train:
        feature_cols.append("label")

    pair_count = 0
    pos_count = 0
    cache = {}

    print(f"Streaming candidate features to {output_features_file}...")
    with open(output_features_file, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(feature_cols)

        # Inspect candidate file header to determine format
        with open(candidates_file, "r", encoding="utf-8", errors="replace") as f_in:
            reader = csv.reader(f_in, delimiter="\t")
            header = next(reader, None)

            if "candidate_entity_ids" in header:
                # Challenge format: source1_entity_id \t candidate_entity_ids
                s1_idx = header.index("source1_entity_id")
                cands_idx = header.index("candidate_entity_ids")
                for row in reader:
                    if not row or len(row) <= s1_idx:
                        continue
                    s1_id = row[s1_idx].strip()
                    if len(row) <= cands_idx or not row[cands_idx].strip():
                        continue
                    rec1 = rec_lookup.get(s1_id, {})
                    cands = [c.strip() for c in row[cands_idx].split(",") if c.strip()]
                    for cand_id in cands:
                        rec2 = rec_lookup.get(cand_id, {})
                        feats = compute_address_pair_features(rec1, rec2, cache=cache)
                        row_out = [s1_id, cand_id, *feats]
                        if is_train:
                            lbl = 1 if (s1_id, cand_id) in gt_pairs else 0
                            if lbl == 1:
                                pos_count += 1
                            row_out.append(lbl)
                        writer.writerow(row_out)
                        pair_count += 1
                        if pair_count % 100000 == 0:
                            print(f"  Processed {pair_count:,} pairs...", flush=True)
            else:
                # Long format: source1_entity_id \t candidate_entity_id ...
                s1_idx = header.index("source1_entity_id")
                cand_idx = header.index("candidate_entity_id")
                for row in reader:
                    if not row or len(row) <= max(s1_idx, cand_idx):
                        continue
                    s1_id = row[s1_idx].strip()
                    cand_id = row[cand_idx].strip()
                    rec1 = rec_lookup.get(s1_id, {})
                    rec2 = rec_lookup.get(cand_id, {})

                    feats = compute_address_pair_features(rec1, rec2, cache=cache)
                    row_out = [s1_id, cand_id, *feats]
                    if is_train:
                        lbl = 1 if (s1_id, cand_id) in gt_pairs else 0
                        if lbl == 1:
                            pos_count += 1
                        row_out.append(lbl)

                    writer.writerow(row_out)
                    pair_count += 1
                    if pair_count % 100000 == 0:
                        print(f"  Processed {pair_count:,} pairs...", flush=True)

    print(f"\nFinished computing features for {pair_count:,} pairs.")
    if is_train and gt_pairs:
        recall = pos_count / max(len(gt_pairs), 1)
        print(f"Ground Truth Recall in feature set: {pos_count:,} / {len(gt_pairs):,} ({recall*100:.2f}%)")
    print(f"Features saved to: {output_features_file}")

    # Also save copy to address_features_train.tsv if outputting address_features.tsv
    alt_out = "data/processed/address_features_train.tsv" if is_train and output_features_file != "data/processed/address_features_train.tsv" else None
    if alt_out:
        import shutil
        shutil.copyfile(output_features_file, alt_out)
        print(f"Also saved synchronized copy to: {alt_out}")

    return output_features_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute address similarity features")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all", help="Which split to compute")
    args = parser.parse_args()

    train_cands = "data/processed/address_candidates_train.tsv"
    train_norm = "data/processed/address_normalized_train.tsv"
    train_gt = "dataset/train/train_ground_truth.tsv"
    train_feats = "data/processed/address_features.tsv"

    test_cands = "data/processed/address_candidates_test.tsv"
    test_norm = "data/processed/address_normalized_test.tsv"
    test_feats = "data/processed/address_features_test.tsv"

    if args.split in ("train", "all") and os.path.exists(train_cands):
        compute_address_features(
            candidates_file=train_cands,
            normalized_file=train_norm,
            output_features_file=train_feats,
            ground_truth_file=train_gt
        )

    if args.split in ("test", "all") and os.path.exists(test_cands):
        compute_address_features(
            candidates_file=test_cands,
            normalized_file=test_norm,
            output_features_file=test_feats,
            ground_truth_file=None
        )
