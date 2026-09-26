"""
Step 6 — Address Blocking Evaluation Module
Amazon ML Challenge 2026: Business Entity Resolution (Person 2 - Address Domain Track)

Evaluates:
  1. Address Recall Ceiling:
     Formula: (# of true matching pairs captured by address blocking) / (# of total true matching pairs in ground truth)
     Answers: "What percentage of true Source-1 -> Source-2/Source-3 matches are present in the address candidate set?"

  2. Search Space Reduction Ratio:
     Formula: 1.0 - (# of candidate pairs generated) / (Total S1 entities * Total S2/S3 candidate entities)
     Measures how effectively blocking reduces the brute-force O(N*M) comparison space.

  3. S1 Entity-Level Coverage:
     - Full recall entities (all true matches captured)
     - Partial recall entities (some true matches captured)
     - Zero recall entities (no true matches captured)
     - Singletons (no matches in ground truth)

  4. Average and Distribution of Candidates per S1 Entity
  5. Per-Country Breakdown (US, India, France, and open sets)
"""

import os
import sys
import csv
from collections import defaultdict
import pandas as pd

# Set stdout encoding
sys.stdout.reconfigure(encoding='utf-8')


def evaluate_address_blocking(
    gt_file: str = "dataset/train/train_ground_truth.tsv",
    candidates_file: str = "data/processed/address_candidates_train.tsv",
    normalized_file: str = "data/processed/address_normalized_train.tsv"
) -> dict:
    print("=" * 80)
    print("      PERSON 2 — ADDRESS DOMAIN TRACK: BLOCKING EVALUATION REPORT")
    print("=" * 80)

    # 1. Load Ground Truth
    print(f"\n[1] Loading Ground Truth from {gt_file}...")
    gt_map = {}  # s1_id -> set of true matched ids
    total_true_pairs = 0
    singletons = 0

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
                m_set = {m.strip() for m in m_str.split(",") if m.strip()}
                gt_map[s1_id] = m_set
                total_true_pairs += len(m_set)
            else:
                gt_map[s1_id] = set()
                singletons += 1

    total_s1_gt = len(gt_map)
    print(f"  Total Source 1 entities in GT : {total_s1_gt:,}")
    print(f"  Total true match pairs        : {total_true_pairs:,}")
    print(f"  Singletons (zero matches)     : {singletons:,} ({singletons/total_s1_gt*100:.2f}%)")

    # 2. Load Normalized Records to count country sizes
    print(f"\n[2] Loading record metadata from {normalized_file}...")
    country_counts = defaultdict(lambda: {"S1": 0, "S23": 0})
    s1_country_map = {}
    with open(normalized_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        eid_idx = header.index("entity_id")
        c_idx = header.index("country")
        src_idx = header.index("source")
        for row in reader:
            if not row or len(row) <= max(eid_idx, c_idx, src_idx):
                continue
            eid = row[eid_idx].strip()
            c = row[c_idx].strip()
            src = row[src_idx].strip()
            if src == "S1":
                country_counts[c]["S1"] += 1
                s1_country_map[eid] = c
            else:
                country_counts[c]["S23"] += 1

    total_s1 = sum(v["S1"] for v in country_counts.values())
    total_s23 = sum(v["S23"] for v in country_counts.values())
    possible_comparisons = total_s1 * total_s23
    print(f"  Source 1 entities across data : {total_s1:,}")
    print(f"  Candidate pool (S2+S3)        : {total_s23:,}")
    print(f"  Brute-force comparison space  : {possible_comparisons:,} pairs")

    # 3. Load Candidates
    print(f"\n[3] Loading generated candidate pairs from {candidates_file}...")
    cands_by_s1 = defaultdict(set)
    total_candidate_pairs = 0

    # Support either challenge format (source1_entity_id \t candidate_entity_ids)
    # or long pair format (source1_entity_id \t candidate_entity_id \t ...)
    with open(candidates_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if "candidate_entity_ids" in header:
            # Challenge format
            s1_idx = header.index("source1_entity_id")
            cands_idx = header.index("candidate_entity_ids")
            for row in reader:
                if not row or len(row) <= s1_idx:
                    continue
                s1_id = row[s1_idx].strip()
                if len(row) > cands_idx and row[cands_idx].strip():
                    c_list = [c.strip() for c in row[cands_idx].split(",") if c.strip()]
                    cands_by_s1[s1_id].update(c_list)
                    total_candidate_pairs += len(c_list)
                else:
                    if s1_id not in cands_by_s1:
                        cands_by_s1[s1_id] = set()
        else:
            # Long pair format
            s1_idx = header.index("source1_entity_id")
            cand_idx = header.index("candidate_entity_id")
            for row in reader:
                if not row or len(row) <= max(s1_idx, cand_idx):
                    continue
                s1_id = row[s1_idx].strip()
                cand_id = row[cand_idx].strip()
                cands_by_s1[s1_id].add(cand_id)
                total_candidate_pairs += 1

    print(f"  Total candidate pairs loaded  : {total_candidate_pairs:,}")
    print(f"  S1 entities with candidate row: {len(cands_by_s1):,}")

    # 4. Evaluate Recall and Metrics
    true_pairs_captured = 0
    full_entity_recall = 0
    partial_entity_recall = 0
    zero_entity_recall = 0
    country_metrics = defaultdict(lambda: {"true_pairs": 0, "found_pairs": 0, "cands": 0, "s1_count": 0})

    for s1_id, true_set in gt_map.items():
        c_code = s1_country_map.get(s1_id, "UNKNOWN")
        country_metrics[c_code]["s1_count"] += 1
        cand_set = cands_by_s1.get(s1_id, set())
        country_metrics[c_code]["cands"] += len(cand_set)

        if not true_set:
            continue  # Singleton

        country_metrics[c_code]["true_pairs"] += len(true_set)
        captured = true_set & cand_set
        num_cap = len(captured)
        true_pairs_captured += num_cap
        country_metrics[c_code]["found_pairs"] += num_cap

        if num_cap == len(true_set):
            full_entity_recall += 1
        elif num_cap > 0:
            partial_entity_recall += 1
        else:
            zero_entity_recall += 1

    recall_ceiling = true_pairs_captured / max(total_true_pairs, 1)
    reduction_ratio = 1.0 - (total_candidate_pairs / max(possible_comparisons, 1))
    avg_cands_per_s1 = total_candidate_pairs / max(total_s1, 1)

    print("\n" + "=" * 80)
    print("                    EVALUATION RESULTS & FORMULAS")
    print("=" * 80)
    print(f"\n1. ADDRESS RECALL CEILING:")
    print(f"   Formula: Captured True Pairs / Total True Pairs = {true_pairs_captured:,} / {total_true_pairs:,}")
    print(f"   --> RECALL CEILING = {recall_ceiling:.6f} ({recall_ceiling*100:.2f}%)")

    print(f"\n2. SEARCH SPACE REDUCTION RATIO:")
    print(f"   Formula: 1.0 - (Candidate Pairs / (Total S1 * Total S2/S3))")
    print(f"          = 1.0 - ({total_candidate_pairs:,} / {possible_comparisons:,})")
    print(f"   --> REDUCTION RATIO = {reduction_ratio:.8f} ({reduction_ratio*100:.6f}%)")

    print(f"\n3. CANDIDATE SET VOLUME:")
    print(f"   Total Candidate Pairs Generated : {total_candidate_pairs:,}")
    print(f"   Average Candidates per S1 Entity: {avg_cands_per_s1:.2f}")

    print(f"\n4. ENTITY-LEVEL RECALL BREAKDOWN (Non-singletons = {total_s1_gt - singletons:,}):")
    non_singletons = total_s1_gt - singletons
    print(f"   Full recall entities (100% matches captured): {full_entity_recall:,} ({full_entity_recall/non_singletons*100:.2f}%)")
    print(f"   Partial recall entities                      : {partial_entity_recall:,} ({partial_entity_recall/non_singletons*100:.2f}%)")
    print(f"   Zero recall entities (missed entirely)       : {zero_entity_recall:,} ({zero_entity_recall/non_singletons*100:.2f}%)")

    print(f"\n5. PER-COUNTRY BREAKDOWN:")
    print(f"   {'Country':<15} | {'S1 Entities':<12} | {'True Pairs':<12} | {'Captured':<10} | {'Recall':<10} | {'Candidates':<12} | {'Avg Cands/S1'}")
    print("   " + "-" * 88)
    for c_code in sorted(country_metrics.keys()):
        m = country_metrics[c_code]
        c_recall = m["found_pairs"] / max(m["true_pairs"], 1) * 100 if m["true_pairs"] > 0 else 100.0
        c_avg = m["cands"] / max(m["s1_count"], 1)
        print(f"   {c_code:<15} | {m['s1_count']:<12,d} | {m['true_pairs']:<12,d} | {m['found_pairs']:<10,d} | {c_recall:>6.2f}%    | {m['cands']:<12,d} | {c_avg:>7.2f}")

    print("=" * 80 + "\n")

    return {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "total_true_pairs": total_true_pairs,
        "true_pairs_captured": true_pairs_captured,
        "total_candidate_pairs": total_candidate_pairs,
        "avg_cands_per_s1": avg_cands_per_s1,
        "full_entity_recall": full_entity_recall,
        "partial_entity_recall": partial_entity_recall,
        "zero_entity_recall": zero_entity_recall
    }


if __name__ == "__main__":
    evaluate_address_blocking()
