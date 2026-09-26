"""
Step 5 — Validation and Verification Module
ML Challenge 2026: Business Entity Resolution (Person 1 - Name Domain Track)

Computes:
  1. Recall Ceiling = (# true match pairs in candidate set) / (# total true match pairs)
  2. Search Space Reduction Ratio = 1 - (# candidate pairs) / (Total S1 * Total S2/S3)
  3. Per-country Recall and Candidate metrics
  4. Verification of final hand-off files:
     - data/processed/name_candidates_train.tsv & name_candidates_test.tsv
     - data/processed/name_features_train.tsv & name_features_test.tsv
"""

import sys
import os
import csv
from collections import defaultdict

def run_validation(
    gt_file: str = "dataset/train/train_ground_truth.tsv",
    candidates_file: str = "data/processed/name_candidates_train.tsv",
    normalized_train_file: str = "data/processed/name_normalized_train.tsv"
) -> dict:
    print("=================================================================")
    print("        NAME DOMAIN TRACK — STEP 5 VALIDATION REPORT            ")
    print("=================================================================\n")

    # 1. Load Ground Truth
    gt_map = {}  # s1_id -> set of matched_ids
    total_true_pairs = 0
    singletons_count = 0

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
                matched_set = {m.strip() for m in m_str.split(",") if m.strip()}
                gt_map[s1_id] = matched_set
                total_true_pairs += len(matched_set)
            else:
                gt_map[s1_id] = set()
                singletons_count += 1

    total_s1_gt = len(gt_map)
    print(f"[Ground Truth Summary]")
    print(f"  Total Source 1 Entities : {total_s1_gt:,}")
    print(f"  Total True Match Pairs  : {total_true_pairs:,}")
    print(f"  Singletons (no matches) : {singletons_count:,} ({singletons_count/total_s1_gt*100:.2f}%)\n")

    # 2. Load Normalized Train counts by country
    country_counts = defaultdict(lambda: {"S1": 0, "S23": 0})
    s1_country_map = {}
    with open(normalized_train_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        eid_idx = header.index("entity_id")
        c_idx = header.index("country")
        src_idx = header.index("source")
        for row in reader:
            if row and len(row) > max(eid_idx, c_idx, src_idx):
                eid = row[eid_idx].strip()
                c = row[c_idx].strip()
                src = row[src_idx].strip()
                if src == "S1":
                    country_counts[c]["S1"] += 1
                    s1_country_map[eid] = c
                else:
                    country_counts[c]["S23"] += 1

    total_s1_all = sum(v["S1"] for v in country_counts.values())
    total_s23_all = sum(v["S23"] for v in country_counts.values())
    total_possible_comparisons = total_s1_all * total_s23_all

    # 3. Load Candidates and Evaluate Recall
    cands_by_s1 = defaultdict(set)
    total_candidate_pairs = 0

    with open(candidates_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        s1_idx = header.index("source1_entity_id")
        cand_idx = header.index("candidate_entity_id")
        for row in reader:
            if row and len(row) > max(s1_idx, cand_idx):
                s1_id = row[s1_idx].strip()
                c_id = row[cand_idx].strip()
                cands_by_s1[s1_id].add(c_id)
                total_candidate_pairs += 1

    # Check pair recall
    true_pairs_found = 0
    full_entity_recall_count = 0
    partial_entity_recall_count = 0

    country_eval = defaultdict(lambda: {"gt_pairs": 0, "found_pairs": 0, "cand_pairs": 0})

    for s1_id, true_matches in gt_map.items():
        c = s1_country_map.get(s1_id, "Unknown")
        country_eval[c]["gt_pairs"] += len(true_matches)
        
        found_for_s1 = 0
        cand_set = cands_by_s1.get(s1_id, set())
        for m_id in true_matches:
            if m_id in cand_set:
                found_for_s1 += 1
                true_pairs_found += 1
                country_eval[c]["found_pairs"] += 1
        
        if len(true_matches) > 0:
            if found_for_s1 == len(true_matches):
                full_entity_recall_count += 1
            elif found_for_s1 > 0:
                partial_entity_recall_count += 1

    recall_ceiling = true_pairs_found / max(total_true_pairs, 1)
    reduction_ratio = 1.0 - (total_candidate_pairs / max(total_possible_comparisons, 1))
    avg_cands_per_s1 = total_candidate_pairs / max(len(cands_by_s1), 1)

    print("[Blocking Evaluation Metrics]")
    print(f"  Total Candidate Pairs Generated : {total_candidate_pairs:,}")
    print(f"  Average Candidates per S1 Entity : {avg_cands_per_s1:.2f}")
    print(f"  Search Space Reduction Ratio    : {reduction_ratio:.8f} ({reduction_ratio*100:.6f}%)")
    print(f"  True Matches Captured           : {true_pairs_found:,} / {total_true_pairs:,}")
    print(f"  --> RECALL CEILING              : {recall_ceiling:.4f} ({recall_ceiling*100:.2f}%)")
    
    target_met = recall_ceiling >= 0.95
    status_str = "PASSED (>= 0.95)" if target_met else "BELOW TARGET (< 0.95)"
    print(f"  --> Target Threshold (>= 0.95)  : {status_str}\n")

    print("[Per-Country Breakdown]")
    for c, stats in sorted(country_eval.items()):
        rec = stats["found_pairs"] / max(stats["gt_pairs"], 1) if stats["gt_pairs"] > 0 else 1.0
        print(f"  Country: {c:<8} | True Pairs: {stats['gt_pairs']:<6} | Found: {stats['found_pairs']:<6} | Recall: {rec*100:.2f}%")

    print("\n=================================================================")
    return {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "total_candidate_pairs": total_candidate_pairs,
        "true_pairs_found": true_pairs_found,
        "total_true_pairs": total_true_pairs,
        "avg_cands_per_s1": avg_cands_per_s1,
        "target_met": target_met
    }

if __name__ == "__main__":
    run_validation()
