"""
ML Challenge 2026 — Comprehensive Submission Verification Script

Verifies all 15 validation rules requested for the final submission:
  1. Number of rows in test_source1.tsv == number of unique source1_entity_id rows in matching_results.tsv
  2. Every test Source 1 ID exists in matching_results.tsv.
  3. No extra Source 1 IDs exist.
  4. No duplicate source1_entity_id rows.
  5. Number of rows in candidate_pairs.tsv == number of unique Source 1 IDs in test_source1.tsv.
  6. Every test Source 1 ID exists in candidate_pairs.tsv.
  7. No duplicate candidate Source 1 rows.
  8. Every matched ID begins with S2- or S3-.
  9. Every matched ID exists in test_source2.tsv or test_source3.tsv.
  10. Every candidate ID exists in test_source2.tsv or test_source3.tsv.
  11. Every matched ID for a Source 1 entity is present in that entity's candidate_entity_ids.
  12. No duplicate IDs occur inside matched_entity_ids.
  13. No duplicate IDs occur inside candidate_entity_ids.
  14. All files are truly tab-separated.
  15. Exact headers verified:
      matching_results.tsv: source1_entity_id \\t matched_entity_ids
      candidate_pairs.tsv:  source1_entity_id \\t candidate_entity_ids
"""

import os
import sys

def verify(
    matching_path="output/matching_results.tsv",
    candidate_path="output/candidate_pairs.tsv",
    test_dir="dataset/test"
):
    print("=" * 75)
    print("  COMPREHENSIVE 15-POINT SUBMISSION VERIFICATION")
    print("=" * 75)

    all_passed = True
    def record(idx, desc, passed, detail=""):
        nonlocal all_passed
        status = "PASS" if passed else "FAIL"
        if not passed: all_passed = False
        print(f"[{status}] Rule {idx:2d}: {desc} {detail}")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # Load S1 IDs
    print("\nLoading test_source1.tsv...")
    s1_ids_list = []
    with open(s1_path, "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        for line in f:
            parts = line.split("\t")
            if parts and parts[0].strip():
                s1_ids_list.append(parts[0].strip())

    s1_ids_set = set(s1_ids_list)
    total_s1 = len(s1_ids_list)
    print(f"  test_source1: {total_s1:,} entities ({len(s1_ids_set):,} unique)")

    # Verify matching_results.tsv
    print("\nInspecting matching_results.tsv...")
    match_s1_list = []
    match_map = {}
    match_dup_s1 = set()
    match_intra_dup = set()
    match_bad_prefix = set()
    all_matched_ids = set()

    with open(matching_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        # Rule 14 & 15: header
        rule14_match = "\t" in header_line and not ("," in header_line and "\t" not in header_line)
        expected_match_header = ["source1_entity_id", "matched_entity_ids"]
        actual_match_header = [c.strip().lower() for c in header_line.rstrip("\r\n").split("\t")]
        rule15_match = (actual_match_header == expected_match_header)

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            s1_id, m_str = parts[0].strip(), parts[1].strip()
            match_s1_list.append(s1_id)
            if s1_id in match_map:
                match_dup_s1.add(s1_id)

            if m_str:
                m_ids = [m.strip() for m in m_str.split(",") if m.strip()]
                if len(m_ids) != len(set(m_ids)):
                    match_intra_dup.add(s1_id)
                for mid in m_ids:
                    if not mid.startswith(("S2-", "S3-")):
                        match_bad_prefix.add(mid)
                    all_matched_ids.add(mid)
                match_map[s1_id] = set(m_ids)
            else:
                match_map[s1_id] = set()

    # Verify candidate_pairs.tsv
    print("Inspecting candidate_pairs.tsv...")
    cand_s1_list = []
    cand_map = {}
    cand_dup_s1 = set()
    cand_intra_dup = set()
    cand_bad_prefix = set()
    all_cand_ids = set()

    with open(candidate_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        rule14_cand = "\t" in header_line and not ("," in header_line and "\t" not in header_line)
        expected_cand_header = ["source1_entity_id", "candidate_entity_ids"]
        actual_cand_header = [c.strip().lower() for c in header_line.rstrip("\r\n").split("\t")]
        rule15_cand = (actual_cand_header == expected_cand_header)

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            s1_id, c_str = parts[0].strip(), parts[1].strip()
            cand_s1_list.append(s1_id)
            if s1_id in cand_map:
                cand_dup_s1.add(s1_id)

            if c_str:
                c_ids = [c.strip() for c in c_str.split(",") if c.strip()]
                if len(c_ids) != len(set(c_ids)):
                    cand_intra_dup.add(s1_id)
                for cid in c_ids:
                    if not cid.startswith(("S2-", "S3-")):
                        cand_bad_prefix.add(cid)
                    all_cand_ids.add(cid)
                cand_map[s1_id] = set(c_ids)
            else:
                cand_map[s1_id] = set()

    # Rule checks
    print("\n--- RESULTS ---")
    # 1. Number of rows in test_source1 == number of unique S1 in matching_results
    record(1, "Number of rows in test_source1 == unique S1 rows in matching_results",
           total_s1 == len(match_map), f"({len(match_map):,} vs {total_s1:,})")

    # 2. Every test S1 ID exists in matching_results
    missing_in_match = s1_ids_set - set(match_map.keys())
    record(2, "Every test Source 1 ID exists in matching_results.tsv",
           len(missing_in_match) == 0, f"(Missing: {len(missing_in_match)})")

    # 3. No extra Source 1 IDs exist in matching_results
    extra_in_match = set(match_map.keys()) - s1_ids_set
    record(3, "No extra Source 1 IDs exist in matching_results.tsv",
           len(extra_in_match) == 0, f"(Extra: {len(extra_in_match)})")

    # 4. No duplicate source1_entity_id rows in matching_results
    record(4, "No duplicate source1_entity_id rows in matching_results.tsv",
           len(match_dup_s1) == 0, f"(Duplicates: {len(match_dup_s1)})")

    # 5. Number of rows in candidate_pairs == number of unique S1 in test_source1
    record(5, "Number of rows in candidate_pairs.tsv == unique S1 IDs in test_source1",
           total_s1 == len(cand_map), f"({len(cand_map):,} vs {total_s1:,})")

    # 6. Every test Source 1 ID exists in candidate_pairs.tsv
    missing_in_cand = s1_ids_set - set(cand_map.keys())
    record(6, "Every test Source 1 ID exists in candidate_pairs.tsv",
           len(missing_in_cand) == 0, f"(Missing: {len(missing_in_cand)})")

    # 7. No duplicate candidate Source 1 rows
    record(7, "No duplicate candidate Source 1 rows in candidate_pairs.tsv",
           len(cand_dup_s1) == 0, f"(Duplicates: {len(cand_dup_s1)})")

    # 8. Every matched ID begins with S2- or S3-
    record(8, "Every matched ID begins with S2- or S3-",
           len(match_bad_prefix) == 0, f"(Offenders: {len(match_bad_prefix)})")

    # 9 & 10. Check existence in S2 / S3
    print("\nVerifying matched and candidate IDs against test_source2.tsv and test_source3.tsv...")
    s23_valid_ids = set()
    for p in [s2_path, s3_path]:
        with open(p, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.split("\t", 1)
                if parts and parts[0].strip():
                    s23_valid_ids.add(parts[0].strip())
    print(f"  Loaded {len(s23_valid_ids):,} valid S2/S3 IDs from test set.")

    unknown_matched = all_matched_ids - s23_valid_ids
    record(9, "Every matched ID exists in test_source2.tsv or test_source3.tsv",
           len(unknown_matched) == 0, f"(Unknown: {len(unknown_matched)})")

    unknown_cand = all_cand_ids - s23_valid_ids
    record(10, "Every candidate ID exists in test_source2.tsv or test_source3.tsv",
           len(unknown_cand) == 0, f"(Unknown: {len(unknown_cand)})")

    # 11. Every matched ID for an S1 entity is present in candidate_entity_ids
    not_in_cands = set()
    for s1_id, m_set in match_map.items():
        c_set = cand_map.get(s1_id, set())
        diff = m_set - c_set
        if diff:
            not_in_cands.add(s1_id)
    record(11, "Every matched ID for a Source 1 entity is present in that entity's candidate_entity_ids",
           len(not_in_cands) == 0, f"(Violations: {len(not_in_cands)})")

    # 12. No duplicate IDs inside matched_entity_ids
    record(12, "No duplicate IDs occur inside matched_entity_ids",
           len(match_intra_dup) == 0, f"(Violations: {len(match_intra_dup)})")

    # 13. No duplicate IDs inside candidate_entity_ids
    record(13, "No duplicate IDs occur inside candidate_entity_ids",
           len(cand_intra_dup) == 0, f"(Violations: {len(cand_intra_dup)})")

    # 14. All files are truly tab-separated
    record(14, "All output files are truly tab-separated (.tsv)",
           rule14_match and rule14_cand)

    # 15. Exact headers verified
    record(15, "Exact headers verified on both files",
           rule15_match and rule15_cand)

    print("\n" + "=" * 75)
    if all_passed:
        print("  ALL 15 VERIFICATION RULES PASSED PERFECTLY!")
    else:
        print("  VERIFICATION FAILED ON ONE OR MORE RULES.")
    print("=" * 75)

    return all_passed

if __name__ == "__main__":
    verify()
