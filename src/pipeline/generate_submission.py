"""
Amazon ML Challenge 2026: Business Entity Resolution
Pipeline End-to-End Submission Generator

Processes the complete test dataset across all countries (India, US, France)
and generates:
  1. output/matching_results.tsv
  2. output/candidate_pairs.tsv

Adheres strictly to all competition rules:
  - Exactly one row per test Source 1 entity in exact source order.
  - Tab-separated TSV with required headers.
  - Empty matched_entity_ids / candidate_entity_ids for singletons / no-candidate entities.
  - Matched IDs are strictly a subset of candidate IDs.
  - Only valid S2-* and S3-* IDs, no S1 self-matches, no duplicate IDs.
  - Single-pass streaming over Source 2 and Source 3 files.
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.address_track.normalize_address import normalize_address
from src.name_track.normalize_name import normalize_name
from src.name_track.features_name import char_ngram_cosine
from src.name_track.blocking_name import STOPWORDS


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def parse_source_row(line: str):
    """Parse a source TSV row without dropping trailing empty fields."""
    parts = line.rstrip("\r\n").split("\t")
    if not parts:
        return None
    eid = parts[0].strip()
    if not eid:
        return None
    raw_name = parts[1] if len(parts) > 1 else ""
    raw_addr = parts[2] if len(parts) > 2 else ""
    country = parts[3].strip() if len(parts) > 3 else ""
    return eid, raw_name, raw_addr, country


def are_names_compatible(n1: str, n2: str) -> bool:
    """
    Check if two normalized business names are consistent/compatible.
    Returns True for exact match, acronym, substring, or high token/char overlap.
    Returns False for completely conflicting business names sharing an address.
    """
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())

    content1 = {t for t in tokens1 if len(t) >= 3 and t not in STOPWORDS}
    content2 = {t for t in tokens2 if len(t) >= 3 and t not in STOPWORDS}
    if content1 and content2 and (content1 & content2):
        return True

    if len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4]:
        return True

    if char_ngram_cosine(n1, n2, n=3) >= 0.38:
        return True

    return False


def _assert_test_universe(all_s1_ids, source1_path):
    """Refuse train/sampled universes so submissions cannot silently omit test IDs."""
    basename = os.path.basename(source1_path).lower()
    if "train" in basename:
        raise SystemExit(
            f"Refusing to generate a submission from train file: {source1_path}. "
            "Use dataset/test/test_source1.tsv."
        )
    n = len(all_s1_ids)
    n_unique = len(set(all_s1_ids))
    if n != n_unique:
        raise SystemExit(
            f"Source 1 file has duplicate entity_ids ({n} rows, {n_unique} unique)."
        )
    tr_count = sum(1 for eid in all_s1_ids if eid.startswith("S1-TR"))
    if n and tr_count / n > 0.5:
        raise SystemExit(
            f"Source 1 IDs look like the train set ({tr_count}/{n} start with S1-TR). "
            "Submission must be generated from test_source1.tsv."
        )
    return n


def write_complete_outputs(
    all_s1_ids,
    match_map,
    candidate_map,
    matching_output_path,
    candidate_output_path,
):
    """Write exactly one TSV row per Source 1 entity, including unmatched/empty rows."""
    os.makedirs(os.path.dirname(matching_output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(candidate_output_path) or ".", exist_ok=True)

    total_matches_global = 0
    total_candidates_global = 0
    singletons_global = 0
    matched_entities_global = 0
    seen = set()

    tmp_match = matching_output_path + ".tmp"
    tmp_cand = candidate_output_path + ".tmp"

    with open(tmp_match, "w", encoding="utf-8", newline="\n") as f_match, \
         open(tmp_cand, "w", encoding="utf-8", newline="\n") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in all_s1_ids:
            if s1_id in seen:
                continue
            seen.add(s1_id)

            matches_set = set(match_map.get(s1_id, set()))
            cands_set = set(candidate_map.get(s1_id, set()))

            def _valid_s23(ids):
                out = []
                used = set()
                for mid in ids:
                    if not mid or mid.startswith("S1-"):
                        continue
                    if not mid.startswith(("S2-", "S3-")):
                        continue
                    if mid in used:
                        continue
                    used.add(mid)
                    out.append(mid)
                return out

            sorted_matches = _valid_s23(sorted(matches_set))
            sorted_cands = _valid_s23(sorted(cands_set))
            cand_lookup = set(sorted_cands)
            for mid in sorted_matches:
                if mid not in cand_lookup:
                    sorted_cands.append(mid)
                    cand_lookup.add(mid)
            sorted_cands = sorted(sorted_cands)

            if sorted_matches:
                matched_entities_global += 1
                total_matches_global += len(sorted_matches)
                f_match.write(f"{s1_id}\t{','.join(sorted_matches)}\n")
            else:
                singletons_global += 1
                f_match.write(f"{s1_id}\t\n")

            if sorted_cands:
                total_candidates_global += len(sorted_cands)
                f_cand.write(f"{s1_id}\t{','.join(sorted_cands)}\n")
            else:
                f_cand.write(f"{s1_id}\t\n")

    os.replace(tmp_match, matching_output_path)
    os.replace(tmp_cand, candidate_output_path)

    written = len(seen)
    if written != len(all_s1_ids):
        raise SystemExit(
            f"Output writer dropped Source 1 rows: wrote {written}, expected {len(all_s1_ids)}."
        )

    return {
        "matched_entities": matched_entities_global,
        "singletons": singletons_global,
        "total_matches": total_matches_global,
        "total_candidates": total_candidates_global,
        "rows_written": written,
    }


def verify_row_completeness(all_s1_ids, matching_output_path, candidate_output_path):
    """Independent post-write check: every Source 1 ID has exactly one row in both files."""
    expected = list(all_s1_ids)

    def _ids(path):
        ids = []
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            if "\t" not in header:
                raise SystemExit(f"{path} is not tab-separated.")
            for line in f:
                s1, tab, _rest = line.partition("\t")
                if not tab:
                    continue
                ids.append(s1)
        return ids

    match_ids = _ids(matching_output_path)
    cand_ids = _ids(candidate_output_path)
    if match_ids != expected:
        missing = set(expected) - set(match_ids)
        extra = set(match_ids) - set(expected)
        raise SystemExit(
            f"matching_results.tsv completeness failed: "
            f"rows={len(match_ids)} expected={len(expected)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    if cand_ids != expected:
        missing = set(expected) - set(cand_ids)
        extra = set(cand_ids) - set(expected)
        raise SystemExit(
            f"candidate_pairs.tsv completeness failed: "
            f"rows={len(cand_ids)} expected={len(expected)} "
            f"missing={len(missing)} extra={len(extra)}"
        )


def run_official_validator(matching_output_path, candidate_output_path, test_dir):
    validator = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    cmd = [
        sys.executable,
        validator,
        "--matching", matching_output_path,
        "--candidate", candidate_output_path,
        "--test-dir", test_dir,
    ]
    print("\n[STEP 4] Running official submission validator...", flush=True)
    print("  " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit("Official validator did not PASS. Submission files were not accepted.")
    return result.returncode


def run_pipeline(
    test_source1_path: str = "dataset/test/test_source1.tsv",
    test_source2_path: str = "dataset/test/test_source2.tsv",
    test_source3_path: str = "dataset/test/test_source3.tsv",
    matching_output_path: str = "output/matching_results.tsv",
    candidate_output_path: str = "output/candidate_pairs.tsv",
    max_cands_per_s1: int = 20,
    run_validator: bool = True,
):
    test_source1_path = _resolve(test_source1_path)
    test_source2_path = _resolve(test_source2_path)
    test_source3_path = _resolve(test_source3_path)
    matching_output_path = _resolve(matching_output_path)
    candidate_output_path = _resolve(candidate_output_path)
    test_dir = os.path.dirname(test_source1_path)

    print("=" * 75, flush=True)
    print("  AMAZON ML CHALLENGE 2026 — SUBMISSION GENERATION PIPELINE", flush=True)
    print("=" * 75, flush=True)
    print(f"  Source 1: {test_source1_path}", flush=True)
    print(f"  Source 2: {test_source2_path}", flush=True)
    print(f"  Source 3: {test_source3_path}", flush=True)

    start_time = time.time()

    print("\n[STEP 1] Reading and indexing test_source1.tsv...", flush=True)
    t1_start = time.time()

    all_s1_ids = []
    s1_data = {}
    addr_index = defaultdict(list)
    name_index = defaultdict(list)
    country_s1_count = defaultdict(int)
    skipped_blank = 0

    with open(test_source1_path, "r", encoding="utf-8", errors="replace") as f:
        header = next(f)
        if "\t" not in header:
            raise SystemExit(f"test_source1 is not tab-separated: {test_source1_path}")
        for line in f:
            parsed = parse_source_row(line)
            if parsed is None:
                skipped_blank += 1
                continue
            eid, raw_name, raw_addr, country = parsed
            all_s1_ids.append(eid)
            country_s1_count[country or "<missing>"] += 1

            n_norm = normalize_name(raw_name, country)
            a_norm = normalize_address(raw_addr, country)
            s1_data[eid] = (n_norm, a_norm)

            if a_norm:
                addr_index[(country, a_norm)].append(eid)
            if n_norm and len(n_norm) >= 3:
                name_index[(country, n_norm)].append(eid)

            if len(all_s1_ids) % 500000 == 0:
                print(f"  Indexed {len(all_s1_ids):,} entities ({time.time()-t1_start:.1f}s)...", flush=True)

    total_s1 = _assert_test_universe(all_s1_ids, test_source1_path)
    print(f"  Total Source 1 entities indexed : {total_s1:,} in {time.time()-t1_start:.1f}s", flush=True)
    print(f"  Blank/header-like rows skipped  : {skipped_blank:,}", flush=True)
    print(f"  Unique address keys in index    : {len(addr_index):,}", flush=True)
    print(f"  Unique name keys in index       : {len(name_index):,}", flush=True)
    print("  Country distribution:", flush=True)
    for c, cnt in sorted(country_s1_count.items()):
        print(f"    - {c:<10}: {cnt:,} entities ({cnt/total_s1*100:.1f}%)", flush=True)

    candidate_map = defaultdict(set)
    match_map = defaultdict(set)

    for src_label, src_path in [("Source 2", test_source2_path), ("Source 3", test_source3_path)]:
        src_start = time.time()
        print(f"\n[STEP 2] Streaming {src_label} ({src_path})...", flush=True)
        records_processed = 0
        addr_matches_found = 0
        name_matches_found = 0

        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            header = next(f)
            for line in f:
                parsed = parse_source_row(line)
                if parsed is None:
                    continue
                records_processed += 1
                cid, raw_name, raw_addr, country = parsed
                if cid.startswith("S1-"):
                    continue

                c_addr_norm = normalize_address(raw_addr, country)
                c_name_norm = normalize_name(raw_name, country)

                addr_key = (country, c_addr_norm)
                name_key = (country, c_name_norm)

                if c_addr_norm and addr_key in addr_index:
                    matched_s1 = addr_index[addr_key]
                    for s1_id in matched_s1:
                        if len(candidate_map[s1_id]) < max_cands_per_s1:
                            candidate_map[s1_id].add(cid)

                        s1_name_norm, _ = s1_data[s1_id]
                        if are_names_compatible(s1_name_norm, c_name_norm):
                            match_map[s1_id].add(cid)
                            addr_matches_found += 1

                elif c_name_norm and name_key in name_index:
                    matched_s1 = name_index[name_key]
                    for s1_id in matched_s1:
                        _, s1_addr_norm = s1_data[s1_id]
                        s1_words = set(s1_addr_norm.split())
                        c_words = set(c_addr_norm.split())
                        common = {w for w in (s1_words & c_words) if len(w) >= 3 and w not in STOPWORDS}
                        if common:
                            if len(candidate_map[s1_id]) < max_cands_per_s1:
                                candidate_map[s1_id].add(cid)
                            if len(common) >= 2 or s1_addr_norm == c_addr_norm:
                                match_map[s1_id].add(cid)
                                name_matches_found += 1

                if records_processed % 1000000 == 0:
                    elapsed_src = time.time() - src_start
                    rate = records_processed / elapsed_src if elapsed_src else 0
                    print(
                        f"  Processed {records_processed:,} {src_label} records "
                        f"({elapsed_src:.1f}s, {rate:.0f} rec/s)...",
                        flush=True,
                    )

        print(f"  Completed {src_label}: {records_processed:,} records in {time.time()-src_start:.1f}s", flush=True)
        print(f"  Address match hits: {addr_matches_found:,} | Name match hits: {name_matches_found:,}", flush=True)

    del addr_index, name_index, s1_data
    gc.collect()

    print("\n[STEP 3] Writing final submission files for the FULL Source 1 universe...", flush=True)
    t_write = time.time()
    stats = write_complete_outputs(
        all_s1_ids, match_map, candidate_map, matching_output_path, candidate_output_path
    )
    verify_row_completeness(all_s1_ids, matching_output_path, candidate_output_path)
    print(f"  Wrote matching results : {matching_output_path} ({os.path.getsize(matching_output_path):,} bytes)", flush=True)
    print(f"  Wrote candidate pairs  : {candidate_output_path} ({os.path.getsize(candidate_output_path):,} bytes)", flush=True)
    print(f"  Writing completed in   : {time.time()-t_write:.1f}s", flush=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75, flush=True)
    print("  PIPELINE EXECUTION SUMMARY", flush=True)
    print("=" * 75, flush=True)
    print(f"  Total Source 1 entities processed : {total_s1:,}", flush=True)
    print(f"  Rows written (matching)            : {stats['rows_written']:,}", flush=True)
    print(f"  Entities with matches              : {stats['matched_entities']:,} ({stats['matched_entities']/total_s1*100:.2f}%)", flush=True)
    print(f"  Singletons (no match)              : {stats['singletons']:,} ({stats['singletons']/total_s1*100:.2f}%)", flush=True)
    print(f"  Total matched pairs                : {stats['total_matches']:,}", flush=True)
    print(f"  Total candidate pairs              : {stats['total_candidates']:,}", flush=True)
    print(f"  Average candidates per S1 entity   : {stats['total_candidates']/total_s1:.2f}", flush=True)
    print(f"  Total execution time               : {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    print("=" * 75, flush=True)

    if run_validator:
        run_official_validator(matching_output_path, candidate_output_path, test_dir)

    return {
        "total_s1": total_s1,
        "matched_entities": stats["matched_entities"],
        "singletons": stats["singletons"],
        "total_matches": stats["total_matches"],
        "total_candidates": stats["total_candidates"],
        "matching_path": matching_output_path,
        "candidate_path": candidate_output_path,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate complete test-set submission TSVs.")
    parser.add_argument("--source1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--source2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--source3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-cands", type=int, default=20)
    parser.add_argument("--skip-validator", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        test_source1_path=args.source1,
        test_source2_path=args.source2,
        test_source3_path=args.source3,
        matching_output_path=args.matching,
        candidate_output_path=args.candidate,
        max_cands_per_s1=args.max_cands,
        run_validator=not args.skip_validator,
    )


if __name__ == "__main__":
    main()
