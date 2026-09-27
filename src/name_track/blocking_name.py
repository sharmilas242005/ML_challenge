"""
Step 3 — Name-Based Blocking Module (Optimized for Multi-Million Row Scale)
ML Challenge 2026: Business Entity Resolution (Person 1 - Name Domain Track)

Implements three blocking passes and unions their candidate pairs:
  Pass A) Exact prefix key: (country, prefix of first N characters)
  Pass B) Phonetic code: (country, metaphone code of distinctive name token)
  Pass C) Token inverted index on distinctive content tokens

Deduplicates on (source1_entity_id, candidate_entity_id).
Produces long candidate format:
  source1_entity_id, candidate_entity_id, source, block_reason

Scale-Ready Design:
- Processes country-by-country (US, India, France, and open labels).
- Memory-compact parallel array storage (indices, not heavy dicts).
- Capping on oversized buckets to prevent combinatorial explosion.
- Strict candidate-per-entity ranking and capping.
- Line-by-line disk streaming of candidate pairs.
"""

import sys
import os
import csv
import gc
from collections import defaultdict
try:
    import jellyfish
except ImportError:
    jellyfish = None

STOPWORDS = {
    'and', 'the', 'for', 'with', 'from', 'inc', 'corp', 'corporation',
    'ltd', 'limited', 'pvt', 'private', 'llc', 'llp', 'pllc', 'co', 'company',
    'sarl', 'sas', 'sasu', 'sci', 'eurl', 'sa', 'cie', 'fils', 'freres',
    'group', 'services', 'service', 'solutions', 'solution', 'enterprises',
    'enterprise', 'industries', 'industry', 'trading', 'consulting', 'india',
    'france', 'usa', 'center', 'centre', 'international', 'association',
    'de', 'du', 'des', 'la', 'le', 'les', 'et', 'en', 'un', 'une'
}


def get_distinctive_tokens(name_norm: str) -> list[str]:
    """Extract informative content tokens from normalized name."""
    if not name_norm:
        return []
    tokens = name_norm.split()
    return [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]


def get_phonetic_code(token: str) -> str:
    """Compute metaphone phonetic code for a token."""
    if not token or len(token) < 2:
        return ""
    try:
        code = jellyfish.metaphone(token)
        return code if code else ""
    except Exception:
        return ""


def block_country_shard(
    s1_ids: list[str],
    s1_names: list[str],
    cand_ids: list[str],
    cand_names: list[str],
    cand_sources: list[str],
    writer: csv.writer,
    prefix_len: int = 5,
    max_cands_per_s1: int = 25,
    max_bucket_size: int = 1500
) -> tuple[int, int]:
    """
    Run 3-pass union blocking for a single country shard.
    Streams candidate pairs directly into writer.
    Returns (num_pairs_generated, num_capped_buckets).
    """
    num_pairs = 0
    capped_buckets = 0

    # Build indices over candidate set
    prefix_index = defaultdict(list)
    token_index = defaultdict(list)
    phonetic_index = defaultdict(list)

    for idx, name in enumerate(cand_names):
        if not name:
            continue
        # Pass A: prefix
        if len(name) >= prefix_len:
            pref = name[:prefix_len]
            prefix_index[pref].append(idx)
        elif len(name) >= 3:
            prefix_index[name].append(idx)

        # Pass B & C: tokens and phonetics
        tokens = get_distinctive_tokens(name)
        for t in tokens:
            token_index[t].append(idx)
            ph = get_phonetic_code(t)
            if ph:
                phonetic_index[ph].append(idx)

    # For each S1 record, collect candidate matches across all 3 passes
    for s1_id, s1_name in zip(s1_ids, s1_names):
        if not s1_name:
            continue

        cands_for_s1 = defaultdict(list)

        # Pass A: exact prefix
        s1_pref = s1_name[:prefix_len] if len(s1_name) >= prefix_len else s1_name
        if s1_pref in prefix_index:
            bucket = prefix_index[s1_pref]
            if len(bucket) > max_bucket_size:
                bucket = bucket[:max_bucket_size]
                capped_buckets += 1
            for c_idx in bucket:
                cands_for_s1[c_idx].append("exact_key")

        # Pass B & C: tokens & phonetics
        s1_tokens = get_distinctive_tokens(s1_name)
        for t in s1_tokens:
            if t in token_index:
                bucket = token_index[t]
                if len(bucket) > max_bucket_size:
                    bucket = bucket[:max_bucket_size]
                    capped_buckets += 1
                for c_idx in bucket:
                    cands_for_s1[c_idx].append("token_ngram")

            ph = get_phonetic_code(t)
            if ph and ph in phonetic_index:
                bucket = phonetic_index[ph]
                if len(bucket) > max_bucket_size:
                    bucket = bucket[:max_bucket_size]
                    capped_buckets += 1
                for c_idx in bucket:
                    cands_for_s1[c_idx].append("phonetic")

        # Sort candidates by number of matched passes descending, then cap
        sorted_cands = sorted(
            cands_for_s1.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )[:max_cands_per_s1]

        for c_idx, reasons in sorted_cands:
            cand_id = cand_ids[c_idx]
            cand_src = cand_sources[c_idx]
            reason_str = "+".join(sorted(set(reasons)))
            writer.writerow([s1_id, cand_id, cand_src, reason_str])
            num_pairs += 1

    return num_pairs, capped_buckets


def block_names(
    normalized_file: str,
    output_candidates_file: str,
    prefix_len: int = 5,
    max_cands_per_s1: int = 25,
    max_bucket_size: int = 1500
) -> dict:
    """
    Main blocking function for both Train and Test sets.
    Streams input country by country to keep memory bounded at ~200MB.
    """
    os.makedirs(os.path.dirname(output_candidates_file), exist_ok=True)
    print(f"\n=== Running Name Blocking on {normalized_file} ===")
    
    # First pass: find all unique countries in the file
    countries = set()
    total_rows = 0
    with open(normalized_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        country_idx = header.index("country")
        for row in reader:
            if row and len(row) > country_idx:
                c = row[country_idx].strip()
                if c:
                    countries.add(c)
                total_rows += 1

    print(f"Total rows in {normalized_file}: {total_rows:,}. Countries: {sorted(countries)}")

    total_pairs = 0
    total_capped = 0
    total_s1 = 0
    total_cands = 0

    with open(output_candidates_file, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "source", "block_reason"])

        for country in sorted(countries):
            print(f"\nLoading records for country shard: '{country}'...")
            s1_ids, s1_names = [], []
            cand_ids, cand_names, cand_sources = [], [], []

            with open(normalized_file, "r", encoding="utf-8", errors="replace") as f_in:
                reader = csv.reader(f_in, delimiter="\t")
                header = next(reader, None)
                eid_idx = header.index("entity_id")
                name_idx = header.index("name_norm")
                c_idx = header.index("country")
                src_idx = header.index("source")

                for row in reader:
                    if not row or len(row) <= max(eid_idx, name_idx, c_idx, src_idx):
                        continue
                    if row[c_idx].strip() == country:
                        eid = row[eid_idx].strip()
                        name = row[name_idx].strip()
                        src = row[src_idx].strip()
                        if src == "S1":
                            s1_ids.append(eid)
                            s1_names.append(name)
                        else:
                            cand_ids.append(eid)
                            cand_names.append(name)
                            cand_sources.append(src)

            n_s1 = len(s1_ids)
            n_cands = len(cand_ids)
            total_s1 += n_s1
            total_cands += n_cands
            print(f"  Shard '{country}': {n_s1:,} S1 entities, {n_cands:,} Candidate entities.")

            if n_s1 > 0 and n_cands > 0:
                pairs_count, capped_count = block_country_shard(
                    s1_ids, s1_names,
                    cand_ids, cand_names, cand_sources,
                    writer,
                    prefix_len=prefix_len,
                    max_cands_per_s1=max_cands_per_s1,
                    max_bucket_size=max_bucket_size
                )
                total_pairs += pairs_count
                total_capped += capped_count
                print(f"  Completed '{country}': {pairs_count:,} candidate pairs generated, {capped_count:,} buckets capped.")

            # Explicit memory cleanup between country shards
            del s1_ids, s1_names, cand_ids, cand_names, cand_sources
            gc.collect()

    possible_comps = max(total_s1 * total_cands, 1)
    reduction_ratio = 1.0 - (total_pairs / possible_comps)

    print(f"\n--- Blocking Summary for {os.path.basename(output_candidates_file)} ---")
    print(f"Total candidate pairs: {total_pairs:,}")
    print(f"Total buckets capped: {total_capped:,}")
    print(f"Search space reduction ratio: {reduction_ratio:.8f} ({reduction_ratio*100:.6f}%)")
    print(f"Output saved to: {output_candidates_file}")

    return {
        "candidate_file": output_candidates_file,
        "total_pairs": total_pairs,
        "total_s1": total_s1,
        "total_cands": total_cands,
        "capped_buckets": total_capped,
        "reduction_ratio": reduction_ratio
    }


if __name__ == "__main__":
    train_norm = "data/processed/name_normalized_train.tsv"
    train_cands = "data/processed/name_candidates_train.tsv"
    test_norm = "data/processed/name_normalized_test.tsv"
    test_cands = "data/processed/name_candidates_test.tsv"

    if os.path.exists(train_norm):
        block_names(train_norm, train_cands)

    if os.path.exists(test_norm):
        block_names(test_norm, test_cands)
