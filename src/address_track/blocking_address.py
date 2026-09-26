"""
Step 3 — Address-Based Blocking Module
Amazon ML Challenge 2026: Business Entity Resolution (Person 2 - Address Domain Track)

Implements multi-pass address blocking:
  Pass 0) Exact normalized address match
  Pass A) Geographic component blocking:
          - (country, pincode) when pincode present
          - (country, city_norm) when city present
  Pass B) Token-based inverted index on distinctive content tokens
  Pass C) Character n-gram / prefix blocking on normalized address text
  Pass D) Union, deduplication, priority ranking, and capping per Source 1 entity

Outputs:
  1. Challenge format: data/processed/address_candidates.tsv
     source1_entity_id \\t candidate_entity_ids (comma-separated, every S1 has a row)
  2. Long pair format: data/processed/address_candidates_train.tsv
     source1_entity_id \\t candidate_entity_id \\t source \\t block_reason
"""

import os
import sys
import csv
import gc
from collections import defaultdict
import pandas as pd

# Generic address stopwords excluded from token inverted index to avoid massive buckets
ADDRESS_STOPWORDS = {
    'road', 'street', 'avenue', 'boulevard', 'drive', 'lane', 'court', 'place', 'highway', 'square',
    'parkway', 'way', 'path', 'floor', 'apartment', 'suite', 'unit', 'number', 'building', 'house',
    'plot', 'survey', 'room', 'block', 'sector', 'phase', 'society', 'near', 'opposite', 'behind',
    'beside', 'north', 'south', 'east', 'west', 'central', 'new', 'old', 'city', 'state', 'town',
    'de', 'du', 'des', 'la', 'le', 'les', 'et', 'en', 'rue', 'allee', 'impasse', 'chemin', 'route',
    'and', 'the', 'for', 'with', 'from', 'first', 'second', 'third'
}


def get_distinctive_address_tokens(address_norm: str) -> list[str]:
    """Extract informative content tokens from normalized address."""
    if not isinstance(address_norm, str) or not address_norm:
        return []
    tokens = address_norm.split()
    return [t for t in tokens if len(t) >= 4 and t not in ADDRESS_STOPWORDS]


def block_address_shard(
    s1_records: list[dict],
    cand_records: list[dict],
    max_cands_per_s1: int = 25,
    max_bucket_size: int = 500
) -> list[tuple[str, str, str, str]]:
    """
    Runs multi-pass blocking on a single country shard.
    Returns list of tuples: (s1_id, cand_id, cand_source, block_reason)
    """
    pairs = []

    # 1. Build inverted indexes over candidate records (S2 & S3)
    exact_addr_idx = defaultdict(list)
    pincode_idx = defaultdict(list)
    city_idx = defaultdict(list)
    token_idx = defaultdict(list)
    prefix_idx = defaultdict(list)

    for idx, c in enumerate(cand_records):
        cid = c["entity_id"]
        a_norm = c["address_norm"]
        pin = c["pincode"]
        city = c["city_norm"]
        core = c.get("address_core", a_norm)

        # Pass 0: Exact address
        if a_norm:
            exact_addr_idx[a_norm].append(idx)
            # Pass C: Character prefix of normalized address
            if len(a_norm) >= 6:
                prefix_idx[a_norm[:6]].append(idx)

        # Pass A: Pincode & City
        if pin:
            pincode_idx[pin].append(idx)
        if city and len(city) >= 3:
            city_idx[city].append(idx)

        # Pass B: Distinctive content tokens
        tokens = get_distinctive_address_tokens(core)
        for t in tokens:
            token_idx[t].append(idx)

    # 2. Collect and score candidate matches for each S1 entity
    for s1 in s1_records:
        s1_id = s1["entity_id"]
        a_norm = s1["address_norm"]
        pin = s1["pincode"]
        city = s1["city_norm"]
        core = s1.get("address_core", a_norm)

        cand_reasons = defaultdict(set)
        cand_scores = defaultdict(int)

        # Pass 0: Exact address match (highest confidence)
        if a_norm and a_norm in exact_addr_idx:
            for idx in exact_addr_idx[a_norm]:
                cand_reasons[idx].add("exact_address")
                cand_scores[idx] += 100

        # Pass A: Geographic components
        if pin and pin in pincode_idx:
            bucket = pincode_idx[pin]
            if len(bucket) <= max_bucket_size:
                for idx in bucket:
                    cand_reasons[idx].add("geo_pincode")
                    cand_scores[idx] += 15

        if city and city in city_idx:
            bucket = city_idx[city]
            # Cap city bucket to prevent explosion
            if len(bucket) <= max_bucket_size:
                for idx in bucket:
                    cand_reasons[idx].add("geo_city")
                    cand_scores[idx] += 5

        # Pass B: Content token overlap
        tokens = get_distinctive_address_tokens(core)
        for t in tokens:
            if t in token_idx:
                bucket = token_idx[t]
                if len(bucket) <= max_bucket_size:
                    for idx in bucket:
                        cand_reasons[idx].add("token_inverted_index")
                        cand_scores[idx] += 8

        # Pass C: N-gram prefix match
        if a_norm and len(a_norm) >= 6:
            pref = a_norm[:6]
            if pref in prefix_idx:
                bucket = prefix_idx[pref]
                if len(bucket) <= max_bucket_size:
                    for idx in bucket:
                        cand_reasons[idx].add("ngram_block")
                        cand_scores[idx] += 4

        # Rank candidates by composite score descending, then cap
        sorted_cands = sorted(
            cand_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_cands_per_s1]

        for idx, _ in sorted_cands:
            cand = cand_records[idx]
            reasons_str = "+".join(sorted(cand_reasons[idx]))
            pairs.append((s1_id, cand["entity_id"], cand["source"], reasons_str))

    return pairs


def block_addresses(
    normalized_file: str = "data/processed/address_normalized_train.tsv",
    challenge_output: str = "data/processed/address_candidates.tsv",
    long_output: str = "data/processed/address_candidates_train.tsv",
    max_cands_per_s1: int = 25,
    max_bucket_size: int = 500
) -> dict:
    """
    Main blocking runner.
    Loads normalized records, shards by country, executes multi-pass blocking,
    and produces both challenge format and long format outputs.
    """
    os.makedirs(os.path.dirname(challenge_output), exist_ok=True)
    os.makedirs(os.path.dirname(long_output), exist_ok=True)
    print(f"\n=== Running Address Blocking on {normalized_file} ===")

    # Load normalized records into memory
    print(f"Reading {normalized_file}...")
    df = pd.read_csv(normalized_file, sep="\t", dtype=str).fillna("")
    print(f"Total normalized records loaded: {len(df):,}")

    countries = sorted(set(df["country"].unique()) - {""})
    print(f"Unique country shards: {countries}")

    # Preserve exact S1 entity order from source file
    all_s1_ids = df[df["source"] == "S1"]["entity_id"].tolist()
    cands_by_s1 = defaultdict(list)
    total_pairs = 0
    total_s1 = len(all_s1_ids)
    total_cand_entities = 0

    with open(long_output, "w", encoding="utf-8", newline="") as f_long:
        long_writer = csv.writer(f_long, delimiter="\t")
        long_writer.writerow(["source1_entity_id", "candidate_entity_id", "source", "block_reason"])

        for country in countries:
            df_country = df[df["country"] == country]
            s1_sub = df_country[df_country["source"] == "S1"]
            cand_sub = df_country[df_country["source"].isin(["S2", "S3"])]

            n_s1 = len(s1_sub)
            n_cand = len(cand_sub)
            total_cand_entities += n_cand

            print(f"\nProcessing Shard '{country}': {n_s1:,} S1 entities, {n_cand:,} S2/S3 candidate entities...")
            if n_s1 == 0 or n_cand == 0:
                continue

            s1_records = s1_sub.to_dict("records")
            cand_records = cand_sub.to_dict("records")

            pairs = block_address_shard(
                s1_records,
                cand_records,
                max_cands_per_s1=max_cands_per_s1,
                max_bucket_size=max_bucket_size
            )

            for s1_id, cid, csrc, reason in pairs:
                long_writer.writerow([s1_id, cid, csrc, reason])
                cands_by_s1[s1_id].append(cid)
                total_pairs += 1

            print(f"  Shard '{country}': Generated {len(pairs):,} candidate pairs.")
            del s1_records, cand_records, pairs
            gc.collect()

    # Write challenge format: source1_entity_id \t candidate_entity_ids
    print(f"\nWriting challenge-format candidates to {challenge_output}...")
    with open(challenge_output, "w", encoding="utf-8", newline="") as f_chal:
        chal_writer = csv.writer(f_chal, delimiter="\t")
        chal_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for s1_id in all_s1_ids:
            cands = cands_by_s1.get(s1_id, [])
            cands_str = ",".join(cands)
            chal_writer.writerow([s1_id, cands_str])

    possible_comps = max(total_s1 * total_cand_entities, 1)
    reduction_ratio = 1.0 - (total_pairs / possible_comps)
    avg_cands = total_pairs / max(total_s1, 1)

    print("\n" + "=" * 60)
    print("             ADDRESS BLOCKING SUMMARY")
    print("=" * 60)
    print(f"Total Source 1 entities        : {total_s1:,}")
    print(f"Total Candidate pool (S2/S3)   : {total_cand_entities:,}")
    print(f"Total Candidate pairs generated: {total_pairs:,}")
    print(f"Average candidates per S1      : {avg_cands:.2f}")
    print(f"Search space reduction ratio   : {reduction_ratio:.8f} ({reduction_ratio*100:.6f}%)")
    print(f"Saved challenge candidates to  : {challenge_output}")
    print(f"Saved long format candidates to: {long_output}")

    return {
        "total_pairs": total_pairs,
        "total_s1": total_s1,
        "reduction_ratio": reduction_ratio,
        "avg_cands": avg_cands,
        "challenge_output": challenge_output,
        "long_output": long_output
    }


if __name__ == "__main__":
    train_norm = "data/processed/address_normalized_train.tsv"
    train_cands_chal = "data/processed/address_candidates.tsv"
    train_cands_long = "data/processed/address_candidates_train.tsv"

    test_norm = "data/processed/address_normalized_test.tsv"
    test_cands_chal = "data/processed/address_candidates_test_chal.tsv"
    test_cands_long = "data/processed/address_candidates_test.tsv"

    if os.path.exists(train_norm):
        block_addresses(
            normalized_file=train_norm,
            challenge_output=train_cands_chal,
            long_output=train_cands_long
        )

    if os.path.exists(test_norm):
        block_addresses(
            normalized_file=test_norm,
            challenge_output=test_cands_chal,
            long_output=test_cands_long
        )
