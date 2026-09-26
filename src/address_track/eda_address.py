"""
Step 1 — Exploratory Data Analysis (EDA) on Business Addresses
Amazon ML Challenge 2026: Business Entity Resolution (Person 2 - Address Domain Track)

Examines:
- Missing values in addresses, countries
- Country distributions across sources
- Address length and token statistics
- Pincode / ZIP code patterns across countries
- City patterns and extraction feasibility
- Common address abbreviations and noise
- Landmark phrases ("Near", "Opposite", "Behind", etc.)
- Side-by-side comparison of true-match pairs from train_ground_truth.tsv
"""

import os
import re
import sys
import pandas as pd
from collections import Counter

# Set stdout encoding
sys.stdout.reconfigure(encoding='utf-8')

def run_eda(train_dir="dataset/train", num_examples=25):
    print("=" * 80)
    print("        PERSON 2 — ADDRESS DOMAIN TRACK: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # 1. Load datasets
    print("\n[1] Loading train source files...")
    s1 = pd.read_csv(f"{train_dir}/train_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{train_dir}/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{train_dir}/train_source3.tsv", sep="\t")
    gt = pd.read_csv(f"{train_dir}/train_ground_truth.tsv", sep="\t")

    print(f"  Source 1 records : {len(s1):,}")
    print(f"  Source 2 records : {len(s2):,}")
    print(f"  Source 3 records : {len(s3):,}")
    print(f"  Ground Truth rows: {len(gt):,}")

    # 2. Check missing values
    print("\n[2] Missing Values Analysis:")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        addr_missing = df["business_address"].isna().sum()
        country_missing = df["country"].isna().sum()
        addr_empty = (df["business_address"].fillna("").str.strip() == "").sum()
        print(f"  {name}: total={len(df):,}, missing/empty addr={addr_missing + addr_empty} ({(addr_missing+addr_empty)/len(df)*100:.2f}%), missing country={country_missing}")

    # 3. Country Distribution
    print("\n[3] Country Distribution across sources:")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        counts = df["country"].value_counts().to_dict()
        print(f"  {name}: {counts}")

    # 4. Address Length and Token Stats
    print("\n[4] Address Length & Token Statistics:")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        addrs = df["business_address"].dropna().astype(str)
        char_lens = addrs.apply(len)
        token_counts = addrs.apply(lambda x: len(x.split()))
        print(f"  {name}:")
        print(f"    Char length  : mean={char_lens.mean():.1f}, min={char_lens.min()}, median={char_lens.median():.1f}, max={char_lens.max()}")
        print(f"    Token count  : mean={token_counts.mean():.1f}, min={token_counts.min()}, median={token_counts.median():.1f}, max={token_counts.max()}")

    # 5. Pincode / ZIP code pattern analysis
    print("\n[5] Pincode / ZIP Code Patterns:")
    # Check digits in addresses
    zip_us = re.compile(r'\b\d{5}(?:-\d{4})?\b')
    pin_india = re.compile(r'\b[1-9]\d{5}\b')
    pin_france = re.compile(r'\b\d{5}\b')

    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        addrs = df["business_address"].dropna().astype(str)
        countries = df["country"].dropna().astype(str)
        
        has_us_zip = sum(1 for a, c in zip(addrs, countries) if c == "US" and zip_us.search(a))
        total_us = (countries == "US").sum()
        
        has_in_pin = sum(1 for a, c in zip(addrs, countries) if c == "India" and pin_india.search(a))
        total_in = (countries == "India").sum()

        has_fr_pin = sum(1 for a, c in zip(addrs, countries) if c == "France" and pin_france.search(a))
        total_fr = (countries == "France").sum()

        print(f"  {name}:")
        if total_us > 0:
            print(f"    US zip found: {has_us_zip}/{total_us} ({has_us_zip/total_us*100:.1f}%)")
        if total_in > 0:
            print(f"    India pin found: {has_in_pin}/{total_in} ({has_in_pin/total_in*100:.1f}%)")
        if total_fr > 0:
            print(f"    France pin found: {has_fr_pin}/{total_fr} ({has_fr_pin/total_fr*100:.1f}%)")

    # 6. Landmark Keywords
    print("\n[6] Landmark Phrases in Addresses:")
    landmark_pattern = re.compile(r'\b(near|opp|opposite|behind|beside|next to|adjacent|in front of|above|below|floor|flr|nr)\b', re.IGNORECASE)
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        addrs = df["business_address"].dropna().astype(str)
        has_landmark = addrs.apply(lambda a: bool(landmark_pattern.search(a))).sum()
        print(f"  {name}: {has_landmark:,}/{len(df):,} ({has_landmark/len(df)*100:.1f}%) have landmark keywords")

    # 7. Common tokens and potential abbreviations
    print("\n[7] Common Address Tokens (Top 40):")
    all_tokens = []
    for df in [s1, s2, s3]:
        for addr in df["business_address"].dropna().astype(str):
            clean = re.sub(r'[^\w\s]', ' ', addr.lower())
            all_tokens.extend(clean.split())
    counter = Counter(all_tokens)
    print(f"  Top 40 tokens across all addresses:")
    print("  " + ", ".join(f"{tok}({cnt})" for tok, cnt in counter.most_common(40)))

    # 8. True-match pair comparison from Ground Truth
    print("\n[8] Ground Truth True-Match Pair Comparison:")
    s1_map = dict(zip(s1["entity_id"], zip(s1["business_address"].fillna(""), s1["country"], s1["business_name"].fillna(""))))
    s2_map = dict(zip(s2["entity_id"], zip(s2["business_address"].fillna(""), s2["country"], s2["business_name"].fillna(""))))
    s3_map = dict(zip(s3["entity_id"], zip(s3["business_address"].fillna(""), s3["country"], s3["business_name"].fillna(""))))

    pairs = []
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if not matched_str.strip():
            continue
        matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
        s1_addr, country, s1_name = s1_map.get(s1_id, ("", "UNKNOWN", ""))
        for m_id in matched_ids:
            if m_id.startswith("S2-"):
                cand_addr, c_country, cand_name = s2_map.get(m_id, ("", "UNKNOWN", ""))
                src = "S2"
            else:
                cand_addr, c_country, cand_name = s3_map.get(m_id, ("", "UNKNOWN", ""))
                src = "S3"
            pairs.append((s1_id, s1_addr, m_id, cand_addr, src, country, s1_name, cand_name))

    print(f"  Total true-match pairs in ground truth: {len(pairs):,}")
    print(f"\n--- Side-by-Side Comparison of {min(num_examples, len(pairs))} True-Match Addresses ---")
    for i, (s1_id, s1_addr, m_id, cand_addr, src, country, s1_name, cand_name) in enumerate(pairs[:num_examples]):
        print(f"\n[{i+1}] Country: {country} | S1: {s1_id} vs {src}: {m_id}")
        print(f"    S1 Name: {s1_name}")
        print(f"    Match  : {cand_name}")
        print(f"    S1 Addr: {s1_addr}")
        print(f"    Match  : {cand_addr}")

if __name__ == "__main__":
    run_eda()
