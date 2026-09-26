"""
Step 1 — Exploratory Data Analysis (EDA) on Business Names
ML Challenge 2026: Business Entity Resolution (Person 1 - Name Domain Track)

Noise Patterns Observed Across Data Sources:
1. Legal Suffixes & Inconsistencies:
   - Corp <-> Corporation, Pvt <-> Private, Ltd <-> Limited, Inc <-> Incorporated
   - Co <-> Company, LLC <-> L.L.C. <-> Limited Liability Company, LLP <-> L.L.P.
   - French suffixes: SARL <-> S.A.R.L., SAS <-> SASU <-> S.A.S., SCI <-> S.C.I., EURL
   - Redundant/duplicated suffixes: "Pvt Pvt Ltd", "LLC LLC"
   - Inverted suffix position: "Inc Arden's Tire Shop", "INC. DELTAVERA GROWTH"

2. DBA, Trade Names & Alias Prefixes:
   - "DBA: <Name>", "d/b/a <Name>", "trading as <Name>", "formerly known as <Name>", "f/k/a", "aka <Name>", "née <Name>"
   - Core entity name is embedded inside or following the alias clause.

3. Punctuation, Brackets & Formatting:
   - Ampersands vs conjunctions: '&' <-> 'and', '& Fils' <-> 'et Fils'
   - Brackets and metadata: '[Limited]', '(India)', '(Company)', '(ID: 12345)'
   - Punctuation noise: quotes (" ' `), angle brackets (<< >>), slashes, asterisks, extra dots
   - Case variation: ALL-CAPS, all-lowercase, Title Case, mixed casing
   - Multiple or irregular spaces / tab stops

4. Word Order Swaps & Transpositions:
   - Reordered name tokens: "Beaune Sport EURL" <-> "BEAUNE EURL SPORT"
   - Core token transposition: "Om Constructions" <-> "Constructions Om"

5. Typos, Character Edits & OCR Artifacts:
   - Insertion, deletion, transposition of adjacent characters (e.g. "Dmaigesostcis" vs "Diagnostics")
   - Accented characters: 'é', 'è', 'ê', 'ç', 'à' vs unaccented equivalents 'e', 'c', 'a'

6. Transliteration & Multilingual Text:
   - Non-Latin Indic scripts (Devanagari, Tamil, Telugu) vs Latin phonetic equivalents
   - Honorific prefixes: "M/s", "Shri", "Sri", "Dr."
"""

import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

def run_eda(train_dir="dataset/train", num_examples=20):
    print("=== Step 1: EDA on Name Track ===")
    s1 = pd.read_csv(f"{train_dir}/train_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{train_dir}/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{train_dir}/train_source3.tsv", sep="\t")
    gt = pd.read_csv(f"{train_dir}/train_ground_truth.tsv", sep="\t")

    s1_map = dict(zip(s1["entity_id"], zip(s1["business_name"], s1["country"])))
    s2_map = dict(zip(s2["entity_id"], s2["business_name"]))
    s3_map = dict(zip(s3["entity_id"], s3["business_name"]))

    pairs = []
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if not matched_str.strip():
            continue
        matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
        s1_name, country = s1_map.get(s1_id, ("UNKNOWN", "UNKNOWN"))
        for m_id in matched_ids:
            if m_id.startswith("S2-"):
                cand_name = s2_map.get(m_id, "UNKNOWN")
                src = "S2"
            else:
                cand_name = s3_map.get(m_id, "UNKNOWN")
                src = "S3"
            pairs.append((s1_id, s1_name, m_id, cand_name, src, country))

    print(f"Total true match pairs in ground truth: {len(pairs)}")
    print(f"\n--- Printing {min(num_examples, len(pairs))} Known True-Match Pairs Side by Side ---")
    header = f"{'Source 1 ID':<15} | {'Country':<7} | {'Source 1 Business Name':<35} | {'Match ID':<15} | {'Src':<3} | {'Match Business Name'}"
    print(header)
    print("-" * len(header))

    for i, (s1_id, s1_name, m_id, cand_name, src, country) in enumerate(pairs[:num_examples]):
        s1_trunc = (s1_name[:32] + "...") if len(s1_name) > 35 else s1_name
        cand_trunc = (cand_name[:40] + "...") if len(cand_name) > 40 else cand_name
        print(f"{s1_id:<15} | {country:<7} | {s1_trunc:<35} | {m_id:<15} | {src:<3} | {cand_trunc}")

if __name__ == "__main__":
    run_eda()
