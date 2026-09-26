"""
Step 2 — Name Normalization Module
ML Challenge 2026: Business Entity Resolution (Person 1 - Name Domain Track)

Functions:
- normalize_name(raw_name: str, country: str = "") -> str
- tokenize_name(name_norm: str) -> list[str]
- normalize_and_tokenize(raw_name: str, country: str = "") -> tuple[str, list[str]]
- run_unit_tests() -> None
- apply_normalization_to_files(...) -> None
"""

import sys
import re
import unicodedata
import os
import csv
import pandas as pd

# Precompiled regex patterns for legal suffix synonyms across US, India, France
# Replacements are padded with spaces to avoid adjacent token collision
COMPILED_SUFFIXES = [
    # Multi-word suffixes first
    (re.compile(r'\b(private\s+limited|pvt\s*\.?\s*ltd\s*\.?|pvt\s*\.?\s*limited|private\s+ltd\s*\.?)\b', re.IGNORECASE), ' pvt ltd '),
    (re.compile(r'\b(public\s+limited|pub\s*\.?\s*ltd\s*\.?|public\s+ltd\s*\.?)\b', re.IGNORECASE), ' ltd '),
    (re.compile(r'\b(limited\s+liability\s+company|l\s*\.?\s*l\s*\.?\s*c\s*\.?)\b', re.IGNORECASE), ' llc '),
    (re.compile(r'\b(limited\s+liability\s+partnership|l\s*\.?\s*l\s*\.?\s*p\s*\.?)\b', re.IGNORECASE), ' llp '),
    (re.compile(r'\b(professional\s+limited\s+liability\s+company|p\s*\.?\s*l\s*\.?\s*l\s*\.?\s*c\s*\.?)\b', re.IGNORECASE), ' pllc '),
    (re.compile(r'\b(societe\s+a\s+responsabilite\s+limitee|s\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\s*\.?)\b', re.IGNORECASE), ' sarl '),
    (re.compile(r'\b(societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|s\s*\.?\s*a\s*\.?\s*s\s*\.?\s*u\s*\.?)\b', re.IGNORECASE), ' sas '),
    (re.compile(r'\b(societe\s+par\s+actions\s+simplifiee|s\s*\.?\s*a\s*\.?\s*s\s*\.?)\b', re.IGNORECASE), ' sas '),
    (re.compile(r'\b(entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|e\s*\.?\s*u\s*\.?\s*r\s*\.?\s*l\s*\.?)\b', re.IGNORECASE), ' eurl '),
    (re.compile(r'\b(societe\s+civile\s+immobiliere|s\s*\.?\s*c\s*\.?\s*i\s*\.?)\b', re.IGNORECASE), ' sci '),
    (re.compile(r'\b(societe\s+anonyme|s\s*\.?\s*a\s*\.?)\b', re.IGNORECASE), ' sa '),
    (re.compile(r'\b(et\s+fils|&\s*fils)\b', re.IGNORECASE), ' fils '),
    (re.compile(r'\b(et\s+freres|&\s*freres)\b', re.IGNORECASE), ' freres '),
    # Single-word suffixes
    (re.compile(r'\b(corporation|corp\s*\.?)\b', re.IGNORECASE), ' corp '),
    (re.compile(r'\b(incorporated|inc\s*\.?)\b', re.IGNORECASE), ' inc '),
    (re.compile(r'\b(limited|ltd\s*\.?)\b', re.IGNORECASE), ' ltd '),
    (re.compile(r'\b(company|co\s*\.?)\b', re.IGNORECASE), ' co '),
    (re.compile(r'\b(compagnie|cie\s*\.?)\b', re.IGNORECASE), ' cie '),
]

DBA_PATTERN = re.compile(
    r'\b(?:dba|d/b/a|aka|a/k/a|trading\s+as|t/a|formerly\s+known\s+as|fka|f/k/a|n[eé]e)[:\s]+(.+)',
    flags=re.IGNORECASE
)
METADATA_PATTERN = re.compile(r'[\(\[]\s*(?:id:?|#)\s*\d+\s*[\)\]]', flags=re.IGNORECASE)
HONORIFIC_PREFIXES = re.compile(r'^(?:m/s\.?|shri\.?|sri\.?|dr\.?|mr\.?|mrs\.?)\s+', flags=re.IGNORECASE)
NON_ALPHANUM = re.compile(r'[^\w\s]')


def strip_accents(text: str) -> str:
    """Normalize unicode and remove combining diacritics (e.g. é -> e, ç -> c)."""
    if text.isascii():
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_name(raw_name: str, country: str = "") -> str:
    """
    Standardize a raw business name string.
    - Robust against None / empty / non-string.
    - Strips alias/DBA clauses to isolate referenced core name.
    - Strips metadata IDs, honorific prefixes, and diacritics.
    - Expands '&' to 'and'.
    - Standardizes legal suffixes via synonym regexes.
    - Strips punctuation and collapses whitespace.
    """
    if not isinstance(raw_name, str):
        return ""
    text = raw_name.strip()
    if not text:
        return ""

    # Check for DBA / alias clause
    dba_match = DBA_PATTERN.search(text)
    if dba_match:
        target = dba_match.group(1).strip()
        if len(target) >= 3:
            text = target

    # Strip metadata IDs (e.g. (ID: 1234))
    if '(' in text or '[' in text:
        text = METADATA_PATTERN.sub(' ', text)

    # Strip honorific prefixes
    text = HONORIFIC_PREFIXES.sub('', text)

    # Normalize accents
    text = strip_accents(text)

    # Standardize conjunction
    if '&' in text:
        text = text.replace('&', ' and ')

    # Lowercase
    text = text.lower()

    # Standardize legal suffixes
    for pat, rep in COMPILED_SUFFIXES:
        text = pat.sub(rep, text)

    # Strip punctuation while preserving alphanumeric Unicode characters
    text = NON_ALPHANUM.sub(' ', text)

    # Collapse whitespace
    return ' '.join(text.split())


def tokenize_name(name_norm: str) -> list[str]:
    """Tokenize a normalized name string into a list of word tokens."""
    if not name_norm:
        return []
    return name_norm.split()


def normalize_and_tokenize(raw_name: str, country: str = "") -> tuple[str, list[str]]:
    """Return both normalized name string and list of tokens."""
    norm = normalize_name(raw_name, country)
    return norm, tokenize_name(norm)


def run_unit_tests():
    """Unit-test normalize_name against 20 noisy real-world pairs from Step 1."""
    test_cases = [
        ("Zephay Labs Inc", "Global aka Zephay Labs Inc"),
        ("Red Perfect Trading", "Perfect Red Trading"),
        ("Nandlal Kisan LLP", "Nandlal  Kisan  LLP"),
        ("Om Constructions Pvt Ltd", "Om  Constructions  Pvt  Ltd"),
        ("Thermal & Fils SASU", "Thermal & Fils S.A.S."),
        ("Siliguri Media Pvt Ltd", "Nova d/b/a Siliguri Media Pvt Ltd"),
        ("Naman Trust", "[Naman Trust]"),
        ("Consulting Sai Nanak Private Limited", "Consulting Sai Nanak Private Limited (ID: 7462)"),
        ("Garcia Mercury Corp", "Garcia Mercury CORP."),
        ("Market Sciences (India) Ltd", "Market Sciences (India) [Limited]"),
        ("Chayan Trading Private Limited", "Chayan  Trading Private Limited"),
        ("M/s Sandeep Software (India) Pvt. Ltd", "Sandeep Software (India) Pvt. Ltd."),
        ("Dr Boon Charitable Trust", "Boon Charitable Trust"),
        ("Fédération du [Lecole]", "Federation du Lecole"),
        ("BANGALORE SÉCURITIES PVT LTD", "Bangalore Securities Pvt Ltd"),
        ("Refuge Sèrvices SARL", "Refuge Services S.A.R.L."),
        ("BEAUNE EURL SPORT", "Beaune Sport EURL"),
        ("Wexnovi DBA: Lakshmi Media Private Limited", "Lakshmi Media Private Limited"),
        ("Novihalozeta trading as Central Financial Consultants", "Central Financial Consultants"),
        ("Rizairiio formerly known as Seven Business Pvt Ltd", "Seven Business Pvt Ltd"),
    ]

    print("=== Running Unit Tests on normalize_name ===")
    passed = 0
    for s1_raw, s2_raw in test_cases:
        norm1 = normalize_name(s1_raw)
        norm2 = normalize_name(s2_raw)
        t1, t2 = set(norm1.split()), set(norm2.split())
        jaccard = len(t1 & t2) / max(len(t1 | t2), 1)
        is_close = (norm1 == norm2) or (jaccard >= 0.7)
        if is_close:
            passed += 1
        status = "PASS" if is_close else "WARN"
        print(f"[{status}] Jaccard={jaccard:.2f} | {s1_raw!r} -> {norm1!r}  vs  {norm2!r}")

    print(f"\nUnit Tests Result: {passed}/{len(test_cases)} passed successfully.")
    assert passed >= 18, f"Expected at least 18/20 test cases to pass, got {passed}"


def apply_normalization_to_files(
    train_files: list[tuple[str, str]],
    test_files: list[tuple[str, str]],
    train_out: str = "data/processed/name_normalized_train.tsv",
    test_out: str = "data/processed/name_normalized_test.tsv"
):
    """
    Applies normalize_name to all train and test source files using memory-bounded streaming.
    Writes:
      entity_id, name_norm, country, source
    to data/processed/name_normalized_{train,test}.tsv.
    """
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    
    print("\n--- Normalizing Train Dataset ---")
    train_dfs = []
    for path, src_tag in train_files:
        print(f"Reading {path} ({src_tag})...")
        df = pd.read_csv(path, sep="\t")
        df["name_norm"] = df["business_name"].fillna("").astype(str).apply(normalize_name)
        df["source"] = src_tag
        train_dfs.append(df[["entity_id", "name_norm", "country", "source"]])
    
    df_train_all = pd.concat(train_dfs, ignore_index=True)
    df_train_all.to_csv(train_out, sep="\t", index=False)
    print(f"Wrote {len(df_train_all):,} rows to {train_out}")

    print("\n--- Normalizing Test Dataset (Low-Memory Streaming) ---")
    total_test = 0
    with open(test_out, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["entity_id", "name_norm", "country", "source"])
        
        for path, src_tag in test_files:
            print(f"Streaming normalization for {path} ({src_tag})...")
            file_count = 0
            with open(path, "r", encoding="utf-8", errors="replace") as f_in:
                reader = csv.reader(f_in, delimiter="\t")
                header = next(reader, None)
                if not header:
                    continue
                try:
                    eid_idx = header.index("entity_id")
                    name_idx = header.index("business_name")
                    country_idx = header.index("country")
                except ValueError as e:
                    print(f"Header missing expected column in {path}: {header}")
                    continue
                
                for row in reader:
                    if not row:
                        continue
                    eid = row[eid_idx].strip()
                    raw_name = row[name_idx] if len(row) > name_idx else ""
                    country = row[country_idx].strip() if len(row) > country_idx else ""
                    norm_name = normalize_name(raw_name, country)
                    writer.writerow([eid, norm_name, country, src_tag])
                    file_count += 1
                    if file_count % 500000 == 0:
                        print(f"  Processed {file_count:,} records from {src_tag}...", flush=True)
                        
            print(f"  Completed {file_count:,} records from {src_tag}.")
            total_test += file_count

    print(f"Wrote {total_test:,} rows to {test_out}")


if __name__ == "__main__":
    run_unit_tests()
    
    train_inputs = [
        ("dataset/train/train_source1.tsv", "S1"),
        ("dataset/train/train_source2.tsv", "S2"),
        ("dataset/train/train_source3.tsv", "S3"),
    ]
    test_inputs = [
        ("dataset/test/test_source1.tsv", "S1"),
        ("dataset/test/test_source2.tsv", "S2"),
        ("dataset/test/test_source3.tsv", "S3"),
    ]
    apply_normalization_to_files(train_inputs, test_inputs)
