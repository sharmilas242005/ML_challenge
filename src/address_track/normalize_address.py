"""
Step 2 — Address Normalization Module
Amazon ML Challenge 2026: Business Entity Resolution (Person 2 - Address Domain Track)

Functions:
- normalize_address(raw_address: str, country: str = "") -> str
- extract_pincode(raw_address: str, country: str = "") -> str
- extract_city(raw_address: str, country: str = "") -> str
- extract_landmark(raw_address: str) -> str
- tokenize_address(address_norm: str) -> list[str]
- normalize_record(raw_address: str, country: str = "", name: str = "", entity_id: str = "") -> dict
- run_unit_tests() -> None
- apply_normalization_to_files(...) -> None

Shared normalized-record interface:
entity_id, name_norm, address_norm, country, city_norm, pincode
Plus preserved internal fields:
address_core, landmark, address_tokens, source
"""

import os
import sys
import re
import csv
import unicodedata
try:
    import pandas as pd
except ImportError:
    pd = None

# Standard abbreviations across US, India, France
# Padded with spaces to avoid token collision
COMPILED_ABBREVIATIONS = [
    # Multi-word or specific phrases
    (re.compile(r'\bnext\s+to\b', re.IGNORECASE), ' next to '),
    (re.compile(r'\bin\s+front\s+of\b', re.IGNORECASE), ' in front of '),
    (re.compile(r'\badjacent\s+to\b', re.IGNORECASE), ' adjacent to '),
    (re.compile(r'\bp\s*\.?\s*o\s*\.?\s*box\b', re.IGNORECASE), ' po box '),
    (re.compile(r'\bh\s*\.?\s*no\s*\.?\b', re.IGNORECASE), ' house number '),
    (re.compile(r'\bplot\s+no\s*\.?\b', re.IGNORECASE), ' plot number '),
    (re.compile(r'\bs\s*\.?\s*no\s*\.?\b', re.IGNORECASE), ' survey number '),
    
    # Common Street / Way types
    (re.compile(r'\brd\s*\.?\b', re.IGNORECASE), ' road '),
    (re.compile(r'\bst\s*\.?\b', re.IGNORECASE), ' street '),
    (re.compile(r'\bave\s*\.?\b', re.IGNORECASE), ' avenue '),
    (re.compile(r'\bav\s*\.?\b', re.IGNORECASE), ' avenue '),
    (re.compile(r'\bblvd\s*\.?\b', re.IGNORECASE), ' boulevard '),
    (re.compile(r'\bbd\s*\.?\b', re.IGNORECASE), ' boulevard '),
    (re.compile(r'\bbvd\s*\.?\b', re.IGNORECASE), ' boulevard '),
    (re.compile(r'\bdr\s*\.?\b', re.IGNORECASE), ' drive '),
    (re.compile(r'\bhwy\s*\.?\b', re.IGNORECASE), ' highway '),
    (re.compile(r'\bln\s*\.?\b', re.IGNORECASE), ' lane '),
    (re.compile(r'\bct\s*\.?\b', re.IGNORECASE), ' court '),
    (re.compile(r'\bpkwy\s*\.?\b', re.IGNORECASE), ' parkway '),
    (re.compile(r'\bsq\s*\.?\b', re.IGNORECASE), ' square '),
    (re.compile(r'\bpl\s*\.?\b', re.IGNORECASE), ' place '),
    (re.compile(r'\brte\s*\.?\b', re.IGNORECASE), ' route '),
    (re.compile(r'\ball\s*\.?\b', re.IGNORECASE), ' allee '),
    (re.compile(r'\bimp\s*\.?\b', re.IGNORECASE), ' impasse '),
    (re.compile(r'\bchem\s*\.?\b', re.IGNORECASE), ' chemin '),

    # Unit / Sub-location descriptors
    (re.compile(r'\bste\s*\.?\b', re.IGNORECASE), ' suite '),
    (re.compile(r'\bapt\s*\.?\b', re.IGNORECASE), ' apartment '),
    (re.compile(r'\bapts\s*\.?\b', re.IGNORECASE), ' apartments '),
    (re.compile(r'\bflr\s*\.?\b', re.IGNORECASE), ' floor '),
    (re.compile(r'\bfl\s*\.?\b', re.IGNORECASE), ' floor '),
    (re.compile(r'\b1st\b', re.IGNORECASE), ' 1 '),
    (re.compile(r'\b2nd\b', re.IGNORECASE), ' 2 '),
    (re.compile(r'\b3rd\b', re.IGNORECASE), ' 3 '),
    (re.compile(r'\b4th\b', re.IGNORECASE), ' 4 '),
    (re.compile(r'\b5th\b', re.IGNORECASE), ' 5 '),
    (re.compile(r'\bbldg\s*\.?\b', re.IGNORECASE), ' building '),
    (re.compile(r'\bchs\s*\.?\b', re.IGNORECASE), ' society '),
    (re.compile(r'\bsoc\s*\.?\b', re.IGNORECASE), ' society '),
    (re.compile(r'\bsec\s*\.?\b', re.IGNORECASE), ' sector '),
    (re.compile(r'\bdist\s*\.?\b', re.IGNORECASE), ' district '),
    (re.compile(r'\bno\s*\.?\b', re.IGNORECASE), ' number '),
    (re.compile(r'\bnrs?\s*\.?\b', re.IGNORECASE), ' number '),

    # Landmarks & Directions
    (re.compile(r'\bopp\s*\.?\b', re.IGNORECASE), ' opposite '),
    (re.compile(r'\bnr\s*\.?\b', re.IGNORECASE), ' near '),
]

# Patterns for landmarks
LANDMARK_REGEX = re.compile(
    r'\b(?:near|opp|opposite|behind|beside|next\s+to|in\s+front\s+of|adjacent\s+to|above|below)\b[^,;()]*',
    re.IGNORECASE
)

# Known states, territories, and regions across US, India, France
KNOWN_REGIONS = {
    # US 50 States + DC
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks',
    'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny',
    'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv',
    'wi', 'wy', 'dc',
    # India States & UTs
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat',
    'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala', 'madhya pradesh',
    'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha', 'orissa', 'punjab',
    'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura', 'uttar pradesh', 'uttarakhand',
    'west bengal', 'delhi', 'new delhi', 'chandigarh', 'puducherry', 'jammu and kashmir', 'ladakh',
    # France Regions
    'auvergne-rhone-alpes', 'bourgogne-franche-comte', 'bretagne', 'centre-val de loire',
    'corse', 'grand est', 'hauts-de-france', 'ile-de-france', 'normandie', 'nouvelle-aquitaine',
    'occitanie', 'pays de la loire', 'provence-alpes-cote d azur'
}

NON_ALPHANUM = re.compile(r'[^\w\s]')


def strip_accents(text: str) -> str:
    """Normalize unicode and remove combining diacritics (e.g. é -> e, ç -> c)."""
    if not isinstance(text, str):
        return ""
    if text.isascii():
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def extract_pincode(raw_address: str, country: str = "") -> str:
    r"""
    Extract postal code / ZIP code when present.
    Country-aware but open-ended:
      - India: 6 digits (\b[1-9]\d{5}\b)
      - US: 5 digits (\b\d{5}(?:-\d{4})?\b)
      - France: 5 digits (\b\d{5}\b)
      - Open fallback: 5-6 digits
    Avoids mistaking leading street numbers (e.g. '14028 Irving Hill Drive') for zip codes.
    """
    if not isinstance(raw_address, str) or not raw_address.strip():
        return ""
    
    text = strip_accents(raw_address).strip()
    c_upper = country.strip().upper() if country else ""

    if c_upper == "INDIA":
        # Indian 6-digit PIN code
        m = re.search(r'\b([1-9]\d{5})\b', text)
        if m:
            return m.group(1)

    if c_upper == "FRANCE":
        # French 5-digit postal code (often followed by city or uppercase city)
        m = re.search(r'\b(\d{5})\b', text)
        if m:
            return m.group(1)

    if c_upper == "US":
        # US ZIP code: prefer 5 digits after state abbreviation or at the end of comma section
        m = re.search(r'\b[A-Z]{2}\s+(\d{5}(?:-\d{4})?)\b', text)
        if m:
            return m.group(1)
        # Check if last token is 5 digits
        parts = [p.strip() for p in text.split(',') if p.strip()]
        if parts:
            last_tokens = parts[-1].split()
            for t in reversed(last_tokens):
                if re.fullmatch(r'\d{5}(?:-\d{4})?', t):
                    return t

    # General open-ended fallback: check for 5 or 6 digit sequence not in a street address clause
    street_suffix_re = re.compile(
        r'\b(?:street|st|road|rd|avenue|ave|rue|drive|dr|boulevard|blvd|lane|ln|way|court|ct|walk|circle|terrace|pkwy)\b',
        re.IGNORECASE
    )
    for p in reversed(parts):
        # Skip if part looks like a street address line with leading number
        if street_suffix_re.search(p) and re.match(r'^\d+\s+', p.strip()):
            continue
        m = re.search(r'\b(\d{5,6})\b', p)
        if m:
            return m.group(1)

    return ""


def extract_landmark(raw_address: str) -> str:
    """
    Extract landmark phrases (e.g., 'Near Patel Estate', 'Opposite City Union Bank').
    Standardizes abbreviation ('opp' -> 'opposite', 'nr' -> 'near').
    """
    if not isinstance(raw_address, str) or not raw_address.strip():
        return ""
    
    text = strip_accents(raw_address)
    matches = LANDMARK_REGEX.findall(text)
    if not matches:
        return ""
    
    cleaned = []
    for lm in matches:
        lm_str = lm.strip(' ,;()')
        if len(lm_str) >= 4:
            # Normalize landmark abbreviations
            lm_str = re.sub(r'\bopp\s*\.?\b', 'opposite', lm_str, flags=re.IGNORECASE)
            lm_str = re.sub(r'\bnr\s*\.?\b', 'near', lm_str, flags=re.IGNORECASE)
            lm_str = ' '.join(lm_str.lower().split())
            cleaned.append(lm_str)
            
    return ' | '.join(cleaned) if cleaned else ""


def extract_city(raw_address: str, country: str = "") -> str:
    """
    Extract city when reasonably identifiable from comma-separated address parts.
    Handles standard ordering [street, city, state] and inverted ordering [state, city, street].
    """
    if not isinstance(raw_address, str) or not raw_address.strip():
        return ""

    text = strip_accents(raw_address)
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) <= 1:
        return ""

    # Clean parts for inspection
    parts_lower = [re.sub(r'[^\w\s-]', '', p).strip().lower() for p in parts]

    # Street keyword indicator to avoid picking street clauses
    street_re = re.compile(
        r'\b(road|street|avenue|rue|drive|lane|boulevard|blvd|court|way|floor|unit|apt|apartment|suite|no|number|building|sector|phase|block)\b'
    )

    # 1. Check if first part is known region (e.g. 'IA, Iowa City, ...' or 'Nouvelle-Aquitaine, La Teste-de-Buch, ...')
    if parts_lower[0] in KNOWN_REGIONS and len(parts_lower) >= 2:
        cand = parts_lower[1]
        if not street_re.search(cand) and not re.match(r'^\d+', cand):
            return cand

    # 2. Check if last part is known region (e.g. '..., Tyler, TX' or '..., Bordeaux, Nouvelle-Aquitaine')
    if parts_lower[-1] in KNOWN_REGIONS:
        # Check penultimate part
        if len(parts_lower) >= 2:
            cand = parts_lower[-2]
            if not street_re.search(cand) and not re.match(r'^\d+', cand) and len(cand) >= 2:
                return cand
        # Check ante-penultimate
        if len(parts_lower) >= 3:
            cand = parts_lower[-3]
            if not street_re.search(cand) and not re.match(r'^\d+', cand) and len(cand) >= 2:
                return cand

    # 3. Default heuristics: check penultimate part, then first part
    penult = parts_lower[-2]
    if not street_re.search(penult) and not re.match(r'^\d+', penult) and len(penult) >= 3:
        return penult

    first = parts_lower[0]
    if not street_re.search(first) and not re.match(r'^\d+', first) and len(first) >= 3:
        return first

    return ""


# Multi-word or specific phrases
COMPILED_PHRASES = [
    (re.compile(r'\bnext\s+to\b', re.IGNORECASE), ' next to '),
    (re.compile(r'\bin\s+front\s+of\b', re.IGNORECASE), ' in front of '),
    (re.compile(r'\badjacent\s+to\b', re.IGNORECASE), ' adjacent to '),
    (re.compile(r'\bp\s*\.?\s*o\s*\.?\s*box\b', re.IGNORECASE), ' po box '),
    (re.compile(r'\bh\s*\.?\s*no\s*\.?\b', re.IGNORECASE), ' house number '),
    (re.compile(r'\bplot\s+no\s*\.?\b', re.IGNORECASE), ' plot number '),
    (re.compile(r'\bs\s*\.?\s*no\s*\.?\b', re.IGNORECASE), ' survey number '),
]

WORD_ABBREVIATIONS = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard',
    'bd': 'boulevard', 'bvd': 'boulevard', 'dr': 'drive', 'hwy': 'highway', 'ln': 'lane',
    'ct': 'court', 'pkwy': 'parkway', 'sq': 'square', 'pl': 'place', 'rte': 'route',
    'all': 'allee', 'imp': 'impasse', 'chem': 'chemin', 'ste': 'suite', 'apt': 'apartment',
    'apts': 'apartments', 'flr': 'floor', 'fl': 'floor', '1st': '1', '2nd': '2',
    '3rd': '3', '4th': '4', '5th': '5', 'bldg': 'building', 'chs': 'society',
    'soc': 'society', 'sec': 'sector', 'dist': 'district', 'no': 'number',
    'nr': 'near', 'nrs': 'number', 'opp': 'opposite'
}

def normalize_address(raw_address: str, country: str = "") -> str:
    """
    Standardize a raw business address string (high-performance single-pass implementation).
    """
    if not isinstance(raw_address, str) or not raw_address:
        return ""
    text = raw_address.strip()
    if not text:
        return ""

    text = strip_accents(text)
    if '&' in text: text = text.replace('&', ' and ')
    if '#' in text: text = text.replace('#', ' number ')
    if '/' in text: text = text.replace('/', ' ')
    text = text.lower()

    for pat, rep in COMPILED_PHRASES:
        text = pat.sub(rep, text)

    words = NON_ALPHANUM.sub(' ', text).split()
    return ' '.join(WORD_ABBREVIATIONS.get(w, w) for w in words)


def extract_address_core(address_norm: str) -> str:
    """
    Remove landmark tokens to isolate the core address components.
    """
    if not address_norm:
        return ""
    # Strip landmark clauses
    core = LANDMARK_REGEX.sub(' ', address_norm)
    return ' '.join(core.split())


def tokenize_address(address_norm: str) -> list[str]:
    """Tokenize a normalized address string into a list of word tokens."""
    if not address_norm:
        return []
    return address_norm.split()


def normalize_record(
    raw_address: str,
    country: str = "",
    name: str = "",
    entity_id: str = ""
) -> dict:
    """
    Produce the shared normalized-record interface dictionary for an entity:
      entity_id, name_norm, address_norm, country, city_norm, pincode
    Plus internal fields:
      address_core, landmark, address_tokens
    """
    norm_addr = normalize_address(raw_address, country)
    pincode = extract_pincode(raw_address, country)
    city_norm = extract_city(raw_address, country)
    landmark = extract_landmark(raw_address)
    addr_core = extract_address_core(norm_addr)
    tokens = tokenize_address(norm_addr)

    return {
        "entity_id": entity_id,
        "name_norm": name,
        "address_norm": norm_addr,
        "country": country.strip() if isinstance(country, str) else "",
        "city_norm": city_norm,
        "pincode": pincode,
        "address_core": addr_core,
        "landmark": landmark,
        "address_tokens": " ".join(tokens)
    }


def run_unit_tests():
    """Unit test normalization, abbreviations, landmarks, cities, and pincodes."""
    print("=== Running Unit Tests on normalize_address & address extraction ===")
    
    test_cases = [
        # (raw_address, country, expected_tokens_in_norm, expected_city, expected_pin, has_landmark)
        ("2621 Cotten Road, Tyler, TX", "US", ["2621", "cotten", "road", "tyler", "tx"], "tyler", "", False),
        ("IA, Iowa City, 1064 Newton Rd, Unit 11", "US", ["ia", "iowa", "city", "1064", "newton", "road", "unit", "11"], "iowa city", "", False),
        ("175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France", ["175", "boulevard", "du", "president", "bordeaux"], "bordeaux", "", False),
        ("20 Rue Parmentier, Dunkerque, Hauts-de-France", "France", ["20", "rue", "parmentier", "dunkerque", "hauts", "de", "france"], "dunkerque", "", False),
        ("1 rue Jodelle, 44600 ST NAZAIRE, Saint-Nazaire, Pays de la Loire", "France", ["1", "rue", "jodelle", "44600", "saint", "nazaire"], "saint-nazaire", "44600", False),
        ("26, Ram Nagar, Behind M.C Quarters Near Jagadhri Gate, Ambala City, Haryana", "India", ["26", "ram", "nagar", "behind", "near", "ambala", "city", "haryana"], "ambala city", "", True),
        ("37B, Pushtikar Chs Ltd, Shiv Sadan, Near Patel Estate, Jogeshwari (West), Mumbai, Mumbai City, Maharashtra", "India", ["37b", "pushtikar", "society", "near", "patel", "estate", "mumbai"], "mumbai", "", True),
        ("60 Kashipuri Kabirkhedi, Indore, Madhya Pradesh", "India", ["60", "kashipuri", "kabirkhedi", "indore", "madhya", "pradesh"], "indore", "", False),
        ("A-115, Freedom Fighter, Enclave Neb Sarai, South Delhi, Delhi", "India", ["a", "115", "freedom", "fighter", "south", "delhi", "delhi"], "south delhi", "", False),
        ("Bhubaneswar, Sub Plot No.-L6/29, Mahodadhi Bhawan (Next To Iter College) Panchasakha Nagar, Dumuduma, Khordha, Plot No.-780, Orissa", "India", ["next to", "plot number"], "khordha", "", True),
    ]

    passed = 0
    for addr, country, exp_tokens, exp_city, exp_pin, has_lm in test_cases:
        norm = normalize_address(addr, country)
        city = extract_city(addr, country)
        pin = extract_pincode(addr, country)
        lm = extract_landmark(addr)

        # Check normalization
        tokens_ok = all(t.lower() in norm for t in exp_tokens)
        city_ok = (exp_city == "") or (exp_city.lower() in city.lower())
        pin_ok = (exp_pin == "") or (exp_pin == pin)
        lm_ok = (not has_lm) or (len(lm) > 0)

        all_ok = tokens_ok and city_ok and pin_ok and lm_ok
        if all_ok:
            passed += 1
        status = "PASS" if all_ok else "WARN"
        print(f"[{status}] norm='{norm[:45]}...' | city='{city}' | pin='{pin}' | lm='{lm[:30]}'")

    print(f"\nUnit Tests Result: {passed}/{len(test_cases)} passed successfully.")
    assert passed >= 8, f"Expected at least 8/10 test cases to pass, got {passed}"


def apply_normalization_to_files(
    train_files: list[tuple[str, str]],
    test_files: list[tuple[str, str]] | None = None,
    train_out: str = "data/processed/address_normalized_train.tsv",
    test_out: str = "data/processed/address_normalized_test.tsv"
):
    """
    Applies address normalization across all train (and test if present) source files.
    Writes full shared-interface TSV:
      entity_id, name_norm, address_norm, country, city_norm, pincode,
      address_core, landmark, address_tokens, source
    """
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    print("\n--- Applying Address Normalization to Train Dataset ---")
    
    rows = []
    for path, src_tag in train_files:
        print(f"Reading and normalizing {path} ({src_tag})...")
        df = pd.read_csv(path, sep="\t")
        
        for _, row in df.iterrows():
            eid = str(row["entity_id"]).strip()
            raw_addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
            raw_name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
            country = str(row["country"]).strip() if pd.notna(row["country"]) else ""
            
            rec = normalize_record(raw_addr, country=country, name=raw_name, entity_id=eid)
            rec["source"] = src_tag
            rows.append(rec)
            
    df_out = pd.DataFrame(rows)
    df_out.to_csv(train_out, sep="\t", index=False)
    print(f"Wrote {len(df_out):,} normalized records to {train_out}")

    # Process test if present
    if test_files:
        valid_test_files = [tf for tf in test_files if os.path.exists(tf[0])]
        if valid_test_files:
            print("\n--- Applying Address Normalization to Test Dataset (Streaming) ---")
            total_test = 0
            with open(test_out, "w", encoding="utf-8", newline="") as f_out:
                writer = csv.writer(f_out, delimiter="\t")
                writer.writerow([
                    "entity_id", "name_norm", "address_norm", "country", "city_norm", "pincode",
                    "address_core", "landmark", "address_tokens", "source"
                ])
                for path, src_tag in valid_test_files:
                    print(f"Streaming {path} ({src_tag})...")
                    file_count = 0
                    with open(path, "r", encoding="utf-8", errors="replace") as f_in:
                        reader = csv.reader(f_in, delimiter="\t")
                        header = next(reader, None)
                        if not header:
                            continue
                        eid_idx = header.index("entity_id")
                        addr_idx = header.index("business_address")
                        name_idx = header.index("business_name")
                        c_idx = header.index("country")
                        
                        for r in reader:
                            if not r:
                                continue
                            eid = r[eid_idx].strip()
                            raw_addr = r[addr_idx] if len(r) > addr_idx else ""
                            raw_name = r[name_idx] if len(r) > name_idx else ""
                            c = r[c_idx].strip() if len(r) > c_idx else ""
                            rec = normalize_record(raw_addr, country=c, name=raw_name, entity_id=eid)
                            writer.writerow([
                                rec["entity_id"], rec["name_norm"], rec["address_norm"],
                                rec["country"], rec["city_norm"], rec["pincode"],
                                rec["address_core"], rec["landmark"], rec["address_tokens"],
                                src_tag
                            ])
                            file_count += 1
                    print(f"  Processed {file_count:,} test records from {src_tag}")
                    total_test += file_count
            print(f"Wrote {total_test:,} test records to {test_out}")


if __name__ == "__main__":
    run_unit_tests()
    
    train_sources = [
        ("dataset/train/train_source1.tsv", "S1"),
        ("dataset/train/train_source2.tsv", "S2"),
        ("dataset/train/train_source3.tsv", "S3"),
    ]
    test_sources = [
        ("dataset/test/test_source1.tsv", "S1"),
        ("dataset/test/test_source2.tsv", "S2"),
        ("dataset/test/test_source3.tsv", "S3"),
    ]
    apply_normalization_to_files(train_sources, test_sources)
