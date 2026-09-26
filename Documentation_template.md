# Amazon ML Challenge 2026 — Business Entity Resolution
## Technical Documentation

---

## 1. Project Overview & Shared Architecture
- **Problem Statement**: Resolve multi-source business entities (Source 1 $\to$ Source 2 & Source 3) under strict precision-weighted evaluation ($F_{0.5}$).
- **Shared Normalized Record Schema**: `entity_id`, `name_norm`, `address_norm`, `country`, `city_norm`, `pincode`
- **Candidate Pair Schema (Long)**: `source1_entity_id`, `candidate_entity_id`, `source`, `block_reason`
- **Candidate Output (Challenge)**: `source1_entity_id`, `candidate_entity_ids` (comma-separated, every S1 entity represented)

---

## 2. Person 1 — Name-Domain Track
*(To be completed by Person 1)*

---

## 3. Person 2 — Address-Domain Track

### 3.1 Exploratory Data Analysis (EDA) Findings
Thorough exploratory analysis of the raw training datasets (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`) revealed the following concrete data characteristics:
- **Volume & Completeness**:
  - Source 1: 15,000 entities, 0 missing addresses (0.0%), 0 missing countries.
  - Source 2: 17,845 entities, 277 missing addresses (1.55%), 0 missing countries.
  - Source 3: 17,924 entities, 274 missing addresses (1.53%), 0 missing countries.
  - Across all 15,769 true-match pairs in `train_ground_truth.tsv`, 0 missing addresses were observed in Source 2 or Source 3.
- **Country Distribution**:
  - The dataset spans an open set of countries: India (46.3% S1, 46.8% S2, 46.8% S3), United States (39.0% S1, 38.4% S2, 38.9% S3), and France (14.7% S1, 14.8% S2, 14.3% S3).
  - True-match pairs are strictly partitioned within identical country domains.
- **Address Identity in Ground Truth**:
  - In `train_ground_truth.tsv`, 100.00% (15,769 / 15,769) of true matches share an identical raw `business_address` string between Source 1 and Source 2/Source 3.
  - While identical in training data, test data requires robust handling of reordered components, abbreviation discrepancies, municipal numbering variations, and landmark phrases.
- **Address Components & Delimiters**:
  - Addresses are predominantly comma-delimited (mean 3.1–3.3 commas per record).
  - Standard component ordering: `<Street / House No>, <City>, <State / Region>`.
  - Inverted / reordered patterns frequently appear: `<State / Region>, <City>, <Street>` or `<City>, <Street>, <Region>`.
  - India records feature municipal/plot notations (`D-61`, `S No. 42/2/3`, `A-115`, `Ews 12`) and extensive landmarks.
  - France records feature French thoroughfare designations (`Rue`, `Boulevard`, `Avenue`, `Impasse`, `Chemin`), number qualifiers (`bis`, `ter`), and combining diacritics (`é`, `è`, `ê`, `ç`).
  - US records feature directionals (`SE`, `NW`), street suffixes (`Road`, `Street`, `Avenue`, `Boulevard`, `Drive`), and unit markers (`Unit`, `Apt`, `Suite`).
- **Landmark Prevalence**:
  - Approximately 13.8% of S1 addresses contain explicit landmark indicator keywords (`near`, `opp`, `opposite`, `behind`, `beside`, `next to`, `in front of`, `adjacent to`, `above`, `below`).
- **Postal Code / Pincode Sparsity**:
  - Pincodes are explicitly embedded in only ~10.5% of US addresses, ~0.1% of India addresses, and ~0.5%–5.5% of France addresses in the raw text. Hence, pincode cannot serve as a mandatory filter.

---

### 3.2 Address Normalization Strategy (`src/address_track/normalize_address.py`)
Address normalization operates deterministically across both training and test data without external APIs:
1. **Unicode NFKD Accent Stripping**: Normalizes accented characters (`é` $\to$ `e`, `ç` $\to$ `c`, `ô` $\to$ `o`) to avoid encoding and transliteration mismatches.
2. **Conjunction & Symbol Standardization**: Standardizes `&` $\to$ `and`, `#` $\to$ `number`, `/` $\to$ whitespace.
3. **Abbreviation Standardization**:
   - Compiles word-bounded regular expressions for standard street and building terms across jurisdictions:
     - Street types: `rd` $\to$ `road`, `st` $\to$ `street`, `ave`/`av` $\to$ `avenue`, `blvd`/`bd`/`bvd` $\to$ `boulevard`, `dr` $\to$ `drive`, `hwy` $\to$ `highway`, `ln` $\to$ `lane`, `ct` $\to$ `court`, `pkwy` $\to$ `parkway`, `pl` $\to$ `place`, `rte` $\to$ `route`, `all` $\to$ `allee`, `imp` $\to$ `impasse`, `chem` $\to$ `chemin`.
     - Sub-units: `ste` $\to$ `suite`, `apt`/`apts` $\to$ `apartment`, `fl`/`flr` $\to$ `floor`, `bldg` $\to$ `building`, `sec` $\to$ `sector`, `dist` $\to$ `district`, `no`/`nrs` $\to$ `number`, `chs`/`soc` $\to$ `society`.
     - Landmark prepositions: `opp` $\to$ `opposite`, `nr` $\to$ `near`.
   - Carefully avoids single-letter replacements (`r`, `a`, `c`) that could corrupt valid words or initials.
4. **Punctuation & Whitespace Collapsing**: Removes non-alphanumeric characters while preserving token boundaries, collapsing multiple whitespace characters.
5. **Output Interface**:
   Generates `data/processed/address_normalized_train.tsv` matching the team interface:
   `entity_id`, `name_norm`, `address_norm`, `country`, `city_norm`, `pincode`, plus internal fields `address_core`, `landmark`, `address_tokens`, and `source`.

---

### 3.3 Component Extraction: Pincode, City, Landmark
- **Pincode Extraction**:
  - Country-aware regexes: India 6-digit PIN (`\b[1-9]\d{5}\b`), France 5-digit postal code (`\b\d{5}\b`), US 5-digit ZIP (`\b[A-Z]{2}\s+(\d{5})\b`), and open-ended fallback.
  - Rigorous street-number exclusion: Distinguishes leading street numbers (`14028 Irving Hill Drive`) from actual postal codes.
- **City Extraction**:
  - Position-aware heuristic analyzing comma-separated components against known states/regions.
  - Correctly extracts cities from standard formats (`..., Tyler, TX` $\to$ `tyler`), inverted formats (`IA, Iowa City, ...` $\to$ `iowa city`), and French regional layouts (`Nouvelle-Aquitaine, La Teste-de-Buch, ...` $\to$ `la teste-de-buch`).
  - Rejects clauses containing street keywords or leading house numbers.
- **Landmark Preservation**:
  - Captures landmark clauses (`near ...`, `opposite ...`, `behind ...`, `beside ...`) into a dedicated `landmark` field.
  - Preserves clean core address components in `address_core` while keeping the full text in `address_norm`.

---

### 3.4 Multi-Pass Address Blocking (`src/address_track/blocking_address.py`)
To reduce the $15,000 \times 35,769 = 536,535,000$ brute-force comparison space while preserving maximal true-pair recall, a 4-pass union blocking architecture is implemented:
- **Pass 0 — Exact Normalized Address Match**: Direct hash lookup on `address_norm`. Captures identical address records with near-zero false positive rate.
- **Pass A — Geographic Component Blocking**:
  - Same Country + Same Pincode (when pincode present).
  - Same Country + Same City (when city present, bucket-size capped to prevent explosion).
- **Pass B — Distinctive Token Inverted Index**:
  - Indexes content tokens of length $\ge 4$ excluding generic address stopwords (`road`, `street`, `avenue`, `floor`, `apartment`, `suite`, etc.).
  - Bucket sizes capped at 500 to keep memory and complexity bounded.
- **Pass C — N-Gram / Character Prefix Blocking**:
  - Indexes 6-character normalized address prefixes to handle minor spelling discrepancies, OCR errors, and partial addresses.
- **Pass D — Priority Ranking & Candidate Capping**:
  - Combines and deduplicates candidate pairs per Source 1 entity.
  - Scores candidates by matching criteria (Pass 0 $\to 100$ pts, Pass A $\to 15$ pts, Pass B $\to 8$ pts, Pass C $\to 4$ pts).
  - Selects the top 25 candidates per Source 1 entity.
  - Strictly enforces: Source 1 $\to$ Source 2/Source 3 only, zero self-matches, zero duplicate candidate IDs.
  - Produces both challenge format (`address_candidates.tsv`) with all 15,000 Source 1 rows and long format (`address_candidates_train.tsv`).

---

### 3.5 Address Similarity Features (`src/address_track/features_address.py`)
Computes 13 stable, normalized numerical address features for every candidate pair:
1. `levenshtein_sim`: Normalized edit similarity ($1 - \frac{\text{edit\_dist}}{\max(L_1, L_2)}$).
2. `token_jaccard`: Word-token Jaccard similarity ($\frac{|T_1 \cap T_2|}{|T_1 \cup T_2|}$).
3. `tfidf_cosine`: Character 3-gram cosine similarity (fast n-gram TF-IDF proxy).
4. `exact_pincode_match`: Boolean flag (1 if both have matching non-empty pincodes, else 0).
5. `exact_city_match`: Boolean flag (1 if both have matching non-empty cities, else 0).
6. `exact_country_match`: Boolean flag (1 if both have matching countries, else 0).
7. `exact_address_match`: Boolean flag (1 if normalized address strings are identical, else 0).
8. `char_ngram_cos`: Character 3-gram cosine overlap.
9. `token_overlap_count`: Integer count of shared distinctive tokens.
10. `address_len_ratio`: Ratio of string lengths ($\frac{\min(L_1, L_2)}{\max(L_1, L_2)}$).
11. `digit_overlap_ratio`: Jaccard similarity of extracted numerical digit sequences.
12. `house_number_match`: Boolean flag (1 if leading building/house numbers match, else 0).
13. `landmark_sim`: Word Jaccard similarity between extracted landmark clauses.
- **Label Column**: Populated as 1 (positive) or 0 (negative) for training split based on `train_ground_truth.tsv`.

---

### 3.6 Quantitative Evaluation & Verification Results

Evaluated via `src/address_track/evaluate_address_blocking.py` against `dataset/train/train_ground_truth.tsv`:

$$\text{Recall Ceiling} = \frac{\text{True Matching Pairs Captured in Candidate Set}}{\text{Total True Matching Pairs in Ground Truth}} = \frac{15,769}{15,769} = \mathbf{100.00\%}$$

$$\text{Search Space Reduction Ratio} = 1.0 - \frac{\text{Candidate Pairs Generated}}{\text{Total S1} \times \text{Total S2/S3}} = 1.0 - \frac{328,864}{15,000 \times 35,769} = \mathbf{0.99938706\ (99.9387\%)}$$

| Metric | Address Track Value |
| :--- | :--- |
| **Total Source 1 Entities** | 15,000 |
| **Total True Matching Pairs** | 15,769 |
| **Captured True Matching Pairs** | **15,769 (100.00%)** |
| **Full Recall Entities (100% matches captured)** | **11,968 / 11,968 (100.00%)** |
| **Zero Recall Entities (missed entirely)** | **0 / 11,968 (0.00%)** |
| **Total Candidate Pairs Generated** | 328,864 |
| **Average Candidates per Source 1 Entity** | 21.92 |
| **Search Space Reduction Ratio** | 99.938706% |
| **France Recall Ceiling** | 2,320 / 2,320 (100.00%) |
| **India Recall Ceiling** | 7,279 / 7,279 (100.00%) |
| **US Recall Ceiling** | 6,170 / 6,170 (100.00%) |

---

### 3.7 Important Implementation Decisions & Limitations
- **Decisions**:
  1. *Country-by-Country Processing*: Memory footprint is bounded at $<200$ MB during candidate generation.
  2. *Dual Output Formats*: Provides both the official challenge format (`address_candidates.tsv`) and the pair format (`address_candidates_train.tsv`) for seamless integration with Person 3.
  3. *Exact S1 Row Order*: Preserves the natural Source 1 entity order (`S1-TR000001` $\dots$ `S1-TR015000`) with empty candidate strings for singletons.
  4. *Feature Memoization*: Cached pair computations speed up feature extraction across repeated addresses.
- **Limitations**:
  1. *Missing Address Fallback*: Approximately 1.5% of Source 2/3 entities lack an address; these cannot be retrieved via address blocking alone and rely on Person 1's name-domain candidates during Person 3's candidate union.
  2. *Country Partition Assumption*: Records are currently blocked within matching country shards. If cross-country entity moves occur, geographic blocking will not capture them.

---

## 4. Person 3 — Candidate Merge, Model Training & Local Evaluation
*(To be completed by Person 3)*

---

## 5. Person 4 — Threshold Tuning, Pipeline Integration & Packaging
*(To be completed by Person 4)*
