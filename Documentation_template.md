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

### 4.1 Overview

Person 3 implements the vertical slice from raw candidate lists → merged feature matrix → trained classifier → validated predictions. It consumes the outputs of Person 1 (name track) and Person 2 (address track) without modifying their code, and produces the model artifact and validation predictions that Person 4 will use for threshold tuning.

**Module location:** `src/model/`  
**Reproduce with:** `python -m src.model.train`

---

### 4.2 Candidate Union Strategy (`src/model/merge_features.py`)

Both candidate sets are in long-pair format (`source1_entity_id`, `candidate_entity_id`, `source`, `block_reason`):

| Input | Pairs |
| :--- | ---: |
| `data/processed/name_candidates_train.tsv` | 436,133 |
| `data/processed/address_candidates_train.tsv` | 328,864 |

**Union procedure:**
1. Load both candidate files.
2. Tag provenance: `from_name=1` and `from_address=1` on each side.
3. Outer-join on `(source1_entity_id, candidate_entity_id)` — this is the set union.
4. Deduplicate: each pair appears exactly once regardless of which track generated it.
5. Fill provenance flags (0 where absent).

| Pairs | Count |
| :--- | ---: |
| Name-only | 418,958 |
| Address-only | 311,689 |
| From both tracks | 17,175 |
| **Union total** | **747,822** |

---

### 4.3 Feature Merge Strategy

Features are left-joined onto the union candidate set by `(source1_entity_id, candidate_entity_id)`:

- **Name features** (6 columns, from Person 1's `name_features_train.tsv`): `levenshtein_sim`, `token_jaccard`, `char_ngram_cos`, `token_sort_ratio`, `token_set_ratio`, `exact_match`
- **Address features** (13 columns, from Person 2's `address_features_train.tsv`): renamed with `addr_` prefix to avoid collision — `addr_levenshtein_sim`, `addr_token_jaccard`, `addr_tfidf_cosine`, `addr_exact_pincode_match`, `addr_exact_city_match`, `addr_exact_country_match`, `addr_exact_address_match`, `addr_char_ngram_cos`, `addr_token_overlap_count`, `addr_address_len_ratio`, `addr_digit_overlap_ratio`, `addr_house_number_match`, `addr_landmark_sim`

**Missing value handling:** Pairs generated by only one track have no features from the other track. All missing values are filled with `0.0` — semantically correct since `0.0` represents no similarity.

**Total features:** 19 numeric columns.

**Output:** `data/processed/merged_features_train.tsv` — 747,822 rows × 24 columns (19 features + 2 IDs + 2 provenance flags + label).

---

### 4.4 Label Construction

Labels are assigned centrally from `dataset/train/train_ground_truth.tsv`:

- Ground truth format: `source1_entity_id` → `matched_entity_ids` (comma-separated, may be empty for singletons).
- A candidate pair `(s1_id, cand_id)` receives `label=1` if `cand_id` is in the parsed set of matched IDs for `s1_id`.
- All other pairs receive `label=0`.
- Multi-match S1 entities (matched to both S2 and S3) are handled correctly — both IDs yield `label=1`.
- The label in the source feature files is **ignored and re-assigned centrally** to guarantee consistency across the union.

| Label | Count |
| :--- | ---: |
| Positive (label=1) | 15,769 |
| Negative (label=0) | 732,053 |
| Neg/Pos ratio | 46.4:1 |

---

### 4.5 Candidate Recall Ceiling

Evaluated before model training. The combined recall ceiling is the hard upper bound on any classifier trained on this candidate set.

| Metric | Value |
| :--- | ---: |
| Name-only recall | 95.46% (15,053 / 15,769) |
| Address-only recall | **100.00%** (15,769 / 15,769) |
| **Combined recall ceiling** | **100.00%** (15,769 / 15,769) |
| Missed true matches | **0** |
| Combined candidate pairs | 747,822 |

The 716 true pairs captured exclusively by the address track (not in name candidates) confirm the value of the union approach — they would have been irrecoverably lost with a name-only candidate set.

---

### 4.6 Entity-Level Validation Split

Split is performed by `source1_entity_id` so all candidate rows for the same Source-1 entity stay in the same fold. This prevents data leakage (no entity appears in both train and validation).

- **Split:** 80% train / 20% validation
- **Random seed:** 42 (fully reproducible)

| Split | Entities | Rows | Positives | Negatives | Ratio |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Train | 12,000 | 598,046 | 12,601 | 585,445 | 46.5:1 |
| Validation | 2,999 | 149,776 | 3,168 | 146,608 | 46.3:1 |

---

### 4.7 F0.5 Implementation (`src/model/evaluate.py`)

The challenge-specific F0.5 metric is implemented in `score_f05_entity_level()`.

**Formula:**
$$F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$$

**Per-entity scoring:**
- $P = |\text{predicted} \cap \text{true}| \;/\; |\text{predicted}|$ — undefined predictions default to $P=1.0$ if no true matches (singleton), else $P=0.0$.
- $R = |\text{predicted} \cap \text{true}| \;/\; |\text{true}|$ — undefined recall for singletons defaults to $R=1.0$.
- Singleton correctly predicted (no predictions, no true matches): $F_{0.5} = 1.0$.
- Singleton with a false-positive prediction: $F_{0.5} = 0.0$.

**Macro average** = arithmetic mean of per-entity $F_{0.5}$ scores across all Source-1 entities in the evaluation set.

---

### 4.8 Model Choice & Imbalance Handling (`src/model/train.py`)

**Model:** XGBoost binary classifier (`XGBClassifier`, `objective=binary:logistic`)

**Rationale:** XGBoost 3.2.0 is available in the project environment; LightGBM and CatBoost are not installed. XGBoost provides native handling for class imbalance via `scale_pos_weight`.

**Hyperparameters (baseline):**

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `n_estimators` | 500 (early stop) | Upper bound; early stopping prevents overfitting |
| `max_depth` | 6 | Standard tree depth for tabular entity matching |
| `learning_rate` | 0.05 | Conservative learning rate for stable convergence |
| `subsample` | 0.8 | Row subsampling for regularization |
| `colsample_bytree` | 0.8 | Column subsampling for regularization |
| `scale_pos_weight` | 46.46 | `neg_count / pos_count = 585,445 / 12,601` |
| `early_stopping_rounds` | 30 | Stops at best val logloss iteration |
| `random_state` | 42 | Reproducibility |

**Early stopping:** Best iteration at 238 / 500.

---

### 4.9 Validation Methodology & Results

Predictions at threshold 0.5 (baseline — Person 4 will tune the threshold).

| Metric | Value |
| :--- | ---: |
| Entity-level Macro F0.5 | **0.9996** |
| Macro Precision | **0.9995** |
| Macro Recall | **1.0000** |
| Source-1 entities evaluated | 2,999 |
| Predicted matches | 3,172 |
| Actual matches | 3,168 |
| Correct predictions | 3,168 |

**Distinction:**
- *Candidate recall ceiling* (1.0000) = how many true pairs are reachable in principle.
- *Model validation F0.5* (0.9996) = how well the classifier predicts within those reachable pairs.

The 4 false positives (3,172 predicted vs 3,168 actual) at threshold 0.5 give P≈0.9987 globally, though macro-avg precision is 0.9995 since most S1 entities have no false positives.

---

### 4.10 Feature Importance

Extracted from the trained XGBoost model. Saved to `data/processed/feature_importance.tsv`.

| Rank | Feature | Importance | Track |
| ---: | :--- | ---: | :--- |
| 1 | `addr_levenshtein_sim` | 0.7950 | Address |
| 2 | `addr_exact_address_match` | 0.1424 | Address |
| 3 | `addr_token_jaccard` | 0.0624 | Address |
| 4 | `char_ngram_cos` | 0.000032 | Name |
| 5 | `levenshtein_sim` | 0.000021 | Name |
| 6 | `token_sort_ratio` | 0.000020 | Name |
| 7 | `token_set_ratio` | 0.000018 | Name |
| 8 | `addr_token_overlap_count` | 0.000010 | Address |
| 9 | `addr_house_number_match` | 0.000010 | Address |
| 10 | `addr_digit_overlap_ratio` | 0.000009 | Address |

**Feedback for Person 1 (name track):** Name features collectively contribute ~0.01% of total importance on the training data. This is explained by the EDA finding that 100% of true matches share an identical raw `business_address` in training data — the address signal dominates. On test data with noisy addresses, name features are expected to recover more importance. `char_ngram_cos`, `levenshtein_sim`, and the `token_sort/set_ratio` features (from `rapidfuzz`) are the most useful name signals.

**Feedback for Person 2 (address track):** `addr_levenshtein_sim` (79.5%), `addr_exact_address_match` (14.2%), and `addr_token_jaccard` (6.2%) together account for 99.99% of total importance. The exact-address boolean flag is a particularly powerful signal — consistent with the EDA result that training addresses are identical. `addr_tfidf_cosine`, `addr_char_ngram_cos`, `addr_exact_country_match`, `addr_address_len_ratio`, and `addr_landmark_sim` have near-zero importance on training data but may activate on test data with more variation.

---

### 4.11 Model Artifact & Output Paths

| Artifact | Path |
| :--- | :--- |
| Trained model (pickle) | `output/entity_match_model.pkl` |
| Feature importance | `data/processed/feature_importance.tsv` |
| Validation predictions | `data/processed/validation_predictions.tsv` |
| Merged candidates | `data/processed/merged_candidates_train.tsv` |
| Merged feature matrix | `data/processed/merged_features_train.tsv` |

`validation_predictions.tsv` columns: `source1_entity_id`, `candidate_entity_id`, `label`, `prediction_probability`, `prediction_at_0_5` — ready for Person 4's threshold tuning.

The model pickle contains: `{"model": XGBClassifier, "feature_cols": [...], "random_seed": 42, "val_fraction": 0.20}`.

---

### 4.12 Reproducibility

```bash
python -m src.model.train
```

The pipeline is fully self-contained:
1. Detects existing merged features file (skips re-merge on re-run; use `force_remerge=True` to regenerate).
2. Reproducible entity-level split via `numpy.default_rng(seed=42)`.
3. XGBoost `random_state=42`.
4. All intermediate files saved to `data/processed/`.
5. Model and predictions saved to `output/` and `data/processed/`.

**Test suite:**
```bash
python -m src.model.test_model
```
29 tests covering: candidate union/deduplication, label assignment, entity split, F0.5 formula, singleton handling, missing feature fill, and ID leakage guard.

---

## 5. Person 4 — Threshold Tuning, Pipeline Integration & Packaging
*(To be completed by Person 4)*
