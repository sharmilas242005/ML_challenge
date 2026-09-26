"""
Step 8 — Testing and Validation Suite for Person 2 (Address Domain Track)
Amazon ML Challenge 2026: Business Entity Resolution

Validates:
1. Normalization (lowercasing, whitespace collapsing, punctuation standardization, diacritics stripping)
2. Abbreviation expansion (rd -> road, st -> street, ave -> avenue, opp -> opposite, nr -> near, etc.)
3. Pincode / ZIP extraction (US 5-digit, India 6-digit, France 5-digit, open-ended)
4. City extraction (standard and inverted address components)
5. Landmark preservation (separating landmark phrases without deleting useful address components)
6. Missing addresses handling (graceful fallback, zero errors)
7. Duplicate candidates check (guarantee set uniqueness per S1 entity)
8. Source 1 -> Source 2/Source 3 restriction (no S1->S1 self matches, only valid S2/S3 IDs)
9. Feature calculations (finite floats, correct ranges [0, 1], stable column names)
10. Train/Test schema consistency (shared interface and feature consistency)
"""

import os
import sys
import unittest
import pandas as pd
from collections import Counter

# Set stdout encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

# Import modules from src.address_track
from src.address_track.normalize_address import (
    normalize_address,
    extract_pincode,
    extract_city,
    extract_landmark,
    extract_address_core,
    normalize_record
)
from src.address_track.features_address import (
    fast_levenshtein_distance,
    char_ngram_cosine_sim,
    compute_address_pair_features
)


class TestAddressTrack(unittest.TestCase):

    def test_01_normalization(self):
        """Test lowercasing, whitespace, accents, and punctuation."""
        raw = "   49  Rue du Maréchal Brune,   Tourcoing,  Hauts-de-France!  "
        norm = normalize_address(raw, "France")
        self.assertEqual(norm, "49 rue du marechal brune tourcoing hauts de france")

        # Conjunction & symbols
        raw2 = "Smith & Sons #12 / B"
        norm2 = normalize_address(raw2, "US")
        self.assertIn("and", norm2)
        self.assertIn("number 12", norm2)

    def test_02_abbreviation_expansion(self):
        """Test standard address abbreviations expansion without token corruption."""
        test_map = {
            "2621 Cotten Rd": "2621 cotten road",
            "600 SE 2nd St": "600 se 2 street",
            "100 Hartsdale Ave": "100 hartsdale avenue",
            "3105 Stonegate Dr": "3105 stonegate drive",
            "A-606 Highway Ste 12": "a 606 highway suite 12",
            "Near Patel Estate Opp Bank": "near patel estate opposite bank",
            "Plot No. 12 Flr 2": "plot number 12 floor 2"
        }
        for raw, expected in test_map.items():
            norm = normalize_address(raw)
            self.assertEqual(norm, expected, f"Failed expanding: {raw}")

    def test_03_pincode_extraction(self):
        """Test postal code extraction across countries."""
        # France
        pin_fr = extract_pincode("1 rue Jodelle, 44600 ST NAZAIRE, Saint-Nazaire, Pays de la Loire", "France")
        self.assertEqual(pin_fr, "44600")

        # India
        pin_in = extract_pincode("Near Jagadhri Gate, Ambala City, Haryana 134003", "India")
        self.assertEqual(pin_in, "134003")

        # US
        pin_us = extract_pincode("2621 Cotten Road, Tyler, TX 75701", "US")
        self.assertEqual(pin_us, "75701")

        # Street number must NOT be extracted as pincode
        no_pin = extract_pincode("14028 Irving Hill Drive, Knightdale, NC", "US")
        self.assertNotEqual(no_pin, "14028")

    def test_04_city_extraction(self):
        """Test city extraction from standard and inverted components."""
        # Standard US
        c1 = extract_city("2621 Cotten Road, Tyler, TX", "US")
        self.assertEqual(c1.lower(), "tyler")

        # Inverted US
        c2 = extract_city("IA, Iowa City, 1064 Newton Rd, Unit 11", "US")
        self.assertEqual(c2.lower(), "iowa city")

        # Standard France
        c3 = extract_city("175 Boulevard du President Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France")
        self.assertEqual(c3.lower(), "bordeaux")

        # Inverted France
        c4 = extract_city("Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac", "France")
        self.assertEqual(c4.lower(), "la teste-de-buch")

    def test_05_landmark_preservation(self):
        """Test landmark detection and separation."""
        raw = "37B, Pushtikar Chs Ltd, Shiv Sadan, Near Patel Estate, Jogeshwari, Mumbai"
        lm = extract_landmark(raw)
        self.assertIn("near patel estate", lm.lower())

        core = extract_address_core(normalize_address(raw))
        self.assertNotIn("near patel estate", core)

    def test_06_missing_addresses_handling(self):
        """Test that None, NaN, empty strings degrade gracefully."""
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_address(None), "")
        self.assertEqual(extract_city(""), "")
        self.assertEqual(extract_pincode(""), "")
        self.assertEqual(extract_landmark(""), "")

        rec = normalize_record("", country="US", name="Test Inc", entity_id="S1-001")
        self.assertEqual(rec["entity_id"], "S1-001")
        self.assertEqual(rec["address_norm"], "")
        self.assertEqual(rec["pincode"], "")

    def test_07_no_duplicate_candidates(self):
        """Verify address_candidates.tsv contains no duplicate candidate IDs per row."""
        cands_file = "data/processed/address_candidates.tsv"
        self.assertTrue(os.path.exists(cands_file))
        df = pd.read_csv(cands_file, sep="\t", dtype=str).fillna("")
        for _, row in df.iterrows():
            c_str = row["candidate_entity_ids"].strip()
            if c_str:
                c_list = [c.strip() for c in c_str.split(",") if c.strip()]
                self.assertEqual(len(c_list), len(set(c_list)), f"Duplicate candidate found for {row['source1_entity_id']}")

    def test_08_source1_restriction(self):
        """Verify candidates only contain valid S2/S3 IDs and no S1 self-matches."""
        cands_file = "data/processed/address_candidates.tsv"
        df = pd.read_csv(cands_file, sep="\t", dtype=str).fillna("")
        for _, row in df.iterrows():
            s1_id = row["source1_entity_id"]
            self.assertTrue(s1_id.startswith("S1-"))
            c_str = row["candidate_entity_ids"].strip()
            if c_str:
                for c in c_str.split(","):
                    c = c.strip()
                    self.assertTrue(c.startswith("S2-") or c.startswith("S3-"), f"Invalid candidate ID: {c}")
                    self.assertNotEqual(s1_id, c, "Self-match detected!")

    def test_09_feature_calculations(self):
        """Test feature values are within expected mathematical ranges."""
        r1 = {"address_norm": "2621 cotten road tyler tx", "pincode": "75701", "city_norm": "tyler", "country": "US", "landmark": ""}
        r2 = {"address_norm": "2621 cotten road tyler tx", "pincode": "75701", "city_norm": "tyler", "country": "US", "landmark": ""}
        r3 = {"address_norm": "100 different street dallas tx", "pincode": "75001", "city_norm": "dallas", "country": "US", "landmark": ""}

        # Identical pair
        feats_ident = compute_address_pair_features(r1, r2)
        # levenshtein_sim, token_jaccard, tfidf_cosine, exact_pincode_match, exact_city_match, exact_country_match, exact_address_match
        self.assertEqual(feats_ident[0], 1.0)
        self.assertEqual(feats_ident[1], 1.0)
        self.assertEqual(feats_ident[2], 1.0)
        self.assertEqual(feats_ident[3], 1)
        self.assertEqual(feats_ident[4], 1)
        self.assertEqual(feats_ident[5], 1)
        self.assertEqual(feats_ident[6], 1)

        # Different pair
        feats_diff = compute_address_pair_features(r1, r3)
        self.assertLess(feats_diff[0], 0.7)
        self.assertLess(feats_diff[1], 0.5)
        self.assertEqual(feats_diff[3], 0)  # Pincode mismatch
        self.assertEqual(feats_diff[4], 0)  # City mismatch
        self.assertEqual(feats_diff[5], 1)  # Same country (US)

    def test_10_schema_consistency(self):
        """Verify generated files adhere to required column schemas."""
        # 1. Normalized record schema
        norm_file = "data/processed/address_normalized_train.tsv"
        self.assertTrue(os.path.exists(norm_file))
        df_norm = pd.read_csv(norm_file, sep="\t", nrows=2)
        expected_norm_cols = [
            "entity_id", "name_norm", "address_norm", "country", "city_norm", "pincode"
        ]
        for col in expected_norm_cols:
            self.assertIn(col, df_norm.columns)

        # 2. Candidate file schema
        cands_file = "data/processed/address_candidates.tsv"
        df_cands = pd.read_csv(cands_file, sep="\t", nrows=2)
        self.assertListEqual(df_cands.columns.tolist(), ["source1_entity_id", "candidate_entity_ids"])

        # 3. Features file schema
        feats_file = "data/processed/address_features.tsv"
        self.assertTrue(os.path.exists(feats_file))
        df_feats = pd.read_csv(feats_file, sep="\t", nrows=2)
        expected_feat_cols = [
            "source1_entity_id", "candidate_entity_id", "levenshtein_sim",
            "token_jaccard", "tfidf_cosine", "exact_pincode_match",
            "exact_city_match", "exact_country_match", "label"
        ]
        for col in expected_feat_cols:
            self.assertIn(col, df_feats.columns)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAddressTrack)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
