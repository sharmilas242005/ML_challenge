"""
Person 3 — Test Suite
Amazon ML Challenge 2026: Business Entity Resolution

Tests:
  1. Candidate union/deduplication
  2. Ground-truth label assignment
  3. Entity-level split (no leakage)
  4. F0.5 calculation
  5. Singleton handling
  6. Missing feature handling
  7. No ID leakage into model features
  8. Feature column identification
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.model.evaluate import (
    f05_score_single,
    score_f05_entity_level,
    entity_level_split,
)
from src.model.merge_features import (
    load_ground_truth,
    ALL_FEATURE_COLS,
)
from src.model.train import get_feature_cols, check_no_id_leakage


class TestCandidateUnion(unittest.TestCase):
    """Test candidate union and deduplication logic."""

    def test_union_no_duplicates(self):
        """Union of two candidate sets should produce deduplicated pairs."""
        name_cands = pd.DataFrame([
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001"},
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-002"},
            {"source1_entity_id": "S1-002", "candidate_entity_id": "S2-003"},
        ])
        addr_cands = pd.DataFrame([
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001"},  # duplicate
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-004"},
            {"source1_entity_id": "S1-003", "candidate_entity_id": "S2-005"},
        ])

        name_cands["from_name"] = 1
        addr_cands["from_address"] = 1
        name_pairs = name_cands[["source1_entity_id", "candidate_entity_id", "from_name"]].drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"]
        )
        addr_pairs = addr_cands[["source1_entity_id", "candidate_entity_id", "from_address"]].drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"]
        )

        merged = pd.merge(name_pairs, addr_pairs,
                          on=["source1_entity_id", "candidate_entity_id"], how="outer")

        # 5 unique pairs: S1-001 x {S2-001, S2-002, S2-004}, S1-002 x S2-003, S1-003 x S2-005
        self.assertEqual(len(merged), 5, f"Expected 5 unique pairs, got {len(merged)}")

        # S2-001 should appear from both tracks
        overlap = merged[merged["candidate_entity_id"] == "S2-001"]
        self.assertEqual(len(overlap), 1, "Duplicate pair S2-001 should be deduplicated")
        self.assertEqual(overlap.iloc[0]["from_name"], 1)
        self.assertEqual(overlap.iloc[0]["from_address"], 1)

    def test_name_only_pair_preserved(self):
        """Pairs from name track only must survive union."""
        name_cands = pd.DataFrame([
            {"source1_entity_id": "S1-010", "candidate_entity_id": "S2-099", "from_name": 1},
        ])
        addr_cands = pd.DataFrame([
            {"source1_entity_id": "S1-010", "candidate_entity_id": "S2-100", "from_address": 1},
        ])

        merged = pd.merge(
            name_cands.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]),
            addr_cands.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]),
            on=["source1_entity_id", "candidate_entity_id"], how="outer"
        )
        self.assertEqual(len(merged), 2)

    def test_addr_only_pair_preserved(self):
        """Pairs from address track only must survive union."""
        name_cands = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "from_name"])
        addr_cands = pd.DataFrame([
            {"source1_entity_id": "S1-010", "candidate_entity_id": "S2-100", "from_address": 1},
        ])

        merged = pd.merge(
            name_cands.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]),
            addr_cands.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]),
            on=["source1_entity_id", "candidate_entity_id"], how="outer"
        )
        self.assertEqual(len(merged), 1)


class TestLabelAssignment(unittest.TestCase):
    """Test ground-truth label assignment."""

    def setUp(self):
        """Create a mini ground truth map."""
        self.gt_map = {
            "S1-001": {"S2-001", "S3-001"},  # matches both sources
            "S1-002": {"S2-002"},             # single match
            "S1-003": set(),                  # singleton
        }
        # Build positive pairs set
        self.positive_pairs: set[tuple[str, str]] = set()
        for s1, matches in self.gt_map.items():
            for m in matches:
                self.positive_pairs.add((s1, m))

    def test_positive_label_assigned(self):
        """True match pairs must receive label=1."""
        self.assertIn(("S1-001", "S2-001"), self.positive_pairs)
        self.assertIn(("S1-001", "S3-001"), self.positive_pairs)
        self.assertIn(("S1-002", "S2-002"), self.positive_pairs)

    def test_negative_label_assigned(self):
        """Non-match pairs must receive label=0."""
        self.assertNotIn(("S1-001", "S2-999"), self.positive_pairs)
        self.assertNotIn(("S1-003", "S2-001"), self.positive_pairs)  # singleton

    def test_multimath_s1_both_positive(self):
        """S1 entity matching both S2 and S3 must have both as positives."""
        self.assertIn(("S1-001", "S2-001"), self.positive_pairs)
        self.assertIn(("S1-001", "S3-001"), self.positive_pairs)
        self.assertEqual(len([p for p in self.positive_pairs if p[0] == "S1-001"]), 2)

    def test_no_false_positive_from_parsing(self):
        """Comma-split must not create extra pairs from concatenated IDs."""
        # Simulate reading a row with comma-separated IDs
        raw_ids = "S2-001,S3-001"
        parsed = {m.strip() for m in raw_ids.split(",") if m.strip()}
        self.assertEqual(parsed, {"S2-001", "S3-001"})
        # Concatenated string itself must NOT be a positive pair
        self.assertNotIn(("S1-001", "S2-001,S3-001"), self.positive_pairs)

    def test_singleton_has_no_positive_pairs(self):
        """Singleton source-1 entities must have no positive pairs."""
        singleton_pairs = [p for p in self.positive_pairs if p[0] == "S1-003"]
        self.assertEqual(len(singleton_pairs), 0)


class TestEntityLevelSplit(unittest.TestCase):
    """Test that train/val split does not leak entities across folds."""

    def setUp(self):
        """Create synthetic DataFrame with 100 S1 entities, 10 rows each."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(100):
            s1_id = f"S1-{i:04d}"
            for j in range(10):
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": f"S2-{rng.integers(1000, 9999)}",
                    "label": int(j == 0),  # exactly 1 positive per entity
                })
        self.df = pd.DataFrame(rows)

    def test_no_entity_overlap_between_splits(self):
        """Source-1 entities must not appear in both train and val."""
        df_train, df_val = entity_level_split(self.df, val_frac=0.2, random_seed=42)
        train_ids = set(df_train["source1_entity_id"].unique())
        val_ids = set(df_val["source1_entity_id"].unique())
        overlap = train_ids & val_ids
        self.assertEqual(len(overlap), 0, f"Entity leakage detected: {overlap}")

    def test_all_entities_accounted_for(self):
        """Every Source-1 entity must be in exactly one split."""
        df_train, df_val = entity_level_split(self.df, val_frac=0.2, random_seed=42)
        all_ids = set(self.df["source1_entity_id"].unique())
        train_ids = set(df_train["source1_entity_id"].unique())
        val_ids = set(df_val["source1_entity_id"].unique())
        self.assertEqual(train_ids | val_ids, all_ids)

    def test_val_fraction_approximately_correct(self):
        """Validation should contain approximately 20% of entities."""
        df_train, df_val = entity_level_split(self.df, val_frac=0.2, random_seed=42)
        val_frac_actual = len(df_val["source1_entity_id"].unique()) / 100
        self.assertAlmostEqual(val_frac_actual, 0.20, delta=0.05)

    def test_reproducibility(self):
        """Same seed must produce same split every time."""
        df_train1, df_val1 = entity_level_split(self.df, val_frac=0.2, random_seed=99)
        df_train2, df_val2 = entity_level_split(self.df, val_frac=0.2, random_seed=99)
        pd.testing.assert_frame_equal(
            df_val1.reset_index(drop=True),
            df_val2.reset_index(drop=True)
        )


class TestF05Metric(unittest.TestCase):
    """Test the F0.5 metric implementation."""

    def test_perfect_prediction(self):
        """Perfect P=1 and R=1 should give F0.5=1.0."""
        f = f05_score_single(precision=1.0, recall=1.0)
        self.assertAlmostEqual(f, 1.0, places=6)

    def test_zero_precision(self):
        """P=0, R=1 should give F0.5=0.0."""
        f = f05_score_single(precision=0.0, recall=1.0)
        self.assertAlmostEqual(f, 0.0, places=6)

    def test_zero_recall(self):
        """P=1, R=0 should give F0.5=0.0."""
        f = f05_score_single(precision=1.0, recall=0.0)
        self.assertAlmostEqual(f, 0.0, places=6)

    def test_both_zero(self):
        """P=0, R=0 should give F0.5=0.0 (denominator is 0)."""
        f = f05_score_single(precision=0.0, recall=0.0)
        self.assertAlmostEqual(f, 0.0, places=6)

    def test_known_value(self):
        """
        F0.5 = 1.25 * 0.8 * 0.6 / (0.25 * 0.8 + 0.6)
             = 0.6 / (0.2 + 0.6)
             = 0.6 / 0.8
             = 0.75
        """
        f = f05_score_single(precision=0.8, recall=0.6)
        self.assertAlmostEqual(f, 0.75, places=6)

    def test_high_precision_weighted(self):
        """F0.5 weights precision more: P=0.9, R=0.5 should score higher than P=0.5, R=0.9."""
        f_high_p = f05_score_single(precision=0.9, recall=0.5)
        f_high_r = f05_score_single(precision=0.5, recall=0.9)
        self.assertGreater(f_high_p, f_high_r,
                           "F0.5 should weight precision more heavily than recall")

    def test_formula_matches_sklearn_fbeta(self):
        """Verify formula matches sklearn's fbeta_score when used globally."""
        from sklearn.metrics import fbeta_score
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        sk_f05 = fbeta_score(y_true, y_pred, beta=0.5)
        tp = 2; fp = 0; fn = 1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        manual_f05 = f05_score_single(p, r)
        self.assertAlmostEqual(manual_f05, sk_f05, places=5)


class TestSingletonHandling(unittest.TestCase):
    """Test correct scoring for singleton Source-1 entities."""

    def _make_df_with_singleton(self, predicted_ids: list) -> tuple[pd.DataFrame, np.ndarray, dict]:
        """Helper to create a small evaluation scenario with a singleton."""
        s1_id = "S1-SINGLE"
        gt_map = {s1_id: set()}  # singleton — no true matches
        rows = []
        for cid in ["S2-001", "S2-002", "S2-003"]:
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "label": 0,
            })
        df = pd.DataFrame(rows)
        # Probability array: 1.0 for predicted, 0.0 for others
        probs = np.array([1.0 if row["candidate_entity_id"] in predicted_ids else 0.0
                          for _, row in df.iterrows()])
        return df, probs, gt_map

    def test_singleton_no_prediction_scores_10(self):
        """Singleton with no prediction (correct) should score F0.5=1.0."""
        df, probs, gt_map = self._make_df_with_singleton(predicted_ids=[])
        result = score_f05_entity_level(df, probs, gt_map, threshold=0.5)
        self.assertAlmostEqual(result["f05"], 1.0, places=5,
                               msg="Correctly predicted singleton should score F0.5=1.0")

    def test_singleton_with_prediction_scores_00(self):
        """Singleton with a false-positive prediction should score F0.5=0.0."""
        df, probs, gt_map = self._make_df_with_singleton(predicted_ids=["S2-001"])
        result = score_f05_entity_level(df, probs, gt_map, threshold=0.5)
        self.assertAlmostEqual(result["f05"], 0.0, places=5,
                               msg="False positive on singleton should score F0.5=0.0")


class TestMissingFeatureHandling(unittest.TestCase):
    """Test that missing (NaN) feature values are filled correctly."""

    def test_nan_filled_with_zero(self):
        """Missing numeric features should be filled with 0.0, not propagated."""
        df = pd.DataFrame([
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001",
             "levenshtein_sim": None, "token_jaccard": 0.5, "label": 0},
        ])
        for col in ["levenshtein_sim", "token_jaccard"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        self.assertEqual(df.iloc[0]["levenshtein_sim"], 0.0)
        self.assertFalse(np.isnan(df.iloc[0]["levenshtein_sim"]))
        self.assertEqual(df.iloc[0]["token_jaccard"], 0.5)

    def test_no_nans_in_feature_matrix_after_fill(self):
        """Simulate a partial merge and verify all NaNs are handled."""
        n = 50
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "source1_entity_id": [f"S1-{i}" for i in range(n)],
            "candidate_entity_id": [f"S2-{i}" for i in range(n)],
            "levenshtein_sim": rng.uniform(0, 1, n),
            "token_jaccard": np.where(rng.uniform(0, 1, n) > 0.5, np.nan, rng.uniform(0, 1, n)),
            "label": rng.integers(0, 2, n),
        })
        for col in ["levenshtein_sim", "token_jaccard"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        self.assertFalse(df["levenshtein_sim"].isna().any())
        self.assertFalse(df["token_jaccard"].isna().any())


class TestNoIDLeakage(unittest.TestCase):
    """Test that entity ID columns are never included as model features."""

    def test_id_cols_excluded_from_features(self):
        """source1_entity_id and candidate_entity_id must not be in feature cols."""
        df = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "levenshtein_sim", "token_jaccard", "char_ngram_cos",
            "addr_levenshtein_sim", "from_name", "from_address", "label"
        ])
        feature_cols = get_feature_cols(df)
        self.assertNotIn("source1_entity_id", feature_cols)
        self.assertNotIn("candidate_entity_id", feature_cols)
        self.assertNotIn("label", feature_cols)
        self.assertNotIn("from_name", feature_cols)
        self.assertNotIn("from_address", feature_cols)

    def test_no_id_leakage_check_passes_for_valid_cols(self):
        """check_no_id_leakage should not raise for legitimate feature columns."""
        valid_cols = [
            "levenshtein_sim", "token_jaccard", "char_ngram_cos",
            "token_sort_ratio", "token_set_ratio", "exact_match",
            "addr_levenshtein_sim", "addr_token_jaccard",
        ]
        try:
            check_no_id_leakage(valid_cols)
        except ValueError:
            self.fail("check_no_id_leakage raised ValueError on valid feature columns")

    def test_no_id_leakage_check_raises_for_id_cols(self):
        """check_no_id_leakage must raise ValueError if an ID column slips through."""
        bad_cols = ["levenshtein_sim", "source1_entity_id"]
        with self.assertRaises(ValueError):
            check_no_id_leakage(bad_cols)


class TestGroundTruthLoader(unittest.TestCase):
    """Test the ground truth loading logic against actual file."""

    def test_gt_file_exists(self):
        """Ground truth file must exist."""
        self.assertTrue(
            os.path.exists("dataset/train/train_ground_truth.tsv"),
            "Ground truth file not found at dataset/train/train_ground_truth.tsv"
        )

    def test_gt_loads_correctly(self):
        """Ground truth must load with expected count of entities and pairs."""
        gt_map = load_ground_truth("dataset/train/train_ground_truth.tsv")
        total_s1 = len(gt_map)
        total_pairs = sum(len(v) for v in gt_map.values())
        singletons = sum(1 for v in gt_map.values() if len(v) == 0)

        self.assertEqual(total_s1, 15000, f"Expected 15000 S1 entities, got {total_s1}")
        self.assertEqual(total_pairs, 15769, f"Expected 15769 true pairs, got {total_pairs}")
        self.assertGreater(singletons, 0, "Expected some singleton S1 entities")

    def test_gt_no_concatenated_id_strings(self):
        """Parsed IDs must start with S2- or S3-, never contain commas."""
        gt_map = load_ground_truth("dataset/train/train_ground_truth.tsv")
        for s1_id, match_set in gt_map.items():
            for m_id in match_set:
                self.assertFalse("," in m_id,
                                 f"Concatenated ID found: '{m_id}' for {s1_id}")
                self.assertTrue(
                    m_id.startswith("S2-") or m_id.startswith("S3-"),
                    f"Unexpected match ID format: '{m_id}' for {s1_id}"
                )


def run_tests():
    """Run the full test suite and return the result."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # Add all test classes
    for cls in [
        TestCandidateUnion,
        TestLabelAssignment,
        TestEntityLevelSplit,
        TestF05Metric,
        TestSingletonHandling,
        TestMissingFeatureHandling,
        TestNoIDLeakage,
        TestGroundTruthLoader,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
