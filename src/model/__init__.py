"""
Person 3 — Candidate Merge, Model Training & Local Evaluation
Amazon ML Challenge 2026: Business Entity Resolution

Public API:
    merge_features.merge_candidates()  -> merged candidate TSV
    train.run_training_pipeline()      -> trained model + artifacts
    evaluate.score_f05_entity_level()  -> entity-level F0.5
"""

from src.model.merge_features import merge_candidates
from src.model.evaluate import score_f05_entity_level, evaluate_combined_recall
from src.model.train import run_training_pipeline

__all__ = [
    "merge_candidates",
    "score_f05_entity_level",
    "evaluate_combined_recall",
    "run_training_pipeline",
]
