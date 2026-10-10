"""
Unit tests for the shared evaluation harness (scripts/evaluation.py).

These exist because evaluation.py's metric functions are what every lane's
results in the final report are computed with — a bug here would silently
distort every number in the submission. Each test below is checked against
a hand-computed expected value on a small toy example, not against the
function's own output, so a regression would actually be caught.

How to run (from the project root):
    pip install pytest
    python3 -m pytest tests/ -v
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluation import (
    accuracy,
    macro_f1,
    confusion_matrix_df,
    recall_at_k,
    mean_reciprocal_rank,
    fever_score,
    bootstrap_ci,
    paired_bootstrap_test,
)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def test_recall_at_k_all_hit():
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    gold = ["a", "y"]
    assert recall_at_k(ranked, gold, k=3) == 1.0


def test_recall_at_k_partial_hit():
    # query 1: gold "a" is within top-1 -> hit; query 2: gold "z" is rank 3, k=1 -> miss
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    gold = ["a", "z"]
    assert recall_at_k(ranked, gold, k=1) == 0.5


def test_recall_at_k_no_hit():
    ranked = [["b", "c", "d"]]
    gold = ["a"]
    assert recall_at_k(ranked, gold, k=3) == 0.0


def test_mrr_hand_computed():
    # query 1: gold at rank 1 -> RR=1.0 ; query 2: gold at rank 2 -> RR=0.5 ; query 3: not retrieved -> RR=0
    ranked = [["a", "b"], ["x", "a"], ["m", "n"]]
    gold = ["a", "a", "a"]
    expected = (1.0 + 0.5 + 0.0) / 3
    assert math.isclose(mean_reciprocal_rank(ranked, gold), expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def test_accuracy_hand_computed():
    y_true = ["SUPPORTS", "REFUTES", "SUPPORTS", "REFUTES"]
    y_pred = ["SUPPORTS", "SUPPORTS", "SUPPORTS", "REFUTES"]
    assert accuracy(y_true, y_pred) == 0.75


def test_macro_f1_perfect_prediction():
    y_true = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    y_pred = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    assert macro_f1(y_true, y_pred) == 1.0


def test_macro_f1_hand_computed_two_class():
    # 2 SUPPORTS correct, 1 SUPPORTS predicted as REFUTES, 1 REFUTES correct
    y_true = ["SUPPORTS", "SUPPORTS", "SUPPORTS", "REFUTES"]
    y_pred = ["SUPPORTS", "SUPPORTS", "REFUTES", "REFUTES"]
    # SUPPORTS: TP=2, FP=0, FN=1 -> precision=1.0, recall=2/3 -> F1 = 2*1*0.667/(1+0.667) = 0.8
    # REFUTES: TP=1, FP=1, FN=0 -> precision=0.5, recall=1.0 -> F1 = 2*0.5*1/(1.5) = 0.6667
    expected = (0.8 + 2 / 3) / 2
    assert math.isclose(macro_f1(y_true, y_pred, labels=["SUPPORTS", "REFUTES"]), expected, rel_tol=1e-3)


def test_confusion_matrix_shape_and_totals():
    y_true = ["SUPPORTS", "REFUTES", "SUPPORTS"]
    y_pred = ["SUPPORTS", "SUPPORTS", "SUPPORTS"]
    cm = confusion_matrix_df(y_true, y_pred, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
    assert cm.loc["SUPPORTS", "SUPPORTS"] == 2
    assert cm.loc["REFUTES", "SUPPORTS"] == 1
    assert cm.values.sum() == 3


# ---------------------------------------------------------------------------
# End-to-end FEVER score
# ---------------------------------------------------------------------------

def test_fever_score_requires_both_label_and_evidence():
    y_true = ["SUPPORTS", "SUPPORTS"]
    y_pred = ["SUPPORTS", "SUPPORTS"]  # label correct in both cases
    retrieved = [["ev_1", "ev_2"], ["ev_9"]]
    gold = ["ev_1", "ev_2"]  # case 1: gold IS retrieved; case 2: gold NOT retrieved
    score = fever_score(y_true, y_pred, retrieved, gold, k=2)
    assert score == 0.5


def test_fever_score_nei_only_needs_label():
    y_true = ["NOT ENOUGH INFO"]
    y_pred = ["NOT ENOUGH INFO"]
    retrieved = [["ev_1"]]
    gold = [float("nan")]  # NEI rows have no gold evidence
    assert fever_score(y_true, y_pred, retrieved, gold, k=1) == 1.0


# ---------------------------------------------------------------------------
# Statistical rigor helpers
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_point_estimate():
    values = [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    result = bootstrap_ci(values, n_resamples=500, seed=0)
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
    assert math.isclose(result["point_estimate"], sum(values) / len(values))


def test_bootstrap_ci_narrows_with_more_data():
    # a larger, less noisy sample should give a narrower CI than a small one
    small = [1, 0, 1, 0, 1]
    large = [1, 0, 1, 0, 1] * 40
    small_ci = bootstrap_ci(small, n_resamples=1000, seed=0)
    large_ci = bootstrap_ci(large, n_resamples=1000, seed=0)
    small_width = small_ci["ci_upper"] - small_ci["ci_lower"]
    large_width = large_ci["ci_upper"] - large_ci["ci_lower"]
    assert large_width < small_width


def test_paired_bootstrap_identical_systems_not_significant():
    # system A and B are IDENTICAL -> observed diff is 0 -> must not be significant
    values = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    result = paired_bootstrap_test(values, values, n_resamples=500, seed=0)
    assert result["observed_diff"] == 0.0
    assert result["significant_at_0.05"] is False


def test_paired_bootstrap_detects_large_consistent_difference():
    # system A always wins, system B always loses -> should be significant
    a = [1] * 50
    b = [0] * 50
    result = paired_bootstrap_test(a, b, n_resamples=500, seed=0)
    assert result["observed_diff"] == 1.0
    assert result["significant_at_0.05"] is True
