"""
tests/test_fraud/test_evaluation.py
=====================================
Unit tests for the FraudEvaluator evaluation suite.
"""

import numpy as np
import pandas as pd
import pytest

from fincrime_ml.fraud.evaluation import (
    CostMatrix,
    FraudEvaluator,
    ThresholdMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def binary_data():
    """200-sample binary classification data with known imbalance."""
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_true[:20] = 1  # 10% fraud rate
    RNG.shuffle(y_true)
    # Scores correlated with labels but noisy
    y_scores = np.clip(y_true * 0.6 + RNG.random(n) * 0.5, 0.0, 1.0)
    return y_true, y_scores


@pytest.fixture(scope="module")
def evaluator():
    return FraudEvaluator()


@pytest.fixture(scope="module")
def perfect_scores():
    """y_scores that perfectly separate the classes."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05, 0.35])
    return y_true, y_scores


# ---------------------------------------------------------------------------
# ThresholdMetrics dataclass tests
# ---------------------------------------------------------------------------


def test_threshold_metrics_fields():
    m = ThresholdMetrics(
        threshold=0.5,
        precision=0.8,
        recall=0.6,
        f1=0.686,
        n_alerts=10,
        n_true_positives=8,
        n_false_positives=2,
        n_false_negatives=4,
        alert_rate=0.05,
    )
    assert m.threshold == 0.5
    assert m.precision == 0.8
    assert m.recall == 0.6
    assert m.n_alerts == 10


def test_cost_matrix_fields():
    cm = CostMatrix(
        threshold=0.5,
        fp_cost_per_case=15.0,
        fn_cost_per_case=250.0,
        total_fp_cost=30.0,
        total_fn_cost=500.0,
        total_cost=530.0,
        n_false_positives=2,
        n_false_negatives=2,
    )
    assert cm.total_cost == 530.0
    assert cm.n_false_positives == 2


# ---------------------------------------------------------------------------
# FraudEvaluator instantiation tests
# ---------------------------------------------------------------------------


def test_default_costs():
    ev = FraudEvaluator()
    assert ev.fp_cost == 15.0
    assert ev.fn_cost == 250.0


def test_custom_costs():
    ev = FraudEvaluator(fp_cost=20.0, fn_cost=300.0)
    assert ev.fp_cost == 20.0
    assert ev.fn_cost == 300.0


# ---------------------------------------------------------------------------
# _validate() tests
# ---------------------------------------------------------------------------


def test_validate_raises_length_mismatch():
    ev = FraudEvaluator()
    with pytest.raises(ValueError, match="same length"):
        ev._validate(np.array([0, 1, 0]), np.array([0.1, 0.9]))


def test_validate_raises_non_binary():
    ev = FraudEvaluator()
    with pytest.raises(ValueError, match="binary"):
        ev._validate(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_validate_accepts_series():
    ev = FraudEvaluator()
    y_t, y_s = ev._validate(pd.Series([0, 1, 0]), pd.Series([0.1, 0.9, 0.2]))
    assert isinstance(y_t, np.ndarray)
    assert isinstance(y_s, np.ndarray)


def test_validate_returns_arrays(binary_data):
    ev = FraudEvaluator()
    y_true, y_scores = binary_data
    y_t, y_s = ev._validate(y_true, y_scores)
    assert y_t.dtype == int
    assert y_s.dtype == float


# ---------------------------------------------------------------------------
# threshold_analysis() tests
# ---------------------------------------------------------------------------


def test_threshold_analysis_returns_dataframe(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    assert isinstance(result, pd.DataFrame)


def test_threshold_analysis_column_names(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    expected = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "n_alerts",
        "n_true_positives",
        "n_false_positives",
        "n_false_negatives",
        "alert_rate",
    ]
    for col in expected:
        assert col in result.columns


def test_threshold_analysis_row_count_default(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    assert len(result) == 100


def test_threshold_analysis_custom_thresholds(evaluator, binary_data):
    y_true, y_scores = binary_data
    custom = np.array([0.3, 0.5, 0.7])
    result = evaluator.threshold_analysis(y_true, y_scores, thresholds=custom)
    assert len(result) == 3


def test_threshold_analysis_sorted_ascending(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    assert result["threshold"].is_monotonic_increasing


def test_threshold_analysis_precision_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    assert (result["precision"] >= 0).all()
    assert (result["precision"] <= 1).all()


def test_threshold_analysis_recall_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    assert (result["recall"] >= 0).all()
    assert (result["recall"] <= 1).all()


def test_threshold_analysis_f1_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    assert (result["f1"] >= 0).all()
    assert (result["f1"] <= 1).all()


def test_threshold_analysis_alert_rate_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    assert (result["alert_rate"] >= 0).all()
    assert (result["alert_rate"] <= 1).all()


def test_threshold_analysis_n_alerts_non_negative(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    assert (result["n_alerts"] >= 0).all()


def test_threshold_analysis_accepts_series(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(pd.Series(y_true), pd.Series(y_scores))
    assert len(result) == 100


def test_threshold_analysis_high_threshold_low_alerts(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.threshold_analysis(y_true, y_scores)
    high_t = result[result["threshold"] > result["threshold"].quantile(0.9)]
    low_t = result[result["threshold"] < result["threshold"].quantile(0.1)]
    assert high_t["n_alerts"].mean() <= low_t["n_alerts"].mean()


# ---------------------------------------------------------------------------
# cost_matrix() tests
# ---------------------------------------------------------------------------


def test_cost_matrix_returns_cost_matrix(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.cost_matrix(y_true, y_scores, threshold=0.5)
    assert isinstance(result, CostMatrix)


def test_cost_matrix_threshold_stored(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.cost_matrix(y_true, y_scores, threshold=0.5)
    assert result.threshold == 0.5


def test_cost_matrix_total_equals_sum(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.cost_matrix(y_true, y_scores, threshold=0.5)
    assert abs(result.total_cost - (result.total_fp_cost + result.total_fn_cost)) < 0.01


def test_cost_matrix_default_costs_used(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.cost_matrix(y_true, y_scores, threshold=0.5)
    assert result.fp_cost_per_case == 15.0
    assert result.fn_cost_per_case == 250.0


def test_cost_matrix_custom_costs(binary_data):
    y_true, y_scores = binary_data
    ev = FraudEvaluator(fp_cost=10.0, fn_cost=100.0)
    result = ev.cost_matrix(y_true, y_scores, threshold=0.5)
    assert result.fp_cost_per_case == 10.0
    assert result.fn_cost_per_case == 100.0


def test_cost_matrix_override_costs(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.cost_matrix(y_true, y_scores, threshold=0.5, fp_cost=5.0, fn_cost=50.0)
    assert result.fp_cost_per_case == 5.0
    assert result.fn_cost_per_case == 50.0


def test_cost_matrix_fp_cost_correct(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.cost_matrix(y_true, y_scores, threshold=0.5)
    expected = result.n_false_positives * result.fp_cost_per_case
    assert abs(result.total_fp_cost - expected) < 0.01


def test_cost_matrix_fn_cost_correct(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.cost_matrix(y_true, y_scores, threshold=0.5)
    expected = result.n_false_negatives * result.fn_cost_per_case
    assert abs(result.total_fn_cost - expected) < 0.01


def test_cost_matrix_non_negative(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.cost_matrix(y_true, y_scores, threshold=0.5)
    assert result.total_cost >= 0
    assert result.total_fp_cost >= 0
    assert result.total_fn_cost >= 0


# ---------------------------------------------------------------------------
# optimal_threshold() tests
# ---------------------------------------------------------------------------


def test_optimal_threshold_f1_returns_float(evaluator, binary_data):
    y_true, y_scores = binary_data
    t = evaluator.optimal_threshold(y_true, y_scores, strategy="f1")
    assert isinstance(t, float)


def test_optimal_threshold_cost_returns_float(evaluator, binary_data):
    y_true, y_scores = binary_data
    t = evaluator.optimal_threshold(y_true, y_scores, strategy="cost")
    assert isinstance(t, float)


def test_optimal_threshold_recall_at_precision_returns_float(evaluator, binary_data):
    y_true, y_scores = binary_data
    t = evaluator.optimal_threshold(y_true, y_scores, strategy="recall_at_precision")
    assert isinstance(t, float)


def test_optimal_threshold_f1_in_score_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    t = evaluator.optimal_threshold(y_true, y_scores, strategy="f1")
    assert y_scores.min() <= t <= y_scores.max()


def test_optimal_threshold_invalid_strategy(evaluator, binary_data):
    y_true, y_scores = binary_data
    with pytest.raises(ValueError, match="strategy must be one of"):
        evaluator.optimal_threshold(y_true, y_scores, strategy="invalid")


def test_optimal_threshold_f1_maximises_f1(evaluator, binary_data):
    y_true, y_scores = binary_data
    t_opt = evaluator.optimal_threshold(y_true, y_scores, strategy="f1")
    analysis = evaluator.threshold_analysis(y_true, y_scores)
    best_f1_row = analysis.loc[analysis["f1"].idxmax()]
    # The optimal threshold should produce F1 at least as good as any other in the sweep
    m_opt = evaluator._metrics_at_threshold(y_true, y_scores, t_opt)
    assert m_opt.f1 >= best_f1_row["f1"] - 0.01


def test_optimal_threshold_cost_with_custom_costs(evaluator, binary_data):
    y_true, y_scores = binary_data
    t = evaluator.optimal_threshold(y_true, y_scores, strategy="cost", fp_cost=5.0, fn_cost=500.0)
    assert isinstance(t, float)


def test_optimal_threshold_recall_at_precision_respects_constraint(evaluator, binary_data):
    y_true, y_scores = binary_data
    t = evaluator.optimal_threshold(y_true, y_scores, strategy="recall_at_precision")
    m = evaluator._metrics_at_threshold(y_true, y_scores, t)
    # precision must be >= 0.5 or no candidate was found (fallback to last threshold)
    assert m.precision >= 0.5 or t == float(np.linspace(y_scores.min(), y_scores.max(), 100)[-1])


# ---------------------------------------------------------------------------
# compare_models() tests
# ---------------------------------------------------------------------------


def test_compare_models_returns_dataframe(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.compare_models(y_true, {"model_a": y_scores})
    assert isinstance(result, pd.DataFrame)


def test_compare_models_columns_no_threshold(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.compare_models(y_true, {"model_a": y_scores})
    assert "model" in result.columns
    assert "auc_pr" in result.columns
    assert "roc_auc" in result.columns


def test_compare_models_columns_with_threshold(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.compare_models(y_true, {"model_a": y_scores}, threshold=0.5)
    for col in ("precision", "recall", "f1", "n_alerts"):
        assert col in result.columns


def test_compare_models_multiple_models(evaluator, binary_data):
    y_true, y_scores = binary_data
    noisy = np.clip(y_scores + RNG.normal(0, 0.1, len(y_scores)), 0.0, 1.0)
    result = evaluator.compare_models(y_true, {"model_a": y_scores, "model_b": noisy})
    assert len(result) == 2


def test_compare_models_sorted_by_auc_pr(evaluator, binary_data):
    y_true, y_scores = binary_data
    noisy = np.clip(y_scores + RNG.normal(0, 0.15, len(y_scores)), 0.0, 1.0)
    result = evaluator.compare_models(y_true, {"model_a": y_scores, "model_b": noisy})
    assert result["auc_pr"].is_monotonic_decreasing


def test_compare_models_auc_pr_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.compare_models(y_true, {"model_a": y_scores})
    assert (result["auc_pr"] >= 0).all()
    assert (result["auc_pr"] <= 1).all()


def test_compare_models_roc_auc_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.compare_models(y_true, {"model_a": y_scores})
    assert (result["roc_auc"] >= 0).all()
    assert (result["roc_auc"] <= 1).all()


def test_compare_models_model_name_in_result(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.compare_models(y_true, {"xgb_v1": y_scores})
    assert "xgb_v1" in result["model"].values


# ---------------------------------------------------------------------------
# pr_curve() tests
# ---------------------------------------------------------------------------


def test_pr_curve_returns_dataframe(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.pr_curve(y_true, y_scores)
    assert isinstance(result, pd.DataFrame)


def test_pr_curve_columns(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.pr_curve(y_true, y_scores)
    for col in ("threshold", "precision", "recall"):
        assert col in result.columns


def test_pr_curve_precision_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.pr_curve(y_true, y_scores)
    assert (result["precision"].dropna() >= 0).all()
    assert (result["precision"].dropna() <= 1).all()


def test_pr_curve_recall_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.pr_curve(y_true, y_scores)
    assert (result["recall"].dropna() >= 0).all()
    assert (result["recall"].dropna() <= 1).all()


def test_pr_curve_last_threshold_nan(evaluator, binary_data):
    """sklearn appends NaN as last threshold value."""
    y_true, y_scores = binary_data
    result = evaluator.pr_curve(y_true, y_scores)
    assert pd.isna(result["threshold"].iloc[-1])


# ---------------------------------------------------------------------------
# roc_curve() tests
# ---------------------------------------------------------------------------


def test_roc_curve_returns_dataframe(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.roc_curve(y_true, y_scores)
    assert isinstance(result, pd.DataFrame)


def test_roc_curve_columns(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.roc_curve(y_true, y_scores)
    for col in ("threshold", "fpr", "tpr"):
        assert col in result.columns


def test_roc_curve_fpr_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.roc_curve(y_true, y_scores)
    assert (result["fpr"] >= 0).all()
    assert (result["fpr"] <= 1).all()


def test_roc_curve_tpr_in_range(evaluator, binary_data):
    y_true, y_scores = binary_data
    result = evaluator.roc_curve(y_true, y_scores)
    assert (result["tpr"] >= 0).all()
    assert (result["tpr"] <= 1).all()


# ---------------------------------------------------------------------------
# _metrics_at_threshold() tests
# ---------------------------------------------------------------------------


def test_metrics_at_threshold_returns_dataclass(evaluator, binary_data):
    y_true, y_scores = binary_data
    m = evaluator._metrics_at_threshold(y_true, y_scores, 0.5)
    assert isinstance(m, ThresholdMetrics)


def test_metrics_perfect_recall_at_zero_threshold(evaluator, binary_data):
    y_true, y_scores = binary_data
    m = evaluator._metrics_at_threshold(y_true, y_scores, 0.0)
    assert m.recall == 1.0


def test_metrics_zero_alerts_at_threshold_above_max(evaluator, binary_data):
    y_true, y_scores = binary_data
    m = evaluator._metrics_at_threshold(y_true, y_scores, 2.0)
    assert m.n_alerts == 0
    assert m.precision == 0.0


def test_metrics_tp_plus_fn_equals_total_fraud(evaluator, binary_data):
    y_true, y_scores = binary_data
    m = evaluator._metrics_at_threshold(y_true, y_scores, 0.5)
    assert m.n_true_positives + m.n_false_negatives == int(y_true.sum())


def test_metrics_alert_rate_matches_n_alerts(evaluator, binary_data):
    y_true, y_scores = binary_data
    m = evaluator._metrics_at_threshold(y_true, y_scores, 0.5)
    expected_rate = m.n_alerts / len(y_true)
    assert abs(m.alert_rate - expected_rate) < 1e-6


def test_metrics_f1_is_harmonic_mean(evaluator, binary_data):
    y_true, y_scores = binary_data
    m = evaluator._metrics_at_threshold(y_true, y_scores, 0.5)
    if m.precision + m.recall > 0:
        expected_f1 = 2 * m.precision * m.recall / (m.precision + m.recall)
        assert abs(m.f1 - expected_f1) < 1e-6


def test_metrics_perfect_scores(evaluator, perfect_scores):
    y_true, y_scores = perfect_scores
    m = evaluator._metrics_at_threshold(y_true, y_scores, 0.65)
    assert m.n_false_positives == 0
    assert m.n_false_negatives == 0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
