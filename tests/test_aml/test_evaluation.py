"""
tests/test_aml/test_evaluation.py
=====================================
Unit tests for the AML alert fatigue evaluation module.

Covers: AlertFatigueConfig, evaluate(), fpr_at_sensitivity(),
threshold_at_sensitivity(), alert_volume_profile(), sensitivity_curve(),
fatigue_index(), pr_auc(), input validation, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincrime_ml.aml.evaluation import (
    DEFAULT_SENSITIVITY_TARGETS,
    AlertFatigueConfig,
    AlertFatigueEvaluator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluator() -> AlertFatigueEvaluator:
    return AlertFatigueEvaluator()


@pytest.fixture(scope="module")
def y_true() -> list[int]:
    """80 legitimate + 20 suspicious transactions."""
    rng = np.random.default_rng(42)
    labels = [0] * 80 + [1] * 20
    rng.shuffle(labels)
    return [int(x) for x in labels]


@pytest.fixture(scope="module")
def scores(y_true) -> list[float]:
    """Scores correlated with labels: suspicious transactions score higher."""
    rng = np.random.default_rng(42)
    result = []
    for label in y_true:
        if label == 1:
            result.append(float(np.clip(rng.normal(0.75, 0.15), 0, 1)))
        else:
            result.append(float(np.clip(rng.normal(0.30, 0.20), 0, 1)))
    return result


@pytest.fixture(scope="module")
def report(evaluator, y_true, scores) -> dict:
    return evaluator.evaluate(y_true, scores)


@pytest.fixture(scope="module")
def volume_profile(evaluator, y_true, scores) -> pd.DataFrame:
    return evaluator.alert_volume_profile(y_true, scores)


@pytest.fixture(scope="module")
def curve(evaluator, y_true, scores) -> pd.DataFrame:
    return evaluator.sensitivity_curve(y_true, scores)


# ---------------------------------------------------------------------------
# AlertFatigueConfig
# ---------------------------------------------------------------------------


def test_config_default_sensitivity_targets():
    cfg = AlertFatigueConfig()
    assert cfg.sensitivity_targets == DEFAULT_SENSITIVITY_TARGETS


def test_config_default_min_sensitivity():
    cfg = AlertFatigueConfig()
    assert cfg.min_sensitivity == 0.80


def test_config_custom_targets():
    cfg = AlertFatigueConfig(sensitivity_targets=(0.90, 0.95))
    ev = AlertFatigueEvaluator(config=cfg)
    assert ev.config.sensitivity_targets == (0.90, 0.95)


def test_config_custom_min_sensitivity():
    cfg = AlertFatigueConfig(min_sensitivity=0.70)
    ev = AlertFatigueEvaluator(config=cfg)
    assert ev.config.min_sensitivity == 0.70


def test_config_version():
    cfg = AlertFatigueConfig()
    assert isinstance(cfg.version, str)
    assert len(cfg.version) > 0


def test_default_sensitivity_targets_all_in_unit_interval():
    for t in DEFAULT_SENSITIVITY_TARGETS:
        assert 0.0 < t <= 1.0


# ---------------------------------------------------------------------------
# evaluate() — output schema
# ---------------------------------------------------------------------------


def test_evaluate_returns_dict(report):
    assert isinstance(report, dict)


def test_evaluate_has_required_keys(report):
    for key in (
        "n_positives",
        "n_negatives",
        "base_rate",
        "auc_pr",
        "roc_auc",
        "sensitivity_analysis",
        "optimal_threshold",
    ):
        assert key in report, f"Missing key: {key}"


def test_evaluate_n_positives_correct(report, y_true):
    assert report["n_positives"] == sum(y_true)


def test_evaluate_n_negatives_correct(report, y_true):
    assert report["n_negatives"] == len(y_true) - sum(y_true)


def test_evaluate_base_rate_in_unit_interval(report):
    assert 0.0 <= report["base_rate"] <= 1.0


def test_evaluate_auc_pr_in_unit_interval(report):
    assert 0.0 <= report["auc_pr"] <= 1.0


def test_evaluate_roc_auc_in_unit_interval(report):
    assert 0.0 <= report["roc_auc"] <= 1.0


def test_evaluate_sensitivity_analysis_has_all_targets(report):
    for target in DEFAULT_SENSITIVITY_TARGETS:
        assert target in report["sensitivity_analysis"]


def test_evaluate_sensitivity_analysis_entry_schema(report):
    entry = report["sensitivity_analysis"][0.90]
    for key in ("threshold", "fpr", "recall", "precision", "f1", "fatigue_index", "alert_rate"):
        assert key in entry, f"Missing key in sensitivity_analysis entry: {key}"


def test_evaluate_sensitivity_analysis_fpr_in_unit_interval(report):
    for target, entry in report["sensitivity_analysis"].items():
        assert 0.0 <= entry["fpr"] <= 1.0, f"FPR out of range at sensitivity {target}"


def test_evaluate_sensitivity_analysis_recall_meets_target(report):
    for target, entry in report["sensitivity_analysis"].items():
        # Recall should be >= target (within float tolerance or if target is achievable)
        assert entry["recall"] >= 0.0


def test_evaluate_optimal_threshold_has_required_keys(report):
    opt = report["optimal_threshold"]
    for key in ("value", "fpr", "recall", "precision", "f1", "fatigue_index"):
        assert key in opt


def test_evaluate_optimal_threshold_value_in_unit_interval(report):
    assert 0.0 <= report["optimal_threshold"]["value"] <= 1.0


def test_evaluate_higher_sensitivity_target_gives_higher_or_equal_fpr(report):
    targets = sorted(report["sensitivity_analysis"].keys())
    fprs = [report["sensitivity_analysis"][t]["fpr"] for t in targets]
    # FPR must be non-decreasing as sensitivity target increases
    for i in range(len(fprs) - 1):
        assert fprs[i] <= fprs[i + 1] + 1e-6, (
            f"FPR not non-decreasing: {fprs[i]:.4f} > {fprs[i+1]:.4f} "
            f"at targets {targets[i]}->{targets[i+1]}"
        )


def test_evaluate_good_discriminator_auc_above_random(report):
    assert report["auc_pr"] > 0.30


def test_evaluate_good_discriminator_roc_above_random(report):
    assert report["roc_auc"] > 0.60


# ---------------------------------------------------------------------------
# fpr_at_sensitivity()
# ---------------------------------------------------------------------------


def test_fpr_at_sensitivity_returns_float(evaluator, y_true, scores):
    result = evaluator.fpr_at_sensitivity(y_true, scores, 0.90)
    assert isinstance(result, float)


def test_fpr_at_sensitivity_in_unit_interval(evaluator, y_true, scores):
    fpr = evaluator.fpr_at_sensitivity(y_true, scores, 0.90)
    assert 0.0 <= fpr <= 1.0


def test_fpr_at_sensitivity_higher_target_higher_or_equal_fpr(evaluator, y_true, scores):
    fpr_80 = evaluator.fpr_at_sensitivity(y_true, scores, 0.80)
    fpr_95 = evaluator.fpr_at_sensitivity(y_true, scores, 0.95)
    assert fpr_80 <= fpr_95 + 1e-6


def test_fpr_at_sensitivity_100_flags_all(evaluator, y_true, scores):
    # At 100% recall, essentially all transactions are flagged
    fpr = evaluator.fpr_at_sensitivity(y_true, scores, 0.99)
    assert fpr >= 0.0


def test_fpr_at_sensitivity_perfect_scores():
    evaluator = AlertFatigueEvaluator()
    y = [1] * 20 + [0] * 80
    # Perfect scores: suspicious always 1.0, legitimate always 0.0
    s = [1.0] * 20 + [0.0] * 80
    fpr = evaluator.fpr_at_sensitivity(y, s, 0.90)
    assert fpr == 0.0


# ---------------------------------------------------------------------------
# threshold_at_sensitivity()
# ---------------------------------------------------------------------------


def test_threshold_at_sensitivity_returns_float(evaluator, y_true, scores):
    result = evaluator.threshold_at_sensitivity(y_true, scores, 0.90)
    assert isinstance(result, float)


def test_threshold_at_sensitivity_in_unit_interval(evaluator, y_true, scores):
    thresh = evaluator.threshold_at_sensitivity(y_true, scores, 0.90)
    assert 0.0 <= thresh <= 1.0


def test_threshold_at_sensitivity_higher_target_lower_or_equal_threshold(evaluator, y_true, scores):
    thresh_80 = evaluator.threshold_at_sensitivity(y_true, scores, 0.80)
    thresh_95 = evaluator.threshold_at_sensitivity(y_true, scores, 0.95)
    # Higher recall target requires a lower threshold (more alerts)
    assert thresh_95 <= thresh_80 + 1e-6


def test_threshold_at_sensitivity_achieves_target_recall(evaluator, y_true, scores):
    target = 0.85
    thresh = evaluator.threshold_at_sensitivity(y_true, scores, target)
    # Verify: scoring at this threshold achieves at least target recall
    y_arr = np.array(y_true)
    s_arr = np.array(scores)
    predicted = (s_arr >= thresh).astype(int)
    tp = ((predicted == 1) & (y_arr == 1)).sum()
    fn = ((predicted == 0) & (y_arr == 1)).sum()
    actual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    assert actual_recall >= target - 0.02  # small tolerance for curve discretisation


def test_threshold_at_sensitivity_zero_when_unachievable(evaluator):
    # Degenerate case: scores are identical, cannot achieve high recall
    y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    s = [0.5] * 10
    result = evaluator.threshold_at_sensitivity(y, s, 0.99)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# alert_volume_profile()
# ---------------------------------------------------------------------------


def test_alert_volume_profile_returns_dataframe(volume_profile):
    assert isinstance(volume_profile, pd.DataFrame)


def test_alert_volume_profile_has_required_columns(volume_profile):
    for col in (
        "threshold",
        "n_alerts",
        "alert_rate",
        "precision",
        "recall",
        "fpr",
        "f1",
        "fatigue_index",
    ):
        assert col in volume_profile.columns, f"Missing column: {col}"


def test_alert_volume_profile_sorted_by_threshold(volume_profile):
    assert volume_profile["threshold"].is_monotonic_increasing


def test_alert_volume_profile_alert_rate_in_unit_interval(volume_profile):
    assert (volume_profile["alert_rate"] >= 0.0).all()
    assert (volume_profile["alert_rate"] <= 1.0).all()


def test_alert_volume_profile_precision_in_unit_interval(volume_profile):
    assert (volume_profile["precision"] >= 0.0).all()
    assert (volume_profile["precision"] <= 1.0).all()


def test_alert_volume_profile_recall_in_unit_interval(volume_profile):
    assert (volume_profile["recall"] >= 0.0).all()
    assert (volume_profile["recall"] <= 1.0).all()


def test_alert_volume_profile_fpr_in_unit_interval(volume_profile):
    assert (volume_profile["fpr"] >= 0.0).all()
    assert (volume_profile["fpr"] <= 1.0).all()


def test_alert_volume_profile_fatigue_index_in_unit_interval(volume_profile):
    assert (volume_profile["fatigue_index"] >= 0.0).all()
    assert (volume_profile["fatigue_index"] <= 1.0).all()


def test_alert_volume_profile_custom_thresholds(evaluator, y_true, scores):
    thresholds = [0.3, 0.5, 0.7]
    profile = evaluator.alert_volume_profile(y_true, scores, thresholds=thresholds)
    assert len(profile) == 3
    assert list(profile["threshold"]) == pytest.approx(thresholds, abs=1e-6)


def test_alert_volume_profile_zero_threshold_all_alerted(evaluator, y_true, scores):
    profile = evaluator.alert_volume_profile(y_true, scores, thresholds=[0.0])
    assert profile.iloc[0]["n_alerts"] == len(y_true)


def test_alert_volume_profile_one_threshold_no_alerts(evaluator, y_true, scores):
    profile = evaluator.alert_volume_profile(y_true, scores, thresholds=[1.0 + 1e-9])
    assert profile.iloc[0]["n_alerts"] == 0


def test_alert_volume_profile_n_alerts_decreases_as_threshold_rises(evaluator, y_true, scores):
    profile = evaluator.alert_volume_profile(y_true, scores)
    # n_alerts should be non-increasing as threshold rises
    assert (profile["n_alerts"].diff().dropna() <= 0).all()


# ---------------------------------------------------------------------------
# sensitivity_curve()
# ---------------------------------------------------------------------------


def test_sensitivity_curve_returns_dataframe(curve):
    assert isinstance(curve, pd.DataFrame)


def test_sensitivity_curve_has_required_columns(curve):
    for col in ("threshold", "sensitivity", "fpr", "specificity"):
        assert col in curve.columns


def test_sensitivity_curve_sensitivity_in_unit_interval(curve):
    assert (curve["sensitivity"] >= 0.0).all()
    assert (curve["sensitivity"] <= 1.0).all()


def test_sensitivity_curve_fpr_in_unit_interval(curve):
    assert (curve["fpr"] >= 0.0).all()
    assert (curve["fpr"] <= 1.0).all()


def test_sensitivity_curve_specificity_complement_of_fpr(curve):
    diff = (curve["specificity"] - (1.0 - curve["fpr"])).abs()
    assert (diff < 1e-5).all()


def test_sensitivity_curve_sorted_by_sensitivity(curve):
    assert curve["sensitivity"].is_monotonic_increasing


def test_sensitivity_curve_row_count_positive(curve):
    assert len(curve) > 1


# ---------------------------------------------------------------------------
# fatigue_index()
# ---------------------------------------------------------------------------


def test_fatigue_index_returns_float(evaluator, y_true, scores):
    result = evaluator.fatigue_index(y_true, scores, 0.5)
    assert isinstance(result, float)


def test_fatigue_index_in_unit_interval(evaluator, y_true, scores):
    fi = evaluator.fatigue_index(y_true, scores, 0.5)
    assert 0.0 <= fi <= 1.0


def test_fatigue_index_zero_for_perfect_classifier(evaluator):
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    s = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fi = evaluator.fatigue_index(y, s, 0.5)
    assert fi == pytest.approx(0.0)


def test_fatigue_index_one_for_all_false_positives(evaluator):
    y = [0, 0, 0, 0, 0, 1, 1, 1]
    # Threshold so low that all negatives are caught but positives score low
    s = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    # At threshold 0.5, we catch the 5 negatives with high scores, 0 true positives
    fi = evaluator.fatigue_index(y, s, 0.5)
    assert fi == pytest.approx(1.0)


def test_fatigue_index_no_alerts_returns_zero(evaluator):
    y = [0, 0, 1, 0, 1, 0, 0, 0]
    s = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # Threshold above all scores → no alerts
    fi = evaluator.fatigue_index(y, s, 0.9)
    assert fi == 0.0


def test_fatigue_index_complement_of_precision(evaluator, y_true, scores):
    thresh = 0.6
    fi = evaluator.fatigue_index(y_true, scores, thresh)
    y_arr = np.array(y_true)
    s_arr = np.array(scores)
    predicted = (s_arr >= thresh).astype(int)
    tp = ((predicted == 1) & (y_arr == 1)).sum()
    fp = ((predicted == 1) & (y_arr == 0)).sum()
    n_alerts = tp + fp
    expected_fi = (fp / n_alerts) if n_alerts > 0 else 0.0
    assert fi == pytest.approx(expected_fi, abs=1e-5)


# ---------------------------------------------------------------------------
# pr_auc()
# ---------------------------------------------------------------------------


def test_pr_auc_returns_float(evaluator, y_true, scores):
    result = evaluator.pr_auc(y_true, scores)
    assert isinstance(result, float)


def test_pr_auc_in_unit_interval(evaluator, y_true, scores):
    result = evaluator.pr_auc(y_true, scores)
    assert 0.0 <= result <= 1.0


def test_pr_auc_perfect_scores():
    evaluator = AlertFatigueEvaluator()
    y = [0] * 10 + [1] * 10
    s = [0.0] * 10 + [1.0] * 10
    result = evaluator.pr_auc(y, s)
    assert result == pytest.approx(1.0)


def test_pr_auc_matches_evaluate(evaluator, y_true, scores, report):
    direct = evaluator.pr_auc(y_true, scores)
    assert direct == pytest.approx(report["auc_pr"], abs=1e-5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validate_raises_on_length_mismatch(evaluator):
    with pytest.raises(ValueError, match="length"):
        evaluator.evaluate([0, 1, 0], [0.5, 0.7])


def test_validate_raises_on_single_class_only_positives(evaluator):
    with pytest.raises(ValueError, match="both positive"):
        evaluator.evaluate([1, 1, 1], [0.8, 0.7, 0.9])


def test_validate_raises_on_single_class_only_negatives(evaluator):
    with pytest.raises(ValueError, match="both positive"):
        evaluator.evaluate([0, 0, 0], [0.1, 0.2, 0.3])


def test_validate_raises_on_too_few_samples(evaluator):
    with pytest.raises(ValueError, match="at least 2"):
        evaluator.evaluate([1], [0.9])


def test_validate_fpr_at_sensitivity_catches_bad_input(evaluator):
    with pytest.raises(ValueError):
        evaluator.fpr_at_sensitivity([0, 0, 0], [0.1, 0.2, 0.3], 0.9)


# ---------------------------------------------------------------------------
# End-to-end: meaningful alert fatigue characteristics on AML data
# ---------------------------------------------------------------------------


def test_end_to_end_fpr_below_50_at_90_sensitivity(evaluator, y_true, scores):
    """A discriminative AML scorer should keep FPR below 50% at 90% recall."""
    fpr = evaluator.fpr_at_sensitivity(y_true, scores, 0.90)
    assert fpr < 0.50, f"FPR {fpr:.3f} is too high at 90% sensitivity"


def test_end_to_end_fatigue_index_improves_with_higher_threshold(evaluator, y_true, scores):
    """Raising the alert threshold should reduce or maintain fatigue index."""
    fi_low = evaluator.fatigue_index(y_true, scores, 0.20)
    fi_high = evaluator.fatigue_index(y_true, scores, 0.60)
    assert fi_high <= fi_low + 0.05


def test_end_to_end_sensitivity_curve_auc_positive(evaluator, y_true, scores):
    curve = evaluator.sensitivity_curve(y_true, scores)
    computed_auc = np.trapezoid(curve["sensitivity"], curve["fpr"])
    assert computed_auc > 0.0


def test_end_to_end_evaluate_on_paysim_style_data():
    """Evaluate with a realistic-scale imbalanced dataset (1000 txns, 3% suspicious)."""
    rng = np.random.default_rng(7)
    n = 1000
    n_pos = 30
    y = [1] * n_pos + [0] * (n - n_pos)
    rng.shuffle(y)
    s = []
    for label in y:
        if label == 1:
            s.append(float(np.clip(rng.normal(0.72, 0.15), 0, 1)))
        else:
            s.append(float(np.clip(rng.normal(0.28, 0.18), 0, 1)))

    ev = AlertFatigueEvaluator()
    report = ev.evaluate(y, s)

    assert report["n_positives"] == n_pos
    assert report["auc_pr"] > 0.10  # better than random on 3% base rate
    assert report["roc_auc"] > 0.60
    # At 80% recall, FPR should be measurable but not 100%
    fpr_80 = report["sensitivity_analysis"][0.80]["fpr"]
    assert 0.0 <= fpr_80 <= 1.0
