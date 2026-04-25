"""
tests/test_core/test_scorer.py
================================
Unit tests for the unified FinCrime risk scorer.

Covers: FusionConfig validation, score() output schema, all three fusion
strategies, single-signal degradation, risk tier mapping, audit log,
sorting behaviour, edge cases, and end-to-end fusion correctness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincrime_ml.core.scorer import (
    FUSION_STRATEGIES,
    FinCrimeScorer,
    FusionConfig,
    _assign_risk_tier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(
    n: int = 20,
    fraud_low: float = 0.1,
    fraud_high: float = 0.9,
    aml_low: float = 0.1,
    aml_high: float = 0.9,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a small scored DataFrame with both fraud and AML scores."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "transaction_id": [f"T{i:04d}" for i in range(n)],
            "fraud_score": rng.uniform(fraud_low, fraud_high, n),
            "aml_score": rng.uniform(aml_low, aml_high, n),
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorer() -> FinCrimeScorer:
    return FinCrimeScorer()


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _df(n=50, seed=7)


@pytest.fixture(scope="module")
def result(scorer, df) -> pd.DataFrame:
    return scorer.score(df)


# ---------------------------------------------------------------------------
# FusionConfig — defaults and validation
# ---------------------------------------------------------------------------


def test_config_default_strategy():
    assert FusionConfig().strategy == "weighted_average"


def test_config_default_weights_equal():
    cfg = FusionConfig()
    assert cfg.fraud_weight == 0.5
    assert cfg.aml_weight == 0.5


def test_config_default_version():
    assert FusionConfig().version == "0.1.0"


def test_config_audit_log_enabled_by_default():
    assert FusionConfig().audit_log_enabled is True


def test_config_weights_must_sum_to_one_for_weighted_average():
    with pytest.raises(ValueError, match="sum to 1.0"):
        FusionConfig(fraud_weight=0.6, aml_weight=0.6)


def test_config_weights_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        FusionConfig(strategy="mean")


def test_config_fraud_weight_out_of_range():
    with pytest.raises(ValueError, match="fraud_weight"):
        FusionConfig(fraud_weight=1.5, aml_weight=-0.5)


def test_config_custom_weights_valid():
    cfg = FusionConfig(fraud_weight=0.7, aml_weight=0.3)
    assert cfg.fraud_weight == 0.7


def test_config_max_strategy_does_not_require_equal_weights():
    cfg = FusionConfig(strategy="max", fraud_weight=0.6, aml_weight=0.4)
    assert cfg.strategy == "max"


def test_config_harmonic_strategy_accepted():
    cfg = FusionConfig(strategy="harmonic_mean", fraud_weight=0.5, aml_weight=0.5)
    assert cfg.strategy == "harmonic_mean"


def test_fusion_strategies_constant():
    assert "weighted_average" in FUSION_STRATEGIES
    assert "max" in FUSION_STRATEGIES
    assert "harmonic_mean" in FUSION_STRATEGIES


# ---------------------------------------------------------------------------
# score() — output schema
# ---------------------------------------------------------------------------


def test_score_returns_dataframe(result):
    assert isinstance(result, pd.DataFrame)


def test_score_row_count_preserved(result, df):
    assert len(result) == len(df)


def test_score_has_required_columns(result):
    for col in (
        "transaction_id",
        "fraud_score",
        "aml_score",
        "unified_risk_score",
        "risk_tier",
        "model_version",
        "scored_at",
    ):
        assert col in result.columns


def test_score_unified_risk_score_in_unit_interval(result):
    assert (result["unified_risk_score"] >= 0.0).all()
    assert (result["unified_risk_score"] <= 1.0).all()


def test_score_fraud_score_preserved(result, df):
    merged = result.merge(
        df[["transaction_id", "fraud_score"]], on="transaction_id", suffixes=("_out", "_in")
    )
    assert np.allclose(merged["fraud_score_out"], merged["fraud_score_in"], atol=1e-5)


def test_score_aml_score_preserved(result, df):
    merged = result.merge(
        df[["transaction_id", "aml_score"]], on="transaction_id", suffixes=("_out", "_in")
    )
    assert np.allclose(merged["aml_score_out"], merged["aml_score_in"], atol=1e-5)


def test_score_risk_tier_valid_values(result):
    assert set(result["risk_tier"].unique()).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def test_score_sorted_descending_by_unified_score(result):
    assert result["unified_risk_score"].is_monotonic_decreasing


def test_score_fusion_strategy_column(result):
    assert (result["fusion_strategy"] == "weighted_average").all()


def test_score_model_version_column(result):
    assert (result["model_version"] == "0.1.0").all()


def test_score_scored_at_is_string(result):
    for val in result["scored_at"]:
        assert isinstance(val, str)
        assert "T" in val


def test_score_transaction_id_preserved(result, df):
    assert set(result["transaction_id"]) == set(df["transaction_id"])


# ---------------------------------------------------------------------------
# Weighted average strategy
# ---------------------------------------------------------------------------


def test_weighted_average_equal_weights_is_mean():
    scorer = FinCrimeScorer(FusionConfig(fraud_weight=0.5, aml_weight=0.5))
    df_in = pd.DataFrame({"transaction_id": ["T1"], "fraud_score": [0.6], "aml_score": [0.4]})
    result = scorer.score(df_in)
    assert result.iloc[0]["unified_risk_score"] == pytest.approx(0.5, abs=1e-5)


def test_weighted_average_asymmetric_weights():
    cfg = FusionConfig(fraud_weight=0.8, aml_weight=0.2)
    scorer = FinCrimeScorer(cfg)
    df_in = pd.DataFrame({"transaction_id": ["T1"], "fraud_score": [1.0], "aml_score": [0.0]})
    result = scorer.score(df_in)
    assert result.iloc[0]["unified_risk_score"] == pytest.approx(0.8, abs=1e-5)


def test_weighted_average_fraud_dominant():
    cfg = FusionConfig(fraud_weight=0.9, aml_weight=0.1)
    scorer = FinCrimeScorer(cfg)
    df_in = pd.DataFrame({"transaction_id": ["T1"], "fraud_score": [0.8], "aml_score": [0.2]})
    result = scorer.score(df_in)
    expected = 0.9 * 0.8 + 0.1 * 0.2
    assert result.iloc[0]["unified_risk_score"] == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Max strategy
# ---------------------------------------------------------------------------


def test_max_strategy_picks_higher_score():
    scorer = FinCrimeScorer(FusionConfig(strategy="max", fraud_weight=0.5, aml_weight=0.5))
    df_in = pd.DataFrame(
        {"transaction_id": ["T1", "T2"], "fraud_score": [0.3, 0.9], "aml_score": [0.8, 0.2]}
    )
    result = scorer.score(df_in)
    scores = result.set_index("transaction_id")["unified_risk_score"]
    assert scores["T1"] == pytest.approx(0.8, abs=1e-5)
    assert scores["T2"] == pytest.approx(0.9, abs=1e-5)


def test_max_strategy_never_lower_than_either_input():
    scorer = FinCrimeScorer(FusionConfig(strategy="max", fraud_weight=0.5, aml_weight=0.5))
    df_in = _df(n=30, seed=3)
    result = scorer.score(df_in)
    # result already carries the preserved input scores
    assert (result["unified_risk_score"] >= result["fraud_score"] - 1e-6).all()
    assert (result["unified_risk_score"] >= result["aml_score"] - 1e-6).all()


def test_max_strategy_fusion_col_reflects_strategy():
    scorer = FinCrimeScorer(FusionConfig(strategy="max", fraud_weight=0.5, aml_weight=0.5))
    result = scorer.score(_df(n=5, seed=1))
    assert (result["fusion_strategy"] == "max").all()


# ---------------------------------------------------------------------------
# Harmonic mean strategy
# ---------------------------------------------------------------------------


def test_harmonic_mean_symmetric():
    scorer = FinCrimeScorer(
        FusionConfig(strategy="harmonic_mean", fraud_weight=0.5, aml_weight=0.5)
    )
    df_in = pd.DataFrame({"transaction_id": ["T1"], "fraud_score": [0.8], "aml_score": [0.8]})
    result = scorer.score(df_in)
    assert result.iloc[0]["unified_risk_score"] == pytest.approx(0.8, abs=1e-5)


def test_harmonic_mean_zero_when_either_zero():
    scorer = FinCrimeScorer(
        FusionConfig(strategy="harmonic_mean", fraud_weight=0.5, aml_weight=0.5)
    )
    df_in = pd.DataFrame(
        {"transaction_id": ["T1", "T2"], "fraud_score": [0.0, 0.8], "aml_score": [0.8, 0.0]}
    )
    result = scorer.score(df_in)
    for _, row in result.iterrows():
        assert row["unified_risk_score"] == pytest.approx(0.0, abs=1e-5)


def test_harmonic_mean_lower_than_max():
    scorer_hm = FinCrimeScorer(
        FusionConfig(strategy="harmonic_mean", fraud_weight=0.5, aml_weight=0.5)
    )
    scorer_mx = FinCrimeScorer(FusionConfig(strategy="max", fraud_weight=0.5, aml_weight=0.5))
    df_in = _df(n=20, seed=99)
    hm = scorer_hm.score(df_in)["unified_risk_score"]
    mx = scorer_mx.score(df_in)["unified_risk_score"]
    # harmonic mean is always <= max
    assert (hm.values <= mx.values + 1e-6).all()


# ---------------------------------------------------------------------------
# Single-signal degradation
# ---------------------------------------------------------------------------


def test_score_with_only_fraud_signal():
    scorer = FinCrimeScorer()
    df_in = pd.DataFrame({"transaction_id": ["T1", "T2"], "fraud_score": [0.7, 0.3]})
    result = scorer.score(df_in)
    assert len(result) == 2
    assert (result["aml_score"] == 0.0).all()
    assert (result["unified_risk_score"] == result["fraud_score"]).all()


def test_score_with_only_aml_signal():
    scorer = FinCrimeScorer()
    df_in = pd.DataFrame({"transaction_id": ["T1", "T2"], "aml_score": [0.8, 0.4]})
    result = scorer.score(df_in)
    assert len(result) == 2
    assert (result["fraud_score"] == 0.0).all()
    assert (result["unified_risk_score"] == result["aml_score"]).all()


def test_score_missing_both_signals_raises():
    scorer = FinCrimeScorer()
    df_in = pd.DataFrame({"transaction_id": ["T1"], "amount_gbp": [1000.0]})
    with pytest.raises(KeyError, match="neither"):
        scorer.score(df_in)


def test_score_custom_column_names():
    cfg = FusionConfig(fraud_score_col="f_score", aml_score_col="a_score")
    scorer = FinCrimeScorer(cfg)
    df_in = pd.DataFrame({"transaction_id": ["T1"], "f_score": [0.6], "a_score": [0.4]})
    result = scorer.score(df_in)
    assert result.iloc[0]["unified_risk_score"] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# Risk tier mapping (_assign_risk_tier)
# ---------------------------------------------------------------------------


def test_assign_risk_tier_critical():
    assert _assign_risk_tier(0.85) == "CRITICAL"


def test_assign_risk_tier_critical_above_threshold():
    assert _assign_risk_tier(0.99) == "CRITICAL"


def test_assign_risk_tier_high():
    assert _assign_risk_tier(0.75) == "HIGH"


def test_assign_risk_tier_high_lower_bound():
    assert _assign_risk_tier(0.65) == "HIGH"


def test_assign_risk_tier_medium():
    assert _assign_risk_tier(0.50) == "MEDIUM"


def test_assign_risk_tier_medium_lower_bound():
    assert _assign_risk_tier(0.40) == "MEDIUM"


def test_assign_risk_tier_low():
    assert _assign_risk_tier(0.20) == "LOW"


def test_assign_risk_tier_zero():
    assert _assign_risk_tier(0.0) == "LOW"


def test_risk_tier_consistent_with_unified_score(result):
    for _, row in result.iterrows():
        expected = _assign_risk_tier(row["unified_risk_score"])
        assert row["risk_tier"] == expected


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_audit_log_initially_empty():
    scorer = FinCrimeScorer()
    assert scorer.audit_log == []


def test_audit_log_populated_after_score():
    scorer = FinCrimeScorer()
    scorer.score(_df(n=5, seed=0))
    assert len(scorer.audit_log) == 1


def test_audit_log_event_is_score():
    scorer = FinCrimeScorer()
    scorer.score(_df(n=5, seed=0))
    assert scorer.audit_log[0]["event"] == "score"


def test_audit_log_contains_n_transactions():
    scorer = FinCrimeScorer()
    scorer.score(_df(n=10, seed=0))
    assert scorer.audit_log[0]["n_transactions"] == 10


def test_audit_log_contains_signal_flags():
    scorer = FinCrimeScorer()
    scorer.score(_df(n=5, seed=0))
    entry = scorer.audit_log[0]
    assert entry["has_fraud_signal"] is True
    assert entry["has_aml_signal"] is True


def test_audit_log_immutable_copy():
    scorer = FinCrimeScorer()
    scorer.score(_df(n=5, seed=0))
    log = scorer.audit_log
    log.clear()
    assert len(scorer.audit_log) == 1


def test_audit_log_accumulates():
    scorer = FinCrimeScorer()
    scorer.score(_df(n=5, seed=0))
    scorer.score(_df(n=5, seed=1))
    assert len(scorer.audit_log) == 2


def test_audit_log_disabled():
    cfg = FusionConfig(audit_log_enabled=False)
    scorer = FinCrimeScorer(cfg)
    scorer.score(_df(n=5, seed=0))
    assert scorer.audit_log == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_score_all_zeros():
    scorer = FinCrimeScorer()
    df_in = pd.DataFrame(
        {"transaction_id": ["T1", "T2"], "fraud_score": [0.0, 0.0], "aml_score": [0.0, 0.0]}
    )
    result = scorer.score(df_in)
    assert (result["unified_risk_score"] == 0.0).all()
    assert (result["risk_tier"] == "LOW").all()


def test_score_all_ones():
    scorer = FinCrimeScorer()
    df_in = pd.DataFrame(
        {"transaction_id": ["T1", "T2"], "fraud_score": [1.0, 1.0], "aml_score": [1.0, 1.0]}
    )
    result = scorer.score(df_in)
    assert (result["unified_risk_score"] == 1.0).all()
    assert (result["risk_tier"] == "CRITICAL").all()


def test_score_single_row():
    scorer = FinCrimeScorer()
    df_in = pd.DataFrame({"transaction_id": ["T1"], "fraud_score": [0.5], "aml_score": [0.5]})
    result = scorer.score(df_in)
    assert len(result) == 1


def test_score_without_transaction_id_uses_index():
    scorer = FinCrimeScorer()
    df_in = pd.DataFrame({"fraud_score": [0.6, 0.3], "aml_score": [0.4, 0.7]})
    result = scorer.score(df_in)
    assert "transaction_id" in result.columns
    assert len(result) == 2


# ---------------------------------------------------------------------------
# End-to-end: strategy comparison
# ---------------------------------------------------------------------------


def test_end_to_end_all_strategies_run_without_error():
    df_in = _df(n=100, seed=42)
    for strategy in FUSION_STRATEGIES:
        cfg = FusionConfig(strategy=strategy, fraud_weight=0.5, aml_weight=0.5)
        scorer = FinCrimeScorer(cfg)
        result = scorer.score(df_in)
        assert len(result) == 100
        assert (result["unified_risk_score"].between(0.0, 1.0)).all()


def test_end_to_end_high_fraud_dominates_with_high_fraud_weight():
    cfg = FusionConfig(fraud_weight=0.9, aml_weight=0.1)
    scorer = FinCrimeScorer(cfg)
    df_in = pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(20)],
            "fraud_score": [0.95] * 20,
            "aml_score": [0.05] * 20,
        }
    )
    result = scorer.score(df_in)
    # With 90% fraud weight and fraud=0.95, unified should be ~0.86 (CRITICAL)
    assert (result["unified_risk_score"] > 0.80).all()
    assert (result["risk_tier"] == "CRITICAL").all()


def test_end_to_end_max_strategy_conservative():
    cfg_max = FusionConfig(strategy="max", fraud_weight=0.5, aml_weight=0.5)
    cfg_avg = FusionConfig(strategy="weighted_average", fraud_weight=0.5, aml_weight=0.5)
    scorer_max = FinCrimeScorer(cfg_max)
    scorer_avg = FinCrimeScorer(cfg_avg)
    df_in = _df(n=50, seed=7)

    res_max = scorer_max.score(df_in).set_index("transaction_id")["unified_risk_score"]
    res_avg = scorer_avg.score(df_in).set_index("transaction_id")["unified_risk_score"]

    # Max should always be >= average for equal weights
    common = res_max.index.intersection(res_avg.index)
    assert (res_max[common].values >= res_avg[common].values - 1e-6).all()
