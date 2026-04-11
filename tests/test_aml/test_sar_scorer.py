"""
tests/test_aml/test_sar_scorer.py
====================================
Unit tests for the SAR trigger scorer.

Covers: SARScorerConfig, score() output schema, trigger rule firing,
priority assignment, SAR recommendation logic, summary_report(),
audit log, edge cases (empty result, missing required columns), and
end-to-end alert generation for high-risk transactions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fincrime_ml.aml.sar_scorer import (
    SAR_ALERT_COLS,
    STRUCTURING_LOWER_GBP,
    STRUCTURING_UPPER_GBP,
    SUSPICIOUS_TYPOLOGIES,
    TRIGGER_REGULATORY_REFS,
    SARScorer,
    SARScorerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(**kwargs) -> pd.DataFrame:
    """Build a single-row DataFrame with sensible defaults for required cols."""
    defaults = {
        "transaction_id": "T0001",
        "risk_score": 0.40,
        "risk_tier": "MEDIUM",
        "amount_gbp": 200.0,
        "typology": "normal",
        "structuring_flag": 0,
        "is_mule_sender": 0,
        "is_mule_receiver": 0,
        "rapid_movement_flag": 0,
        "layering_depth": 0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorer() -> SARScorer:
    return SARScorer()


@pytest.fixture(scope="module")
def mixed_df() -> pd.DataFrame:
    """DataFrame covering all six trigger rules plus a below-threshold row."""
    rows = [
        # P1: CRITICAL tier, all six triggers
        {
            "transaction_id": "T_CRITICAL",
            "risk_score": 0.92,
            "risk_tier": "CRITICAL",
            "amount_gbp": 9_100.0,
            "typology": "structuring",
            "is_mule_sender": 1,
            "is_mule_receiver": 0,
            "rapid_movement_flag": 1,
            "layering_depth": 2,
        },
        # P2: HIGH tier, single trigger (high risk score)
        {
            "transaction_id": "T_HIGH",
            "risk_score": 0.70,
            "risk_tier": "HIGH",
            "amount_gbp": 500.0,
            "typology": "normal",
            "is_mule_sender": 0,
            "is_mule_receiver": 0,
            "rapid_movement_flag": 0,
            "layering_depth": 0,
        },
        # P3: MEDIUM tier, one trigger (layering)
        {
            "transaction_id": "T_MEDIUM",
            "risk_score": 0.35,
            "risk_tier": "MEDIUM",
            "amount_gbp": 80.0,
            "typology": "normal",
            "is_mule_sender": 0,
            "is_mule_receiver": 0,
            "rapid_movement_flag": 0,
            "layering_depth": 1,
        },
        # Excluded: below alert threshold
        {
            "transaction_id": "T_EXCLUDED",
            "risk_score": 0.05,
            "risk_tier": "LOW",
            "amount_gbp": 10.0,
            "typology": "normal",
            "is_mule_sender": 0,
            "is_mule_receiver": 0,
            "rapid_movement_flag": 0,
            "layering_depth": 0,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def mixed_alerts(scorer, mixed_df) -> pd.DataFrame:
    return scorer.score(mixed_df)


@pytest.fixture(scope="module")
def empty_alerts(scorer) -> pd.DataFrame:
    """Score a DataFrame where no transaction meets the alert threshold."""
    df = pd.DataFrame(
        [
            {
                "transaction_id": "T_LOW",
                "risk_score": 0.10,
                "risk_tier": "LOW",
                "amount_gbp": 20.0,
                "typology": "normal",
            }
        ]
    )
    return scorer.score(df)


# ---------------------------------------------------------------------------
# SARScorerConfig
# ---------------------------------------------------------------------------


def test_config_default_alert_threshold():
    cfg = SARScorerConfig()
    assert cfg.alert_score_threshold == 0.30


def test_config_default_sar_threshold():
    cfg = SARScorerConfig()
    assert cfg.sar_score_threshold == 0.65


def test_config_default_structuring_lower():
    cfg = SARScorerConfig()
    assert cfg.structuring_lower == STRUCTURING_LOWER_GBP


def test_config_default_structuring_upper():
    cfg = SARScorerConfig()
    assert cfg.structuring_upper == STRUCTURING_UPPER_GBP


def test_config_default_version():
    cfg = SARScorerConfig()
    assert cfg.version == "0.1.0"


def test_config_audit_log_enabled_by_default():
    cfg = SARScorerConfig()
    assert cfg.audit_log_enabled is True


def test_config_custom_thresholds():
    cfg = SARScorerConfig(alert_score_threshold=0.50, sar_score_threshold=0.80)
    scorer = SARScorer(config=cfg)
    assert scorer.config.alert_score_threshold == 0.50
    assert scorer.config.sar_score_threshold == 0.80


def test_config_custom_version():
    cfg = SARScorerConfig(version="2.0.0")
    scorer = SARScorer(config=cfg)
    assert scorer.config.version == "2.0.0"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_trigger_regulatory_refs_covers_all_rules():
    rules = {
        "HIGH_RISK_SCORE",
        "STRUCTURING_AMOUNT",
        "MULE_INVOLVEMENT",
        "RAPID_MOVEMENT",
        "CHAIN_LAYERING",
        "SUSPICIOUS_TYPOLOGY",
    }
    assert rules == set(TRIGGER_REGULATORY_REFS.keys())


def test_suspicious_typologies_contains_key_types():
    for t in ("structuring", "layering", "integration"):
        assert t in SUSPICIOUS_TYPOLOGIES


def test_sar_alert_cols_complete():
    for col in (
        "alert_id",
        "transaction_id",
        "risk_score",
        "risk_tier",
        "priority",
        "n_triggers",
        "trigger_reasons",
        "sar_recommended",
        "regulatory_refs",
        "mlro_summary",
        "amount_gbp",
        "typology",
        "scored_at",
    ):
        assert col in SAR_ALERT_COLS


# ---------------------------------------------------------------------------
# score() — output schema
# ---------------------------------------------------------------------------


def test_score_returns_dataframe(mixed_alerts):
    assert isinstance(mixed_alerts, pd.DataFrame)


def test_score_has_all_required_columns(mixed_alerts):
    for col in SAR_ALERT_COLS:
        assert col in mixed_alerts.columns, f"Missing column: {col}"


def test_score_excludes_below_threshold_rows(mixed_alerts):
    assert "T_EXCLUDED" not in mixed_alerts["transaction_id"].values


def test_score_alert_id_format(mixed_alerts):
    for aid in mixed_alerts["alert_id"]:
        assert aid.startswith("SAR-")
        assert len(aid) == 16  # "SAR-" + 12 hex chars


def test_score_risk_score_preserved(mixed_alerts):
    row = mixed_alerts[mixed_alerts["transaction_id"] == "T_CRITICAL"].iloc[0]
    assert abs(row["risk_score"] - 0.92) < 1e-3


def test_score_risk_tier_preserved(mixed_alerts):
    row = mixed_alerts[mixed_alerts["transaction_id"] == "T_CRITICAL"].iloc[0]
    assert row["risk_tier"] == "CRITICAL"


def test_score_scored_at_is_string(mixed_alerts):
    for val in mixed_alerts["scored_at"]:
        assert isinstance(val, str)
        assert "T" in val  # ISO-8601 datetime separator


def test_score_n_triggers_positive(mixed_alerts):
    assert (mixed_alerts["n_triggers"] >= 1).all()


def test_score_sar_recommended_binary(mixed_alerts):
    assert set(mixed_alerts["sar_recommended"].unique()).issubset({0, 1})


def test_score_sorted_priority_asc(mixed_alerts):
    assert mixed_alerts["priority"].is_monotonic_increasing


def test_score_within_same_priority_sorted_risk_score_desc(mixed_alerts):
    for priority in mixed_alerts["priority"].unique():
        subset = mixed_alerts[mixed_alerts["priority"] == priority]
        if len(subset) > 1:
            assert subset["risk_score"].is_monotonic_decreasing


def test_score_row_count_excludes_below_threshold(mixed_alerts, mixed_df):
    qualifying = mixed_df[mixed_df["risk_score"] >= 0.30]
    # Only rows that also match at least one trigger are included
    assert len(mixed_alerts) <= len(qualifying)


# ---------------------------------------------------------------------------
# Trigger rule firing (each rule independently)
# ---------------------------------------------------------------------------


def test_trigger_high_risk_score_fires():
    scorer = SARScorer()
    df = _row(risk_score=0.70, risk_tier="HIGH")
    alerts = scorer.score(df)
    assert len(alerts) == 1
    assert "HIGH_RISK_SCORE" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_high_risk_score_does_not_fire_below_threshold():
    scorer = SARScorer()
    # risk_score above alert threshold but below sar_score_threshold
    # and no other triggers
    df = _row(risk_score=0.40, risk_tier="MEDIUM", layering_depth=0)
    alerts = scorer.score(df)
    if len(alerts) > 0:
        assert "HIGH_RISK_SCORE" not in alerts.iloc[0]["trigger_reasons"]


def test_trigger_structuring_amount_fires_in_band():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", amount_gbp=9_000.0)
    alerts = scorer.score(df)
    assert len(alerts) == 1
    assert "STRUCTURING_AMOUNT" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_structuring_amount_fires_at_lower_bound():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", amount_gbp=STRUCTURING_LOWER_GBP)
    alerts = scorer.score(df)
    assert "STRUCTURING_AMOUNT" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_structuring_amount_fires_at_upper_bound():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", amount_gbp=STRUCTURING_UPPER_GBP)
    alerts = scorer.score(df)
    assert "STRUCTURING_AMOUNT" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_structuring_amount_does_not_fire_outside_band():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", amount_gbp=100.0, layering_depth=0)
    alerts = scorer.score(df)
    if len(alerts) > 0:
        assert "STRUCTURING_AMOUNT" not in alerts.iloc[0]["trigger_reasons"]


def test_trigger_mule_sender_fires():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", is_mule_sender=1)
    alerts = scorer.score(df)
    assert "MULE_INVOLVEMENT" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_mule_receiver_fires():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", is_mule_receiver=1)
    alerts = scorer.score(df)
    assert "MULE_INVOLVEMENT" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_rapid_movement_fires():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", rapid_movement_flag=1)
    alerts = scorer.score(df)
    assert "RAPID_MOVEMENT" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_chain_layering_fires():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", layering_depth=1)
    alerts = scorer.score(df)
    assert "CHAIN_LAYERING" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_chain_layering_depth_zero_does_not_fire():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", layering_depth=0, amount_gbp=100.0)
    alerts = scorer.score(df)
    if len(alerts) > 0:
        assert "CHAIN_LAYERING" not in alerts.iloc[0]["trigger_reasons"]


def test_trigger_suspicious_typology_structuring():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", typology="structuring")
    alerts = scorer.score(df)
    assert "SUSPICIOUS_TYPOLOGY" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_suspicious_typology_layering():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", typology="layering")
    alerts = scorer.score(df)
    assert "SUSPICIOUS_TYPOLOGY" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_suspicious_typology_integration():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", typology="integration")
    alerts = scorer.score(df)
    assert "SUSPICIOUS_TYPOLOGY" in alerts.iloc[0]["trigger_reasons"]


def test_trigger_unknown_typology_does_not_fire():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", typology="normal", layering_depth=0)
    alerts = scorer.score(df)
    if len(alerts) > 0:
        assert "SUSPICIOUS_TYPOLOGY" not in alerts.iloc[0]["trigger_reasons"]


def test_all_six_triggers_can_fire_simultaneously():
    scorer = SARScorer()
    df = _row(
        risk_score=0.90,
        risk_tier="CRITICAL",
        amount_gbp=9_000.0,
        is_mule_sender=1,
        rapid_movement_flag=1,
        layering_depth=3,
        typology="structuring",
    )
    alerts = scorer.score(df)
    assert alerts.iloc[0]["n_triggers"] == 6


# ---------------------------------------------------------------------------
# Priority assignment
# ---------------------------------------------------------------------------


def test_priority_critical_tier_gives_p1():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="CRITICAL", layering_depth=1)
    alerts = scorer.score(df)
    assert alerts.iloc[0]["priority"] == 1


def test_priority_three_triggers_gives_p1():
    scorer = SARScorer()
    df = _row(
        risk_score=0.40,
        risk_tier="MEDIUM",
        layering_depth=1,
        rapid_movement_flag=1,
        is_mule_sender=1,
    )
    alerts = scorer.score(df)
    assert alerts.iloc[0]["n_triggers"] >= 3
    assert alerts.iloc[0]["priority"] == 1


def test_priority_high_tier_single_trigger_gives_p2():
    scorer = SARScorer()
    df = _row(risk_score=0.70, risk_tier="HIGH")
    alerts = scorer.score(df)
    # HIGH tier, 1 trigger (HIGH_RISK_SCORE) → P2
    assert alerts.iloc[0]["priority"] == 2


def test_priority_two_triggers_gives_p2():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", layering_depth=1, rapid_movement_flag=1)
    alerts = scorer.score(df)
    assert alerts.iloc[0]["n_triggers"] == 2
    assert alerts.iloc[0]["priority"] == 2


def test_priority_medium_one_trigger_gives_p3():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", layering_depth=1)
    alerts = scorer.score(df)
    assert alerts.iloc[0]["n_triggers"] == 1
    assert alerts.iloc[0]["priority"] == 3


def test_assign_priority_static_critical():
    assert SARScorer._assign_priority("CRITICAL", 1) == 1


def test_assign_priority_static_three_triggers():
    assert SARScorer._assign_priority("MEDIUM", 3) == 1


def test_assign_priority_static_high_tier():
    assert SARScorer._assign_priority("HIGH", 1) == 2


def test_assign_priority_static_two_triggers():
    assert SARScorer._assign_priority("LOW", 2) == 2


def test_assign_priority_static_medium():
    assert SARScorer._assign_priority("MEDIUM", 1) == 3


# ---------------------------------------------------------------------------
# SAR recommendation logic
# ---------------------------------------------------------------------------


def test_sar_recommended_when_score_above_sar_threshold():
    scorer = SARScorer()
    df = _row(risk_score=0.70, risk_tier="HIGH")
    alerts = scorer.score(df)
    assert alerts.iloc[0]["sar_recommended"] == 1


def test_sar_recommended_when_priority_1():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="CRITICAL", layering_depth=1)
    alerts = scorer.score(df)
    assert alerts.iloc[0]["priority"] == 1
    assert alerts.iloc[0]["sar_recommended"] == 1


def test_sar_not_recommended_medium_low_score():
    scorer = SARScorer()
    # MEDIUM tier, 1 trigger, score well below sar_score_threshold
    df = _row(risk_score=0.40, risk_tier="MEDIUM", layering_depth=1)
    alerts = scorer.score(df)
    assert alerts.iloc[0]["sar_recommended"] == 0


# ---------------------------------------------------------------------------
# Regulatory references
# ---------------------------------------------------------------------------


def test_regulatory_refs_non_empty_for_alerts(mixed_alerts):
    for refs in mixed_alerts["regulatory_refs"]:
        assert len(refs) > 0


def test_regulatory_refs_include_poca_for_structuring():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", amount_gbp=9_000.0)
    alerts = scorer.score(df)
    assert "POCA 2002" in alerts.iloc[0]["regulatory_refs"]


def test_regulatory_refs_include_mlr_for_mule():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", is_mule_sender=1)
    alerts = scorer.score(df)
    assert "MLR 2017" in alerts.iloc[0]["regulatory_refs"]


def test_collect_regulatory_refs_deduplicates():
    # Passing the same trigger twice should include its ref only once
    scorer = SARScorer()
    refs = scorer._collect_regulatory_refs(["MULE_INVOLVEMENT", "MULE_INVOLVEMENT"])
    ref_str = TRIGGER_REGULATORY_REFS["MULE_INVOLVEMENT"]
    assert refs.count(ref_str) == 1


# ---------------------------------------------------------------------------
# MLRO summary
# ---------------------------------------------------------------------------


def test_mlro_summary_non_empty(mixed_alerts):
    for summary in mixed_alerts["mlro_summary"]:
        assert isinstance(summary, str)
        assert len(summary) > 0


def test_mlro_summary_contains_transaction_id(mixed_alerts):
    row = mixed_alerts[mixed_alerts["transaction_id"] == "T_CRITICAL"].iloc[0]
    assert "T_CRITICAL" in row["mlro_summary"]


def test_mlro_summary_mentions_priority(mixed_alerts):
    row = mixed_alerts[mixed_alerts["transaction_id"] == "T_CRITICAL"].iloc[0]
    assert "Priority 1" in row["mlro_summary"]


def test_mlro_summary_mentions_structuring_poca_when_triggered():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", amount_gbp=9_000.0)
    alerts = scorer.score(df)
    assert "POCA 2002" in alerts.iloc[0]["mlro_summary"]


def test_mlro_summary_mentions_mule_edd_when_triggered():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", is_mule_sender=1)
    alerts = scorer.score(df)
    assert "MLR 2017" in alerts.iloc[0]["mlro_summary"]


def test_mlro_summary_mentions_rapid_movement_when_triggered():
    scorer = SARScorer()
    df = _row(risk_score=0.40, risk_tier="MEDIUM", rapid_movement_flag=1)
    alerts = scorer.score(df)
    assert "FATF" in alerts.iloc[0]["mlro_summary"]


# ---------------------------------------------------------------------------
# summary_report()
# ---------------------------------------------------------------------------


def test_summary_report_returns_dict(scorer, mixed_alerts):
    report = scorer.summary_report(mixed_alerts)
    assert isinstance(report, dict)


def test_summary_report_has_required_keys(scorer, mixed_alerts):
    report = scorer.summary_report(mixed_alerts)
    for key in (
        "n_alerts",
        "n_sar_recommended",
        "sar_rate",
        "priority_counts",
        "top_triggers",
        "mean_risk_score",
    ):
        assert key in report


def test_summary_report_n_alerts_matches(scorer, mixed_alerts):
    report = scorer.summary_report(mixed_alerts)
    assert report["n_alerts"] == len(mixed_alerts)


def test_summary_report_sar_rate_in_unit_interval(scorer, mixed_alerts):
    report = scorer.summary_report(mixed_alerts)
    assert 0.0 <= report["sar_rate"] <= 1.0


def test_summary_report_priority_counts_sum_to_n_alerts(scorer, mixed_alerts):
    report = scorer.summary_report(mixed_alerts)
    total = sum(report["priority_counts"].values())
    assert total == report["n_alerts"]


def test_summary_report_priority_counts_has_keys_1_2_3(scorer, mixed_alerts):
    report = scorer.summary_report(mixed_alerts)
    assert set(report["priority_counts"].keys()) == {1, 2, 3}


def test_summary_report_top_triggers_is_dict(scorer, mixed_alerts):
    report = scorer.summary_report(mixed_alerts)
    assert isinstance(report["top_triggers"], dict)


def test_summary_report_mean_risk_score_positive(scorer, mixed_alerts):
    report = scorer.summary_report(mixed_alerts)
    assert report["mean_risk_score"] > 0.0


def test_summary_report_empty_alerts(scorer, empty_alerts):
    report = scorer.summary_report(empty_alerts)
    assert report["n_alerts"] == 0
    assert report["sar_rate"] == 0.0
    assert report["mean_risk_score"] == 0.0


def test_summary_report_empty_priority_counts_zero(scorer, empty_alerts):
    report = scorer.summary_report(empty_alerts)
    assert report["priority_counts"] == {1: 0, 2: 0, 3: 0}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_audit_log_initially_empty():
    scorer = SARScorer()
    assert scorer.audit_log == []


def test_audit_log_populated_after_score(mixed_alerts):
    scorer = SARScorer()
    scorer.score(_row(risk_score=0.70, risk_tier="HIGH"))
    assert len(scorer.audit_log) == 1


def test_audit_log_event_is_score():
    scorer = SARScorer()
    scorer.score(_row(risk_score=0.70, risk_tier="HIGH"))
    assert scorer.audit_log[0]["event"] == "score"


def test_audit_log_contains_n_alerts():
    scorer = SARScorer()
    scorer.score(_row(risk_score=0.70, risk_tier="HIGH"))
    assert "n_alerts_generated" in scorer.audit_log[0]


def test_audit_log_contains_timestamp():
    scorer = SARScorer()
    scorer.score(_row(risk_score=0.70, risk_tier="HIGH"))
    assert "timestamp" in scorer.audit_log[0]


def test_audit_log_immutable_copy():
    scorer = SARScorer()
    scorer.score(_row(risk_score=0.70, risk_tier="HIGH"))
    log_copy = scorer.audit_log
    log_copy.clear()
    assert len(scorer.audit_log) == 1


def test_audit_log_accumulates_across_calls():
    scorer = SARScorer()
    scorer.score(_row(risk_score=0.70, risk_tier="HIGH"))
    scorer.score(_row(risk_score=0.80, risk_tier="CRITICAL", layering_depth=1))
    assert len(scorer.audit_log) == 2


def test_audit_log_disabled_when_config_off():
    cfg = SARScorerConfig(audit_log_enabled=False)
    scorer = SARScorer(config=cfg)
    scorer.score(_row(risk_score=0.70, risk_tier="HIGH"))
    assert scorer.audit_log == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_score_empty_input_returns_empty_dataframe(scorer):
    empty_df = pd.DataFrame(columns=["risk_score", "risk_tier"])
    result = scorer.score(empty_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    for col in SAR_ALERT_COLS:
        assert col in result.columns


def test_score_empty_result_has_sar_alert_cols(empty_alerts):
    for col in SAR_ALERT_COLS:
        assert col in empty_alerts.columns


def test_score_missing_risk_score_raises():
    scorer = SARScorer()
    df = pd.DataFrame([{"risk_tier": "HIGH", "amount_gbp": 100.0}])
    with pytest.raises(KeyError, match="required columns missing"):
        scorer.score(df)


def test_score_missing_risk_tier_raises():
    scorer = SARScorer()
    df = pd.DataFrame([{"risk_score": 0.70, "amount_gbp": 100.0}])
    with pytest.raises(KeyError, match="required columns missing"):
        scorer.score(df)


def test_score_optional_cols_absent_does_not_raise():
    scorer = SARScorer()
    # Minimal input — only required columns
    df = pd.DataFrame([{"risk_score": 0.40, "risk_tier": "MEDIUM", "typology": "structuring"}])
    result = scorer.score(df)
    assert isinstance(result, pd.DataFrame)


def test_score_without_transaction_id_uses_index():
    scorer = SARScorer()
    df = pd.DataFrame(
        [
            {
                "risk_score": 0.70,
                "risk_tier": "HIGH",
                "amount_gbp": 100.0,
            }
        ]
    )
    result = scorer.score(df)
    assert len(result) == 1
    # transaction_id column should still be present (using index as fallback)
    assert "transaction_id" in result.columns


def test_score_custom_alert_threshold_filters_more():
    cfg = SARScorerConfig(alert_score_threshold=0.80)
    scorer = SARScorer(config=cfg)
    df = pd.DataFrame(
        [
            {"transaction_id": "T1", "risk_score": 0.70, "risk_tier": "HIGH"},
            {
                "transaction_id": "T2",
                "risk_score": 0.90,
                "risk_tier": "CRITICAL",
                "layering_depth": 1,
            },
        ]
    )
    result = scorer.score(df)
    # T1 is below 0.80, so only T2 should appear (if it has a trigger)
    if len(result) > 0:
        assert "T1" not in result["transaction_id"].values


# ---------------------------------------------------------------------------
# End-to-end: high-risk transactions generate Priority 1 alerts
# ---------------------------------------------------------------------------


def test_end_to_end_critical_transactions_generate_p1_alerts():
    """CRITICAL-tier transactions with multiple triggers must produce P1 alerts."""
    scorer = SARScorer()

    high_risk_rows = []
    for i in range(5):
        high_risk_rows.append(
            {
                "transaction_id": f"T_HI_{i:03d}",
                "risk_score": 0.88,
                "risk_tier": "CRITICAL",
                "amount_gbp": 9_200.0,
                "typology": "structuring",
                "is_mule_sender": 1,
                "rapid_movement_flag": 1,
                "layering_depth": 2,
            }
        )
    # Low-risk padding rows (no triggers, below threshold)
    for i in range(20):
        high_risk_rows.append(
            {
                "transaction_id": f"T_LO_{i:03d}",
                "risk_score": 0.05,
                "risk_tier": "LOW",
                "amount_gbp": 30.0,
                "typology": "normal",
                "is_mule_sender": 0,
                "rapid_movement_flag": 0,
                "layering_depth": 0,
            }
        )

    df = pd.DataFrame(high_risk_rows)
    alerts = scorer.score(df)

    p1_alerts = alerts[alerts["priority"] == 1]
    assert len(p1_alerts) == 5, f"Expected 5 P1 alerts, got {len(p1_alerts)}"
    assert (p1_alerts["sar_recommended"] == 1).all()
    assert (p1_alerts["n_triggers"] >= 3).all()


def test_end_to_end_output_sorted_highest_priority_first():
    """Alert queue must be sorted priority asc (P1 first) then risk_score desc."""
    scorer = SARScorer()

    df = pd.DataFrame(
        [
            {
                "transaction_id": "T_LOW_RISK",
                "risk_score": 0.40,
                "risk_tier": "MEDIUM",
                "layering_depth": 1,
            },
            {
                "transaction_id": "T_HIGH_RISK",
                "risk_score": 0.90,
                "risk_tier": "CRITICAL",
                "layering_depth": 3,
                "is_mule_sender": 1,
                "rapid_movement_flag": 1,
                "typology": "structuring",
                "amount_gbp": 9_000.0,
            },
        ]
    )

    alerts = scorer.score(df)
    # T_HIGH_RISK must come first (P1 before P3)
    assert alerts.iloc[0]["transaction_id"] == "T_HIGH_RISK"


def test_end_to_end_sar_rate_high_for_critical_batch():
    scorer = SARScorer()
    rows = [
        {
            "transaction_id": f"T{i}",
            "risk_score": 0.80,
            "risk_tier": "CRITICAL",
            "layering_depth": 2,
            "is_mule_sender": 1,
        }
        for i in range(10)
    ]
    df = pd.DataFrame(rows)
    alerts = scorer.score(df)
    report = scorer.summary_report(alerts)
    assert report["sar_rate"] == 1.0
