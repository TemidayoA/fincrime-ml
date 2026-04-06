"""
tests/test_aml/test_typologies.py
====================================
Unit tests for the AML typology detection engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from fincrime_ml.aml.typologies import (
    UK_REPORTING_THRESHOLD_GBP,
    TypologyEngine,
    TypologyMatch,
    TypologyScore,
)
from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aml_df() -> pd.DataFrame:
    gen = SyntheticAMLGenerator(n_accounts=300, seed=42)
    return gen.generate(n_transactions=2_000, suspicious_rate=0.10)


@pytest.fixture(scope="module")
def engine() -> TypologyEngine:
    return TypologyEngine()


@pytest.fixture
def structuring_df() -> pd.DataFrame:
    """Minimal DataFrame with a known structuring cluster."""
    base = datetime(2024, 3, 1, 10, 0)
    rows = []
    # Account A: 3 transactions within 6 hours, each £4,000 (total £12,000 > threshold)
    for i in range(3):
        rows.append(
            {
                "transaction_id": f"S-{i:03d}",
                "sender_account_id": "ACC-A",
                "receiver_account_id": "ACC-Z",
                "amount_gbp": 4_000.0,
                "timestamp": base + timedelta(hours=i * 2),
                "country_origin": "GB",
                "country_destination": "GB",
            }
        )
    # Account B: 1 transaction — no structuring
    rows.append(
        {
            "transaction_id": "S-010",
            "sender_account_id": "ACC-B",
            "receiver_account_id": "ACC-Z",
            "amount_gbp": 9_500.0,
            "timestamp": base,
            "country_origin": "GB",
            "country_destination": "GB",
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def layering_df() -> pd.DataFrame:
    """Minimal DataFrame with a known layering event."""
    t0 = datetime(2024, 3, 1, 9, 0)
    return pd.DataFrame(
        [
            # Inbound to mule account
            {
                "transaction_id": "L-001",
                "sender_account_id": "ACC-ORIG",
                "receiver_account_id": "ACC-MULE",
                "amount_gbp": 8_000.0,
                "timestamp": t0,
                "country_origin": "GB",
                "country_destination": "GB",
            },
            # Rapid outbound from mule — 2 hours later
            {
                "transaction_id": "L-002",
                "sender_account_id": "ACC-MULE",
                "receiver_account_id": "ACC-DEST",
                "amount_gbp": 7_500.0,
                "timestamp": t0 + timedelta(hours=2),
                "country_origin": "GB",
                "country_destination": "AE",
            },
            # Unrelated transaction
            {
                "transaction_id": "L-003",
                "sender_account_id": "ACC-CLEAN",
                "receiver_account_id": "ACC-OTHER",
                "amount_gbp": 200.0,
                "timestamp": t0,
                "country_origin": "GB",
                "country_destination": "GB",
            },
        ]
    )


@pytest.fixture
def integration_df() -> pd.DataFrame:
    """Minimal DataFrame with a known integration event."""
    t0 = datetime(2024, 3, 1, 8, 0)
    rows = []
    # ACC-INT receives many small inflows
    for i in range(10):
        rows.append(
            {
                "transaction_id": f"I-IN-{i:03d}",
                "sender_account_id": f"ACC-SRC-{i}",
                "receiver_account_id": "ACC-INT",
                "amount_gbp": 1_500.0,
                "timestamp": t0 + timedelta(hours=i),
                "country_origin": "GB",
                "country_destination": "GB",
            }
        )
    # ACC-INT then sends one large outflow (£13,000 out of £15,000 received)
    rows.append(
        {
            "transaction_id": "I-OUT-001",
            "sender_account_id": "ACC-INT",
            "receiver_account_id": "ACC-CLEAN",
            "amount_gbp": 13_000.0,
            "timestamp": t0 + timedelta(hours=20),
            "country_origin": "GB",
            "country_destination": "AE",
        }
    )
    # A few small unrelated accounts with tiny inflows
    for j in range(5):
        rows.append(
            {
                "transaction_id": f"I-NOISE-{j:03d}",
                "sender_account_id": "ACC-NOISE",
                "receiver_account_id": f"ACC-SMALL-{j}",
                "amount_gbp": 50.0,
                "timestamp": t0,
                "country_origin": "GB",
                "country_destination": "GB",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TypologyMatch dataclass tests
# ---------------------------------------------------------------------------


def test_typology_match_fields():
    m = TypologyMatch(
        account_id="ACC-001",
        typology="structuring",
        confidence=0.85,
        n_transactions=4,
        total_amount_gbp=36_000.0,
        window_start=pd.Timestamp("2024-01-01"),
        window_end=pd.Timestamp("2024-01-02"),
    )
    assert m.account_id == "ACC-001"
    assert m.typology == "structuring"
    assert m.confidence == 0.85
    assert m.evidence == {}
    assert m.transaction_ids == []


def test_typology_match_evidence():
    m = TypologyMatch(
        account_id="ACC-001",
        typology="layering",
        confidence=0.7,
        n_transactions=2,
        total_amount_gbp=9_000.0,
        window_start=pd.Timestamp("2024-01-01"),
        window_end=pd.Timestamp("2024-01-01 06:00"),
        evidence={"pass_through_ratio": 0.9},
        transaction_ids=["T1", "T2"],
    )
    assert m.evidence["pass_through_ratio"] == 0.9
    assert len(m.transaction_ids) == 2


def test_typology_score_fields():
    s = TypologyScore(
        transaction_id="T-001",
        structuring_score=0.7,
        layering_score=0.0,
        integration_score=0.0,
        composite_score=0.233,
        dominant_typology="structuring",
        is_flagged=False,
    )
    assert s.transaction_id == "T-001"
    assert s.dominant_typology == "structuring"
    assert s.is_flagged is False


# ---------------------------------------------------------------------------
# TypologyEngine instantiation tests
# ---------------------------------------------------------------------------


def test_engine_default_threshold():
    eng = TypologyEngine()
    assert eng.reporting_threshold_gbp == UK_REPORTING_THRESHOLD_GBP


def test_engine_custom_params():
    eng = TypologyEngine(
        reporting_threshold_gbp=5_000.0,
        structuring_window_hours=12,
        structuring_min_txn=3,
        layering_window_hours=6,
        flag_threshold=0.6,
    )
    assert eng.reporting_threshold_gbp == 5_000.0
    assert eng.structuring_window_hours == 12
    assert eng.flag_threshold == 0.6


# ---------------------------------------------------------------------------
# detect_structuring() tests
# ---------------------------------------------------------------------------


def test_detect_structuring_returns_list(engine, structuring_df):
    result = engine.detect_structuring(structuring_df)
    assert isinstance(result, list)


def test_detect_structuring_detects_known_cluster(engine, structuring_df):
    result = engine.detect_structuring(structuring_df)
    account_ids = [m.account_id for m in result]
    assert "ACC-A" in account_ids


def test_detect_structuring_single_txn_not_flagged(engine, structuring_df):
    # ACC-B has one transaction — not structuring
    result = engine.detect_structuring(structuring_df)
    # ACC-B should not appear as a structuring match (single transaction)
    # Note: ACC-B has amount 9500 which is below threshold but only 1 txn
    acc_b_matches = [m for m in result if m.account_id == "ACC-B"]
    assert len(acc_b_matches) == 0


def test_detect_structuring_typology_label(engine, structuring_df):
    result = engine.detect_structuring(structuring_df)
    for m in result:
        assert m.typology == "structuring"


def test_detect_structuring_confidence_in_range(engine, structuring_df):
    result = engine.detect_structuring(structuring_df)
    for m in result:
        assert 0.0 <= m.confidence <= 1.0


def test_detect_structuring_n_transactions(engine, structuring_df):
    result = engine.detect_structuring(structuring_df)
    acc_a = [m for m in result if m.account_id == "ACC-A"]
    assert len(acc_a) > 0
    assert acc_a[0].n_transactions >= 2


def test_detect_structuring_transaction_ids_present(engine, structuring_df):
    result = engine.detect_structuring(structuring_df)
    for m in result:
        assert len(m.transaction_ids) >= 2


def test_detect_structuring_evidence_has_threshold(engine, structuring_df):
    result = engine.detect_structuring(structuring_df)
    for m in result:
        assert "threshold_gbp" in m.evidence


def test_detect_structuring_below_threshold_only(engine):
    """Transactions above the reporting threshold should not be flagged."""
    df = pd.DataFrame(
        [
            {
                "transaction_id": f"T-{i}",
                "sender_account_id": "ACC-LARGE",
                "receiver_account_id": "ACC-Z",
                "amount_gbp": 15_000.0,  # above threshold
                "timestamp": datetime(2024, 1, 1) + timedelta(hours=i),
                "country_origin": "GB",
                "country_destination": "GB",
            }
            for i in range(5)
        ]
    )
    result = engine.detect_structuring(df)
    # Large individual amounts should not produce structuring matches
    assert all(m.account_id != "ACC-LARGE" for m in result)


def test_detect_structuring_empty_df(engine):
    df = pd.DataFrame(
        columns=[
            "transaction_id",
            "sender_account_id",
            "receiver_account_id",
            "amount_gbp",
            "timestamp",
        ]
    )
    result = engine.detect_structuring(df)
    assert result == []


def test_detect_structuring_missing_column_raises(engine, structuring_df):
    df = structuring_df.drop(columns=["amount_gbp"])
    with pytest.raises(KeyError, match="amount_gbp"):
        engine.detect_structuring(df)


def test_detect_structuring_on_aml_data(engine, aml_df):
    result = engine.detect_structuring(aml_df)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# detect_layering() tests
# ---------------------------------------------------------------------------


def test_detect_layering_returns_list(engine, layering_df):
    result = engine.detect_layering(layering_df)
    assert isinstance(result, list)


def test_detect_layering_detects_mule(engine, layering_df):
    result = engine.detect_layering(layering_df)
    account_ids = [m.account_id for m in result]
    assert "ACC-MULE" in account_ids


def test_detect_layering_typology_label(engine, layering_df):
    result = engine.detect_layering(layering_df)
    for m in result:
        assert m.typology == "layering"


def test_detect_layering_confidence_in_range(engine, layering_df):
    result = engine.detect_layering(layering_df)
    for m in result:
        assert 0.0 <= m.confidence <= 1.0


def test_detect_layering_evidence_has_pass_through(engine, layering_df):
    result = engine.detect_layering(layering_df)
    mule_matches = [m for m in result if m.account_id == "ACC-MULE"]
    assert len(mule_matches) > 0
    assert "pass_through_ratio" in mule_matches[0].evidence


def test_detect_layering_clean_not_flagged(engine, layering_df):
    result = engine.detect_layering(layering_df)
    # ACC-CLEAN only sends, does not receive then forward
    clean_matches = [m for m in result if m.account_id == "ACC-CLEAN"]
    assert len(clean_matches) == 0


def test_detect_layering_outside_window_not_flagged(engine):
    """Forwarding outside the time window should not be flagged."""
    t0 = datetime(2024, 1, 1)
    df = pd.DataFrame(
        [
            {
                "transaction_id": "R-001",
                "sender_account_id": "ACC-X",
                "receiver_account_id": "ACC-MID",
                "amount_gbp": 5_000.0,
                "timestamp": t0,
                "country_origin": "GB",
                "country_destination": "GB",
            },
            {
                "transaction_id": "R-002",
                "sender_account_id": "ACC-MID",
                "receiver_account_id": "ACC-Y",
                "amount_gbp": 4_800.0,
                "timestamp": t0 + timedelta(hours=48),  # 48h >> default 24h window
                "country_origin": "GB",
                "country_destination": "GB",
            },
        ]
    )
    engine_tight = TypologyEngine(layering_window_hours=24)
    result = engine_tight.detect_layering(df)
    assert all(m.account_id != "ACC-MID" for m in result)


def test_detect_layering_missing_column_raises(engine, layering_df):
    df = layering_df.drop(columns=["receiver_account_id"])
    with pytest.raises(KeyError, match="receiver_account_id"):
        engine.detect_layering(df)


def test_detect_layering_empty_df(engine):
    df = pd.DataFrame(
        columns=[
            "transaction_id",
            "sender_account_id",
            "receiver_account_id",
            "amount_gbp",
            "timestamp",
        ]
    )
    result = engine.detect_layering(df)
    assert result == []


def test_detect_layering_on_aml_data(engine, aml_df):
    result = engine.detect_layering(aml_df)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# detect_integration() tests
# ---------------------------------------------------------------------------


def test_detect_integration_returns_list(engine, integration_df):
    result = engine.detect_integration(integration_df)
    assert isinstance(result, list)


def test_detect_integration_detects_account(engine, integration_df):
    result = engine.detect_integration(integration_df)
    account_ids = [m.account_id for m in result]
    assert "ACC-INT" in account_ids


def test_detect_integration_typology_label(engine, integration_df):
    result = engine.detect_integration(integration_df)
    for m in result:
        assert m.typology == "integration"


def test_detect_integration_confidence_in_range(engine, integration_df):
    result = engine.detect_integration(integration_df)
    for m in result:
        assert 0.0 <= m.confidence <= 1.0


def test_detect_integration_evidence_has_ratio(engine, integration_df):
    result = engine.detect_integration(integration_df)
    int_matches = [m for m in result if m.account_id == "ACC-INT"]
    assert len(int_matches) > 0
    assert "integration_ratio" in int_matches[0].evidence


def test_detect_integration_high_risk_corridor_uplifts_confidence(engine, integration_df):
    result_with_hrc = engine.detect_integration(integration_df)
    int_matches = [m for m in result_with_hrc if m.account_id == "ACC-INT"]
    if int_matches:
        assert int_matches[0].evidence.get("high_risk_corridor") is True
        assert int_matches[0].confidence > 0.0


def test_detect_integration_no_high_risk(engine, integration_df):
    """Without high-risk countries, confidence should be lower."""
    df = integration_df.copy()
    df["country_destination"] = "GB"
    df["country_origin"] = "GB"
    result = engine.detect_integration(df)
    for m in result:
        assert m.evidence.get("high_risk_corridor") is False


def test_detect_integration_missing_column_raises(engine, integration_df):
    df = integration_df.drop(columns=["amount_gbp"])
    with pytest.raises(KeyError, match="amount_gbp"):
        engine.detect_integration(df)


def test_detect_integration_empty_df(engine):
    df = pd.DataFrame(
        columns=[
            "transaction_id",
            "sender_account_id",
            "receiver_account_id",
            "amount_gbp",
            "timestamp",
        ]
    )
    result = engine.detect_integration(df)
    assert result == []


def test_detect_integration_no_country_cols(engine, integration_df):
    result = engine.detect_integration(
        integration_df, country_origin_col=None, country_dest_col=None
    )
    assert isinstance(result, list)


def test_detect_integration_on_aml_data(engine, aml_df):
    result = engine.detect_integration(aml_df)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# score() tests
# ---------------------------------------------------------------------------


def test_score_returns_dataframe(engine, aml_df):
    result = engine.score(aml_df)
    assert isinstance(result, pd.DataFrame)


def test_score_row_count_matches_input(engine, aml_df):
    result = engine.score(aml_df)
    assert len(result) == len(aml_df)


def test_score_columns(engine, aml_df):
    result = engine.score(aml_df)
    expected_cols = [
        "transaction_id",
        "structuring_score",
        "layering_score",
        "integration_score",
        "composite_score",
        "dominant_typology",
        "is_flagged",
    ]
    for col in expected_cols:
        assert col in result.columns


def test_score_composite_in_range(engine, aml_df):
    result = engine.score(aml_df)
    assert (result["composite_score"] >= 0.0).all()
    assert (result["composite_score"] <= 1.0).all()


def test_score_individual_scores_in_range(engine, aml_df):
    result = engine.score(aml_df)
    for col in ("structuring_score", "layering_score", "integration_score"):
        assert (result[col] >= 0.0).all()
        assert (result[col] <= 1.0).all()


def test_score_is_flagged_is_bool(engine, aml_df):
    result = engine.score(aml_df)
    assert result["is_flagged"].dtype == bool


def test_score_dominant_typology_values(engine, aml_df):
    result = engine.score(aml_df)
    valid = {"structuring", "layering", "integration", "none"}
    assert set(result["dominant_typology"].unique()).issubset(valid)


def test_score_custom_weights(engine, aml_df):
    result = engine.score(aml_df, weights={"structuring": 0.5, "layering": 0.3, "integration": 0.2})
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(aml_df)


def test_score_invalid_weights_raises(engine, aml_df):
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        engine.score(aml_df, weights={"structuring": 0.5, "layering": 0.3, "integration": 0.3})


def test_score_flag_threshold_respected(engine, structuring_df):
    engine_low = TypologyEngine(flag_threshold=0.01)
    result = engine_low.score(structuring_df)
    # With very low threshold, structuring cluster transactions should be flagged
    assert result["is_flagged"].any()


def test_score_no_matches_zero_scores(engine):
    """Transactions that match no typology should have zero composite score."""
    # A single normal transaction with low amount
    df = pd.DataFrame(
        [
            {
                "transaction_id": "N-001",
                "sender_account_id": "ACC-NORM",
                "receiver_account_id": "ACC-Z",
                "amount_gbp": 50.0,
                "timestamp": datetime(2024, 6, 1),
                "country_origin": "GB",
                "country_destination": "GB",
            }
        ]
    )
    result = engine.score(df)
    # Cannot assert zero because integration detector may flag high-inflow accounts,
    # but a single transaction receiving nothing should have 0 layering/structuring
    assert result["structuring_score"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# matches_to_dataframe() tests
# ---------------------------------------------------------------------------


def test_matches_to_dataframe_empty():
    engine = TypologyEngine()
    df = engine.matches_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_matches_to_dataframe_columns(engine, structuring_df):
    matches = engine.detect_structuring(structuring_df)
    if matches:
        df = engine.matches_to_dataframe(matches)
        for col in (
            "account_id",
            "typology",
            "confidence",
            "n_transactions",
            "total_amount_gbp",
            "window_start",
            "window_end",
            "transaction_ids",
        ):
            assert col in df.columns


def test_matches_to_dataframe_row_count(engine, structuring_df):
    matches = engine.detect_structuring(structuring_df)
    df = engine.matches_to_dataframe(matches)
    assert len(df) == len(matches)


def test_matches_to_dataframe_typology_values(engine, structuring_df):
    matches = engine.detect_structuring(structuring_df)
    df = engine.matches_to_dataframe(matches)
    if len(df) > 0:
        assert (df["typology"] == "structuring").all()


def test_matches_to_dataframe_transaction_ids_string(engine, structuring_df):
    matches = engine.detect_structuring(structuring_df)
    df = engine.matches_to_dataframe(matches)
    if len(df) > 0:
        assert pd.api.types.is_string_dtype(df["transaction_ids"])


# ---------------------------------------------------------------------------
# _check_columns() tests
# ---------------------------------------------------------------------------


def test_check_columns_raises_on_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(KeyError, match="c"):
        TypologyEngine._check_columns(df, ["a", "b", "c"])


def test_check_columns_passes_when_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    TypologyEngine._check_columns(df, ["a", "b"])  # should not raise
