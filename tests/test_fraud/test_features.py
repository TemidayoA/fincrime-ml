"""
tests/test_fraud/test_features.py
==================================
Unit tests for fraud feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from fincrime_ml.fraud.features import (
    MCC_RISK_SCORE,
    FraudFeatureEngineer,
    compute_amount_deviation,
    compute_mcc_risk_features,
    compute_velocity_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_df() -> pd.DataFrame:
    """Minimal transaction DataFrame covering multiple accounts and timestamps."""
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:04d}" for i in range(10)],
            "account_id": [
                "ACC001",
                "ACC001",
                "ACC001",
                "ACC002",
                "ACC002",
                "ACC001",
                "ACC002",
                "ACC001",
                "ACC002",
                "ACC001",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:30:00",  # ACC001 txn 0
                    "2024-01-01 01:00:00",  # ACC001 txn 1 (30 min after txn 0)
                    "2024-01-01 02:00:00",  # ACC001 txn 2 (1h after txn 1)
                    "2024-01-01 00:00:00",  # ACC002 txn 3
                    "2024-01-01 12:00:00",  # ACC002 txn 4
                    "2024-01-02 00:00:00",  # ACC001 txn 5 (next day)
                    "2024-01-02 00:30:00",  # ACC002 txn 6
                    "2024-01-08 00:00:00",  # ACC001 txn 7 (7 days later)
                    "2024-01-08 00:00:00",  # ACC002 txn 8
                    "2024-01-01 01:30:00",  # ACC001 txn 9 (within 1h window)
                ]
            ),
            "amount_gbp": [50.0, 75.0, 200.0, 30.0, 100.0, 60.0, 40.0, 90.0, 55.0, 120.0],
            "account_avg_spend": [60.0] * 5 + [60.0] * 5,
            "account_spend_stddev": [20.0] * 10,
            "mcc_risk": [
                "low",
                "medium",
                "high",
                "low",
                "high",
                "medium",
                "low",
                "high",
                "medium",
                "low",
            ],
            "is_fraud": [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        }
    )


@pytest.fixture(scope="module")
def engineer() -> FraudFeatureEngineer:
    return FraudFeatureEngineer()


# ---------------------------------------------------------------------------
# transform() tests
# ---------------------------------------------------------------------------


def test_transform_returns_new_dataframe(engineer, base_df):
    out = engineer.transform(base_df)
    assert out is not base_df


def test_transform_does_not_mutate_input(engineer, base_df):
    original_cols = list(base_df.columns)
    engineer.transform(base_df)
    assert list(base_df.columns) == original_cols


def test_transform_preserves_row_count(engineer, base_df):
    out = engineer.transform(base_df)
    assert len(out) == len(base_df)


def test_transform_adds_expected_columns(engineer, base_df):
    out = engineer.transform(base_df)
    expected = [
        "velocity_count_1h",
        "velocity_amount_1h",
        "velocity_count_24h",
        "velocity_amount_24h",
        "velocity_count_168h",
        "velocity_amount_168h",
        "amount_zscore",
        "amount_over_avg_ratio",
        "mcc_risk_score",
        "is_high_risk_mcc",
    ]
    for col in expected:
        assert col in out.columns, f"Missing expected column: {col}"


def test_transform_raises_on_missing_column(engineer, base_df):
    bad = base_df.drop(columns=["mcc_risk"])
    with pytest.raises(ValueError, match="missing required columns"):
        engineer.transform(bad)


# ---------------------------------------------------------------------------
# Velocity feature tests
# ---------------------------------------------------------------------------


def test_velocity_count_1h_first_transaction_is_zero(engineer, base_df):
    """First transaction per account has no prior transactions in any window."""
    out = engineer.add_velocity_features(base_df)
    # ACC002's earliest transaction (txn 3 at 00:00) should have count=0
    acc002_sorted = out[out["account_id"] == "ACC002"].sort_values("timestamp")
    assert acc002_sorted.iloc[0]["velocity_count_1h"] == 0


def test_velocity_count_is_non_negative(engineer, base_df):
    out = engineer.add_velocity_features(base_df)
    assert (out["velocity_count_1h"] >= 0).all()
    assert (out["velocity_count_24h"] >= 0).all()
    assert (out["velocity_count_168h"] >= 0).all()


def test_velocity_count_1h_less_than_or_equal_24h(engineer, base_df):
    """1h velocity count cannot exceed 24h count for the same transaction."""
    out = engineer.add_velocity_features(base_df)
    assert (out["velocity_count_1h"] <= out["velocity_count_24h"]).all()


def test_velocity_count_24h_less_than_or_equal_168h(engineer, base_df):
    out = engineer.add_velocity_features(base_df)
    assert (out["velocity_count_24h"] <= out["velocity_count_168h"]).all()


def test_velocity_amount_non_negative(engineer, base_df):
    out = engineer.add_velocity_features(base_df)
    assert (out["velocity_amount_1h"] >= 0).all()
    assert (out["velocity_amount_24h"] >= 0).all()
    assert (out["velocity_amount_168h"] >= 0).all()


def test_velocity_excludes_current_transaction(engineer):
    """The current transaction must not be counted in its own window."""
    df = pd.DataFrame(
        {
            "account_id": ["ACC001", "ACC001"],
            "timestamp": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:30:00"]),
            "amount_gbp": [100.0, 200.0],
        }
    )
    out = engineer.add_velocity_features(df)
    # First txn: no prior txns in window
    first = out.sort_values("timestamp").iloc[0]
    assert first["velocity_count_1h"] == 0
    assert first["velocity_amount_1h"] == 0.0


def test_velocity_count_within_1h_window(engineer):
    """Transactions within 1h window should be counted; older ones should not."""
    df = pd.DataFrame(
        {
            "account_id": ["ACC001"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 09:00:00",  # outside 1h window of txn at 10:30
                    "2024-01-01 10:00:00",  # inside 1h window
                    "2024-01-01 10:30:00",  # current
                ]
            ),
            "amount_gbp": [50.0, 75.0, 200.0],
        }
    )
    out = engineer.add_velocity_features(df)
    last = out.sort_values("timestamp").iloc[2]
    # Only txn at 10:00 is within 1h of 10:30; txn at 09:00 is 1.5h before
    assert last["velocity_count_1h"] == 1
    assert last["velocity_amount_1h"] == 75.0


def test_velocity_cross_account_isolation(engineer):
    """Velocity windows must not bleed across different accounts."""
    df = pd.DataFrame(
        {
            "account_id": ["ACC001", "ACC002", "ACC001"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:10:00",
                    "2024-01-01 10:30:00",
                ]
            ),
            "amount_gbp": [100.0, 200.0, 50.0],
        }
    )
    out = engineer.add_velocity_features(df)
    # ACC001 at 10:30 should count only ACC001 txn at 10:00
    acc001_last = out[(out["account_id"] == "ACC001")].sort_values("timestamp").iloc[1]
    assert acc001_last["velocity_count_1h"] == 1
    assert acc001_last["velocity_amount_1h"] == 100.0


def test_velocity_raises_on_missing_column(engineer, base_df):
    with pytest.raises(ValueError, match="missing columns"):
        engineer.add_velocity_features(base_df.drop(columns=["amount_gbp"]))


def test_compute_velocity_features_functional_wrapper(base_df):
    out = compute_velocity_features(base_df)
    assert "velocity_count_1h" in out.columns


def test_velocity_custom_windows(base_df):
    engineer = FraudFeatureEngineer(velocity_windows=(2, 48))
    out = engineer.add_velocity_features(base_df)
    assert "velocity_count_2h" in out.columns
    assert "velocity_count_48h" in out.columns
    assert "velocity_count_1h" not in out.columns


# ---------------------------------------------------------------------------
# Amount deviation tests
# ---------------------------------------------------------------------------


def test_amount_zscore_column_present(engineer, base_df):
    out = engineer.add_amount_deviation_features(base_df)
    assert "amount_zscore" in out.columns


def test_amount_over_avg_ratio_column_present(engineer, base_df):
    out = engineer.add_amount_deviation_features(base_df)
    assert "amount_over_avg_ratio" in out.columns


def test_amount_zscore_at_mean_is_zero(engineer):
    df = pd.DataFrame(
        {
            "amount_gbp": [60.0],
            "account_avg_spend": [60.0],
            "account_spend_stddev": [20.0],
        }
    )
    out = engineer.add_amount_deviation_features(df)
    assert out["amount_zscore"].iloc[0] == pytest.approx(0.0)


def test_amount_zscore_correct_value(engineer):
    df = pd.DataFrame(
        {
            "amount_gbp": [100.0],
            "account_avg_spend": [60.0],
            "account_spend_stddev": [20.0],
        }
    )
    out = engineer.add_amount_deviation_features(df)
    # (100 - 60) / 20 = 2.0
    assert out["amount_zscore"].iloc[0] == pytest.approx(2.0)


def test_amount_zscore_capped_at_10(engineer):
    df = pd.DataFrame(
        {
            "amount_gbp": [10_000.0],
            "account_avg_spend": [50.0],
            "account_spend_stddev": [1.0],
        }
    )
    out = engineer.add_amount_deviation_features(df)
    assert out["amount_zscore"].iloc[0] == pytest.approx(10.0)


def test_amount_zscore_capped_at_negative_10(engineer):
    df = pd.DataFrame(
        {
            "amount_gbp": [0.01],
            "account_avg_spend": [5000.0],
            "account_spend_stddev": [1.0],
        }
    )
    out = engineer.add_amount_deviation_features(df)
    assert out["amount_zscore"].iloc[0] == pytest.approx(-10.0)


def test_amount_zscore_zero_std_no_error(engineer):
    """Zero standard deviation must not raise; _MIN_STD prevents division by zero."""
    df = pd.DataFrame(
        {
            "amount_gbp": [60.0],
            "account_avg_spend": [60.0],
            "account_spend_stddev": [0.0],
        }
    )
    out = engineer.add_amount_deviation_features(df)
    assert np.isfinite(out["amount_zscore"].iloc[0])


def test_amount_over_avg_ratio_correct_value(engineer):
    df = pd.DataFrame(
        {
            "amount_gbp": [120.0],
            "account_avg_spend": [60.0],
            "account_spend_stddev": [20.0],
        }
    )
    out = engineer.add_amount_deviation_features(df)
    assert out["amount_over_avg_ratio"].iloc[0] == pytest.approx(2.0)


def test_amount_deviation_raises_on_missing_column(engineer, base_df):
    with pytest.raises(ValueError, match="missing columns"):
        engineer.add_amount_deviation_features(base_df.drop(columns=["account_avg_spend"]))


def test_compute_amount_deviation_functional_wrapper(base_df):
    out = compute_amount_deviation(base_df)
    assert "amount_zscore" in out.columns


# ---------------------------------------------------------------------------
# MCC risk feature tests
# ---------------------------------------------------------------------------


def test_mcc_risk_score_values(engineer, base_df):
    out = engineer.add_mcc_risk_features(base_df)
    assert set(out["mcc_risk_score"].unique()).issubset({0, 1, 2, 3})


def test_mcc_risk_score_low_is_1(engineer):
    df = pd.DataFrame({"mcc_risk": ["low"]})
    out = engineer.add_mcc_risk_features(df)
    assert out["mcc_risk_score"].iloc[0] == 1


def test_mcc_risk_score_medium_is_2(engineer):
    df = pd.DataFrame({"mcc_risk": ["medium"]})
    out = engineer.add_mcc_risk_features(df)
    assert out["mcc_risk_score"].iloc[0] == 2


def test_mcc_risk_score_high_is_3(engineer):
    df = pd.DataFrame({"mcc_risk": ["high"]})
    out = engineer.add_mcc_risk_features(df)
    assert out["mcc_risk_score"].iloc[0] == 3


def test_mcc_risk_score_unknown_is_0(engineer):
    df = pd.DataFrame({"mcc_risk": ["unknown"]})
    out = engineer.add_mcc_risk_features(df)
    assert out["mcc_risk_score"].iloc[0] == 0


def test_is_high_risk_mcc_high_rows(engineer, base_df):
    out = engineer.add_mcc_risk_features(base_df)
    high_mask = base_df["mcc_risk"] == "high"
    assert (out.loc[high_mask, "is_high_risk_mcc"] == 1).all()


def test_is_high_risk_mcc_non_high_rows(engineer, base_df):
    out = engineer.add_mcc_risk_features(base_df)
    non_high_mask = base_df["mcc_risk"] != "high"
    assert (out.loc[non_high_mask, "is_high_risk_mcc"] == 0).all()


def test_mcc_risk_score_map_complete():
    """All expected risk tiers must be in MCC_RISK_SCORE."""
    for tier in ("low", "medium", "high", "unknown"):
        assert tier in MCC_RISK_SCORE


def test_mcc_risk_score_ordinal_ordering():
    """Risk score must be monotonically increasing: low < medium < high."""
    assert MCC_RISK_SCORE["low"] < MCC_RISK_SCORE["medium"] < MCC_RISK_SCORE["high"]


def test_mcc_risk_raises_on_missing_column(engineer):
    with pytest.raises(ValueError, match="missing column 'mcc_risk'"):
        engineer.add_mcc_risk_features(pd.DataFrame({"amount_gbp": [1.0]}))


def test_compute_mcc_risk_features_functional_wrapper(base_df):
    out = compute_mcc_risk_features(base_df)
    assert "mcc_risk_score" in out.columns
    assert "is_high_risk_mcc" in out.columns


# ---------------------------------------------------------------------------
# Integration: full transform with synthetic generator output
# ---------------------------------------------------------------------------


def test_transform_with_synth_generator_output():
    """Full pipeline: synthetic data -> feature engineering -> no errors."""
    from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator

    gen = SyntheticTransactionGenerator(n_accounts=50, seed=0)
    df = gen.generate(n_transactions=200, fraud_rate=0.05)

    engineer = FraudFeatureEngineer()
    out = engineer.transform(df)

    assert len(out) == len(df)
    assert "velocity_count_1h" in out.columns
    assert "amount_zscore" in out.columns
    assert "mcc_risk_score" in out.columns
    assert out["amount_zscore"].between(-10, 10).all()
    assert out["is_high_risk_mcc"].isin([0, 1]).all()
