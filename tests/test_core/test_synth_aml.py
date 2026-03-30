"""
tests/test_core/test_synth_aml.py
===================================
Unit tests for the synthetic AML transaction generator.
"""

import pandas as pd
import pytest

from fincrime_ml.core.data.synth_aml import (
    STRUCTURING_LOWER_GBP,
    STRUCTURING_UPPER_GBP,
    UK_REPORTING_THRESHOLD_GBP,
    AMLGeneratorConfig,
    SyntheticAMLGenerator,
)


@pytest.fixture
def generator():
    return SyntheticAMLGenerator(n_accounts=500, seed=42)


# ---------------------------------------------------------------------------
# generate() — mixed dataset
# ---------------------------------------------------------------------------


def test_generate_returns_correct_row_count(generator):
    df = generator.generate(n_transactions=1_000, suspicious_rate=0.05)
    assert len(df) == 1_000


def test_generate_schema_columns_present(generator):
    df = generator.generate(n_transactions=500)
    for col in SyntheticAMLGenerator.AML_SCHEMA_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_suspicious_rate_approximately_correct(generator):
    df = generator.generate(n_transactions=5_000, suspicious_rate=0.05)
    actual = df["is_suspicious"].mean()
    assert abs(actual - 0.05) < 0.01


def test_generate_label_is_binary(generator):
    df = generator.generate(n_transactions=1_000, suspicious_rate=0.05)
    assert set(df["is_suspicious"].unique()).issubset({0, 1})


def test_generate_amount_non_negative(generator):
    df = generator.generate(n_transactions=1_000)
    assert (df["amount_gbp"] >= 0).all()


def test_generate_typology_values_valid(generator):
    valid = {"structuring", "layering", "integration", "legitimate"}
    df = generator.generate(n_transactions=1_000, suspicious_rate=0.10)
    assert df["typology"].isin(valid).all()


def test_generate_reproducibility_same_seed():
    gen1 = SyntheticAMLGenerator(n_accounts=300, seed=77)
    gen2 = SyntheticAMLGenerator(n_accounts=300, seed=77)
    df1 = gen1.generate(n_transactions=500)
    df2 = gen2.generate(n_transactions=500)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds_differ():
    gen1 = SyntheticAMLGenerator(n_accounts=300, seed=1)
    gen2 = SyntheticAMLGenerator(n_accounts=300, seed=2)
    df1 = gen1.generate(n_transactions=500)
    df2 = gen2.generate(n_transactions=500)
    assert not df1["transaction_id"].equals(df2["transaction_id"])


# ---------------------------------------------------------------------------
# generate_mule_chains()
# ---------------------------------------------------------------------------


def test_mule_chains_schema(generator):
    df = generator.generate_mule_chains(n_chains=5)
    required = [
        "transaction_id",
        "sender_account_id",
        "receiver_account_id",
        "amount_gbp",
        "chain_id",
        "hop_number",
        "layering_depth",
        "is_suspicious",
        "typology",
        "rapid_movement_flag",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_mule_chains_all_suspicious(generator):
    df = generator.generate_mule_chains(n_chains=5)
    assert (df["is_suspicious"] == 1).all()


def test_mule_chains_typology_is_layering(generator):
    df = generator.generate_mule_chains(n_chains=5)
    assert (df["typology"] == "layering").all()


def test_mule_chains_hop_depth_within_config():
    config = AMLGeneratorConfig(
        n_accounts=500,
        mule_account_rate=0.05,
        chain_depth_min=2,
        chain_depth_max=4,
        seed=42,
    )
    gen = SyntheticAMLGenerator(config=config)
    df = gen.generate_mule_chains(n_chains=10)
    assert df["layering_depth"].between(2, 4).all()


def test_mule_chains_amount_positive(generator):
    df = generator.generate_mule_chains(n_chains=5)
    assert (df["amount_gbp"] > 0).all()


def test_mule_chains_chain_ids_present(generator):
    n_chains = 8
    df = generator.generate_mule_chains(n_chains=n_chains)
    assert df["chain_id"].nunique() == n_chains


def test_mule_chains_insufficient_mules_raises():
    """Raise ValueError when mule pool is too small for chain generation."""
    config = AMLGeneratorConfig(n_accounts=10, mule_account_rate=0.001, seed=42)
    gen = SyntheticAMLGenerator(config=config)
    # Force mule pool below minimum
    gen.accounts["is_mule_account"] = False
    with pytest.raises(ValueError, match="Insufficient mule accounts"):
        gen.generate_mule_chains(n_chains=3)


# ---------------------------------------------------------------------------
# generate_structuring_transactions()
# ---------------------------------------------------------------------------


def test_structuring_schema(generator):
    df = generator.generate_structuring_transactions(n_clusters=10)
    required = [
        "transaction_id",
        "sender_account_id",
        "receiver_account_id",
        "amount_gbp",
        "structuring_flag",
        "typology",
        "cluster_id",
        "is_suspicious",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_structuring_all_suspicious(generator):
    df = generator.generate_structuring_transactions(n_clusters=10)
    assert (df["is_suspicious"] == 1).all()


def test_structuring_flag_set(generator):
    df = generator.generate_structuring_transactions(n_clusters=10)
    assert df["structuring_flag"].all()


def test_structuring_amounts_below_threshold(generator):
    df = generator.generate_structuring_transactions(n_clusters=20)
    assert (df["amount_gbp"] < UK_REPORTING_THRESHOLD_GBP).all()


def test_structuring_amounts_within_smurfing_band(generator):
    df = generator.generate_structuring_transactions(n_clusters=20)
    assert (df["amount_gbp"] >= STRUCTURING_LOWER_GBP).all()
    assert (df["amount_gbp"] <= STRUCTURING_UPPER_GBP).all()


def test_structuring_typology_label(generator):
    df = generator.generate_structuring_transactions(n_clusters=10)
    assert (df["typology"] == "structuring").all()


def test_structuring_cluster_count(generator):
    n_clusters = 15
    df = generator.generate_structuring_transactions(n_clusters=n_clusters)
    assert df["cluster_id"].nunique() == n_clusters


def test_structuring_min_transactions_per_cluster(generator):
    """Each cluster must have at least 3 transactions (as per JMLSG red-flag pattern)."""
    df = generator.generate_structuring_transactions(n_clusters=10)
    cluster_sizes = df.groupby("cluster_id").size()
    assert (cluster_sizes >= 3).all()


# ---------------------------------------------------------------------------
# Account population
# ---------------------------------------------------------------------------


def test_mule_account_rate_respected():
    config = AMLGeneratorConfig(n_accounts=1_000, mule_account_rate=0.03, seed=42)
    gen = SyntheticAMLGenerator(config=config)
    mule_count = gen.accounts["is_mule_account"].sum()
    expected = int(1_000 * 0.03)
    assert abs(mule_count - expected) <= 1


def test_accounts_dataframe_schema(generator):
    required = ["account_id", "is_mule_account", "avg_monthly_balance", "account_age_days"]
    for col in required:
        assert col in generator.accounts.columns, f"Missing account column: {col}"


def test_account_ids_unique(generator):
    assert generator.accounts["account_id"].nunique() == len(generator.accounts)
