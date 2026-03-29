"""
tests/test_core/test_synth_cards.py
====================================
Unit tests for the synthetic transaction generator.
"""

import pytest
import pandas as pd
from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator, GeneratorConfig


@pytest.fixture
def generator():
    return SyntheticTransactionGenerator(n_accounts=500, seed=42)


def test_generate_returns_correct_shape(generator):
    df = generator.generate(n_transactions=1_000, fraud_rate=0.02)
    assert len(df) == 1_000


def test_generate_schema_columns_present(generator):
    df = generator.generate(n_transactions=500)
    for col in SyntheticTransactionGenerator.SCHEMA_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_fraud_rate_approximately_correct(generator):
    df = generator.generate(n_transactions=5_000, fraud_rate=0.02)
    actual = df["is_fraud"].mean()
    # Allow ±0.5% tolerance
    assert abs(actual - 0.02) < 0.005


def test_transaction_ids_unique(generator):
    df = generator.generate(n_transactions=1_000)
    assert df["transaction_id"].nunique() == len(df)


def test_amount_gbp_non_negative(generator):
    df = generator.generate(n_transactions=1_000)
    assert (df["amount_gbp"] >= 0).all()


def test_reproducibility_with_same_seed():
    gen1 = SyntheticTransactionGenerator(n_accounts=200, seed=99)
    gen2 = SyntheticTransactionGenerator(n_accounts=200, seed=99)
    df1 = gen1.generate(n_transactions=500)
    df2 = gen2.generate(n_transactions=500)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    gen1 = SyntheticTransactionGenerator(n_accounts=200, seed=1)
    gen2 = SyntheticTransactionGenerator(n_accounts=200, seed=2)
    df1 = gen1.generate(n_transactions=500)
    df2 = gen2.generate(n_transactions=500)
    # Transaction IDs should differ
    assert not df1["transaction_id"].equals(df2["transaction_id"])


def test_wire_transfers_schema(generator):
    df = generator.generate_wire_transfers(n=200)
    required = ["transfer_id", "sender_bic", "receiver_bic", "sender_iban",
                "receiver_iban", "amount_gbp", "country_origin", "country_destination",
                "is_structured_amount"]
    for col in required:
        assert col in df.columns


def test_mule_account_rate_respected():
    config = GeneratorConfig(n_accounts=2_000, mule_account_rate=0.01, seed=42)
    gen = SyntheticTransactionGenerator(config=config)
    mule_count = gen.accounts["is_mule_account"].sum()
    expected = int(2_000 * 0.01)
    # Allow ±1 for rounding
    assert abs(mule_count - expected) <= 1


def test_fraud_transactions_higher_average_amount(generator):
    df = generator.generate(n_transactions=5_000, fraud_rate=0.05)
    fraud_avg = df[df["is_fraud"] == 1]["amount_gbp"].mean()
    legit_avg = df[df["is_fraud"] == 0]["amount_gbp"].mean()
    # Fraud amounts should be meaningfully higher on average
    assert fraud_avg > legit_avg
