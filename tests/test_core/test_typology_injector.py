"""
tests/test_core/test_typology_injector.py
==========================================
Unit tests for the fraud typology injector.
"""

import pandas as pd
import pytest

from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator
from fincrime_ml.core.data.typology_injector import _REQUIRED_COLS, TypologyInjector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    """Generate a purely legitimate transaction DataFrame for injection tests."""
    gen = SyntheticTransactionGenerator(n_accounts=500, seed=42)
    df = gen.generate(n_transactions=2_000, fraud_rate=0.0)
    # Ensure all records start as legitimate
    assert df["is_fraud"].sum() == 0
    return df


@pytest.fixture
def injector() -> TypologyInjector:
    return TypologyInjector(seed=42)


# ---------------------------------------------------------------------------
# Schema / validation tests
# ---------------------------------------------------------------------------


def test_required_columns_present_in_synth_output(clean_df):
    """SyntheticTransactionGenerator output must satisfy injector requirements."""
    for col in _REQUIRED_COLS:
        assert col in clean_df.columns, f"Missing required column: {col}"


def test_inject_raises_on_missing_column(injector, clean_df):
    df_bad = clean_df.drop(columns=["channel"])
    with pytest.raises(ValueError, match="missing required columns"):
        injector.inject_cnp(df_bad, n_frauds=10)


def test_inject_raises_on_zero_n_frauds(injector, clean_df):
    with pytest.raises(ValueError, match="positive integer"):
        injector.inject_cnp(clean_df, n_frauds=0)


def test_inject_raises_when_n_frauds_exceeds_legit_rows(injector, clean_df):
    with pytest.raises(ValueError, match="legitimate rows available"):
        injector.inject_cnp(clean_df, n_frauds=len(clean_df) + 1)


def test_inject_does_not_mutate_original(injector, clean_df):
    original_fraud_sum = clean_df["is_fraud"].sum()
    injector.inject_cnp(clean_df, n_frauds=50)
    assert clean_df["is_fraud"].sum() == original_fraud_sum


# ---------------------------------------------------------------------------
# CNP typology tests
# ---------------------------------------------------------------------------


def test_inject_cnp_sets_fraud_label(injector, clean_df):
    out = injector.inject_cnp(clean_df, n_frauds=100)
    assert out["is_fraud"].sum() == 100


def test_inject_cnp_channel_is_cnp(injector, clean_df):
    out = injector.inject_cnp(clean_df, n_frauds=100)
    fraud_rows = out[out["is_fraud"] == 1]
    assert fraud_rows["channel"].isin(["CNP_ECOM", "CNP_MOTO"]).all()


def test_inject_cnp_high_risk_mcc(injector, clean_df):
    out = injector.inject_cnp(clean_df, n_frauds=100)
    fraud_rows = out[out["is_fraud"] == 1]
    assert (fraud_rows["mcc_risk"] == "high").all()


def test_inject_cnp_amount_higher_than_original(injector, clean_df):
    """CNP fraud should inflate amounts relative to original legitimate rows."""
    orig_fraud_idx = injector.inject_cnp(clean_df, n_frauds=200)[lambda d: d["is_fraud"] == 1].index
    fraud_avg = injector.inject_cnp(clean_df, n_frauds=200).loc[orig_fraud_idx, "amount_gbp"].mean()
    legit_avg = clean_df["amount_gbp"].mean()
    assert fraud_avg > legit_avg


def test_inject_cnp_off_hours_bias(injector, clean_df):
    out = injector.inject_cnp(clean_df, n_frauds=200)
    fraud_rows = out[out["is_fraud"] == 1]
    assert (fraud_rows["hour_of_day"] < 7).all()


def test_inject_cnp_preserves_row_count(injector, clean_df):
    out = injector.inject_cnp(clean_df, n_frauds=100)
    assert len(out) == len(clean_df)


# ---------------------------------------------------------------------------
# ATO typology tests
# ---------------------------------------------------------------------------


def test_inject_ato_sets_fraud_label(injector, clean_df):
    out = injector.inject_ato(clean_df, n_frauds=80)
    assert out["is_fraud"].sum() == 80


def test_inject_ato_channel_is_ato_type(injector, clean_df):
    out = injector.inject_ato(clean_df, n_frauds=80)
    fraud_rows = out[out["is_fraud"] == 1]
    assert fraud_rows["channel"].isin(["MOBILE_APP", "CNP_ECOM"]).all()


def test_inject_ato_night_hours(injector, clean_df):
    out = injector.inject_ato(clean_df, n_frauds=200)
    fraud_rows = out[out["is_fraud"] == 1]
    assert (fraud_rows["hour_of_day"] < 6).all()


def test_inject_ato_high_risk_origin_country(injector, clean_df):
    from fincrime_ml.core.data.typology_injector import _HIGH_RISK_IP_COUNTRIES

    out = injector.inject_ato(clean_df, n_frauds=100)
    fraud_rows = out[out["is_fraud"] == 1]
    assert fraud_rows["country_origin"].isin(_HIGH_RISK_IP_COUNTRIES).all()


def test_inject_ato_amount_inflated(injector, clean_df):
    legit_avg = clean_df["amount_gbp"].mean()
    out = injector.inject_ato(clean_df, n_frauds=200)
    fraud_avg = out[out["is_fraud"] == 1]["amount_gbp"].mean()
    assert fraud_avg > legit_avg


# ---------------------------------------------------------------------------
# Bust-out typology tests
# ---------------------------------------------------------------------------


def test_inject_bust_out_sets_fraud_label(injector, clean_df):
    out = injector.inject_bust_out(clean_df, n_frauds=60)
    assert out["is_fraud"].sum() == 60


def test_inject_bust_out_high_risk_mcc(injector, clean_df):
    out = injector.inject_bust_out(clean_df, n_frauds=60)
    fraud_rows = out[out["is_fraud"] == 1]
    assert (fraud_rows["mcc_risk"] == "high").all()


def test_inject_bust_out_domestic_transactions(injector, clean_df):
    out = injector.inject_bust_out(clean_df, n_frauds=60)
    fraud_rows = out[out["is_fraud"] == 1]
    assert (fraud_rows["country_origin"] == "GB").all()
    assert (fraud_rows["country_destination"] == "GB").all()


def test_inject_bust_out_high_amounts(injector, clean_df):
    out = injector.inject_bust_out(clean_df, n_frauds=200)
    fraud_avg = out[out["is_fraud"] == 1]["amount_gbp"].mean()
    legit_avg = clean_df["amount_gbp"].mean()
    assert fraud_avg > legit_avg


# ---------------------------------------------------------------------------
# Card skimming typology tests
# ---------------------------------------------------------------------------


def test_inject_card_skimming_sets_fraud_label(injector, clean_df):
    out = injector.inject_card_skimming(clean_df, n_frauds=120)
    assert out["is_fraud"].sum() == 120


def test_inject_card_skimming_pos_channel(injector, clean_df):
    out = injector.inject_card_skimming(clean_df, n_frauds=120)
    fraud_rows = out[out["is_fraud"] == 1]
    assert (fraud_rows["channel"] == "POS").all()


def test_inject_card_skimming_geographic_displacement(injector, clean_df):
    """Skimming fraud should use cloned card in a different country."""
    displacement_countries = {"AE", "US", "DE", "HK", "OTHER"}
    out = injector.inject_card_skimming(clean_df, n_frauds=120)
    fraud_rows = out[out["is_fraud"] == 1]
    assert fraud_rows["country_destination"].isin(displacement_countries).all()


def test_inject_card_skimming_bimodal_amounts(injector, clean_df):
    """Skimming produces a mix of small testing and larger extractive transactions."""
    out = injector.inject_card_skimming(clean_df, n_frauds=200)
    fraud_amounts = out[out["is_fraud"] == 1]["amount_gbp"]
    # Some amounts should be small (testing phase)
    assert (fraud_amounts < 25.0).any()
    # Some amounts should be larger (extraction phase)
    assert (fraud_amounts > 50.0).any()


# ---------------------------------------------------------------------------
# inject_all tests
# ---------------------------------------------------------------------------


def test_inject_all_fraud_rate_approximate(injector, clean_df):
    out = injector.inject_all(clean_df, fraud_rate=0.05)
    actual = out["is_fraud"].mean()
    assert abs(actual - 0.05) < 0.005


def test_inject_all_preserves_row_count(injector, clean_df):
    out = injector.inject_all(clean_df, fraud_rate=0.05)
    assert len(out) == len(clean_df)


def test_inject_all_default_mix_produces_fraud(injector, clean_df):
    out = injector.inject_all(clean_df, fraud_rate=0.03)
    assert out["is_fraud"].sum() > 0


def test_inject_all_custom_mix(injector, clean_df):
    custom_mix = {"cnp": 0.50, "ato": 0.25, "bust_out": 0.15, "card_skimming": 0.10}
    out = injector.inject_all(clean_df, fraud_rate=0.04, typology_mix=custom_mix)
    actual = out["is_fraud"].mean()
    assert abs(actual - 0.04) < 0.005


def test_inject_all_raises_on_bad_fraud_rate(injector, clean_df):
    with pytest.raises(ValueError, match="fraud_rate must be in"):
        injector.inject_all(clean_df, fraud_rate=1.5)


def test_inject_all_raises_on_bad_mix_sum(injector, clean_df):
    bad_mix = {"cnp": 0.60, "ato": 0.20, "bust_out": 0.10, "card_skimming": 0.05}
    with pytest.raises(ValueError, match="must sum to 1.0"):
        injector.inject_all(clean_df, fraud_rate=0.01, typology_mix=bad_mix)


def test_inject_all_reproducible_with_same_seed(clean_df):
    inj1 = TypologyInjector(seed=7)
    inj2 = TypologyInjector(seed=7)
    out1 = inj1.inject_all(clean_df, fraud_rate=0.03)
    out2 = inj2.inject_all(clean_df, fraud_rate=0.03)
    pd.testing.assert_series_equal(out1["is_fraud"], out2["is_fraud"])


def test_inject_all_different_seeds_differ(clean_df):
    inj1 = TypologyInjector(seed=1)
    inj2 = TypologyInjector(seed=2)
    out1 = inj1.inject_all(clean_df, fraud_rate=0.03)
    out2 = inj2.inject_all(clean_df, fraud_rate=0.03)
    # Different seeds should mark different rows as fraud
    assert not out1["is_fraud"].equals(out2["is_fraud"])
