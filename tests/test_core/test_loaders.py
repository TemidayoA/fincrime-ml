"""
tests/test_core/test_loaders.py
================================
Unit tests for the dataset loaders and schema harmonisers.

All tests use in-memory fixtures that mimic the IEEE-CIS raw CSV schema.
No actual dataset files are required to run this suite.
"""


import pandas as pd
import pytest

from fincrime_ml.core.data.loaders import (
    _PRODUCT_CD_CHANNEL_MAP,
    _SENTINEL_MCC,
    _SENTINEL_MCC_NAME,
    _SENTINEL_MCC_RISK,
    _SENTINEL_MERCHANT,
    IEEE_CIS_HARMONISED_COLS,
    IeeeCisLoader,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal IEEE-CIS-format DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_transaction_df() -> pd.DataFrame:
    """Minimal transaction DataFrame matching IEEE-CIS train_transaction.csv schema."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [0, 1, 0, 0, 1],
            "TransactionDT": [86400, 172800, 43200, 259200, 345600],
            "TransactionAmt": [50.0, 250.0, 30.0, 100.0, 500.0],
            "ProductCD": ["W", "H", "C", "S", "R"],
            "card1": [1234.0, 5678.0, 1234.0, 9999.0, 5678.0],
            "card4": ["visa", "mastercard", "visa", "visa", "mastercard"],
            "addr1": [315.0, None, 204.0, 441.0, None],
            "addr2": [87.0, 96.0, 87.0, 87.0, 60.0],
            "P_emaildomain": ["gmail.com", "yahoo.com", None, "hotmail.com", "gmail.com"],
            "R_emaildomain": ["gmail.com", None, "gmail.com", "hotmail.com", "protonmail.com"],
        }
    )


@pytest.fixture(scope="module")
def raw_identity_df() -> pd.DataFrame:
    """Minimal identity DataFrame matching IEEE-CIS train_identity.csv schema."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 4],
            "DeviceType": ["desktop", "mobile", "desktop"],
            "DeviceInfo": ["Chrome/74.0", "iOS/11.0", "Firefox/67.0"],
            "id_01": [-5.0, -10.0, -3.0],
        }
    )


@pytest.fixture(scope="module")
def loader() -> IeeeCisLoader:
    return IeeeCisLoader()


@pytest.fixture(scope="module")
def harmonised_df(loader, raw_transaction_df, raw_identity_df) -> pd.DataFrame:
    """Harmonised output with identity join applied."""
    return loader.load_from_dataframes(raw_transaction_df, raw_identity_df)


@pytest.fixture(scope="module")
def harmonised_no_identity(loader, raw_transaction_df) -> pd.DataFrame:
    """Harmonised output without identity join."""
    return loader.load_from_dataframes(raw_transaction_df, identity_df=None)


# ---------------------------------------------------------------------------
# Schema / output column tests
# ---------------------------------------------------------------------------


def test_output_has_all_harmonised_cols(harmonised_df):
    """Output DataFrame must include every column in IEEE_CIS_HARMONISED_COLS."""
    for col in IEEE_CIS_HARMONISED_COLS:
        assert col in harmonised_df.columns, f"Missing column: {col}"


def test_output_row_count_matches_input(harmonised_df, raw_transaction_df):
    assert len(harmonised_df) == len(raw_transaction_df)


def test_output_column_order(harmonised_df):
    assert list(harmonised_df.columns) == IEEE_CIS_HARMONISED_COLS


# ---------------------------------------------------------------------------
# Identifier mapping tests
# ---------------------------------------------------------------------------


def test_transaction_id_prefixed(harmonised_df):
    assert harmonised_df["transaction_id"].str.startswith("TXN-").all()


def test_transaction_id_contains_original_id(harmonised_df, raw_transaction_df):
    for txn_id in raw_transaction_df["TransactionID"].astype(str):
        assert harmonised_df["transaction_id"].str.contains(txn_id).any()


def test_account_id_format(harmonised_df, raw_transaction_df):
    """account_id must be formatted as ACC + 7-digit zero-padded card1 integer."""
    expected_acc = f"ACC{1234:07d}"
    assert (harmonised_df["account_id"] == expected_acc).any()


def test_account_id_consistent_per_card(harmonised_df):
    """Same card1 value must produce the same account_id."""
    subset = harmonised_df[harmonised_df["account_id"] == f"ACC{1234:07d}"]
    # TransactionID 1 and 3 both have card1=1234 — should get the same account_id
    assert len(subset) == 2


def test_merchant_id_sentinel(harmonised_df):
    assert (harmonised_df["merchant_id"] == _SENTINEL_MERCHANT).all()


# ---------------------------------------------------------------------------
# Channel mapping tests
# ---------------------------------------------------------------------------


def test_product_cd_w_maps_to_cnp_ecom(harmonised_df, raw_transaction_df):
    w_rows = raw_transaction_df[raw_transaction_df["ProductCD"] == "W"].index
    assert (harmonised_df.loc[w_rows, "channel"] == "CNP_ECOM").all()


def test_product_cd_h_maps_to_cnp_ecom(harmonised_df, raw_transaction_df):
    h_rows = raw_transaction_df[raw_transaction_df["ProductCD"] == "H"].index
    assert (harmonised_df.loc[h_rows, "channel"] == "CNP_ECOM").all()


def test_product_cd_c_maps_to_pos(harmonised_df, raw_transaction_df):
    c_rows = raw_transaction_df[raw_transaction_df["ProductCD"] == "C"].index
    assert (harmonised_df.loc[c_rows, "channel"] == "POS").all()


def test_product_cd_s_maps_to_cnp_moto(harmonised_df, raw_transaction_df):
    s_rows = raw_transaction_df[raw_transaction_df["ProductCD"] == "S"].index
    assert (harmonised_df.loc[s_rows, "channel"] == "CNP_MOTO").all()


def test_product_cd_r_maps_to_pos(harmonised_df, raw_transaction_df):
    r_rows = raw_transaction_df[raw_transaction_df["ProductCD"] == "R"].index
    assert (harmonised_df.loc[r_rows, "channel"] == "POS").all()


def test_all_product_cds_mapped():
    """Every key in _PRODUCT_CD_CHANNEL_MAP must produce a known channel value."""
    known_channels = {"CNP_ECOM", "CNP_MOTO", "POS", "WIRE", "MOBILE_APP"}
    for product_cd, channel in _PRODUCT_CD_CHANNEL_MAP.items():
        assert (
            channel in known_channels
        ), f"ProductCD {product_cd!r} maps to unknown channel {channel!r}"


# ---------------------------------------------------------------------------
# Amount and currency tests
# ---------------------------------------------------------------------------


def test_amount_gbp_values_preserved(harmonised_df, raw_transaction_df):
    pd.testing.assert_series_equal(
        harmonised_df["amount_gbp"].reset_index(drop=True),
        raw_transaction_df["TransactionAmt"].round(2).reset_index(drop=True),
        check_names=False,
    )


def test_currency_is_usd_sentinel(harmonised_df):
    assert (harmonised_df["currency"] == "USD").all()


# ---------------------------------------------------------------------------
# Temporal feature tests
# ---------------------------------------------------------------------------


def test_hour_of_day_range(harmonised_df):
    assert harmonised_df["hour_of_day"].between(0, 23).all()


def test_day_of_week_range(harmonised_df):
    assert harmonised_df["day_of_week"].between(0, 6).all()


def test_hour_of_day_from_transaction_dt(harmonised_df, raw_transaction_df):
    """hour_of_day must equal (TransactionDT % 86400) // 3600."""
    expected = (raw_transaction_df["TransactionDT"] % 86400) // 3600
    pd.testing.assert_series_equal(
        harmonised_df["hour_of_day"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


def test_is_weekend_consistent_with_day_of_week(harmonised_df):
    weekend_mask = harmonised_df["day_of_week"].isin([5, 6])
    assert (harmonised_df.loc[weekend_mask, "is_weekend"] == 1).all()
    assert (harmonised_df.loc[~weekend_mask, "is_weekend"] == 0).all()


def test_transaction_dt_raw_preserved(harmonised_df, raw_transaction_df):
    pd.testing.assert_series_equal(
        harmonised_df["transaction_dt_raw"].reset_index(drop=True),
        raw_transaction_df["TransactionDT"].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Fraud label tests
# ---------------------------------------------------------------------------


def test_is_fraud_preserved(harmonised_df, raw_transaction_df):
    pd.testing.assert_series_equal(
        harmonised_df["is_fraud"].reset_index(drop=True),
        raw_transaction_df["isFraud"].astype(int).reset_index(drop=True),
        check_names=False,
    )


def test_fraud_count_matches(harmonised_df, raw_transaction_df):
    assert harmonised_df["is_fraud"].sum() == raw_transaction_df["isFraud"].sum()


# ---------------------------------------------------------------------------
# MCC sentinel tests
# ---------------------------------------------------------------------------


def test_mcc_is_sentinel(harmonised_df):
    assert (harmonised_df["mcc"] == _SENTINEL_MCC).all()


def test_mcc_name_is_sentinel(harmonised_df):
    assert (harmonised_df["mcc_name"] == _SENTINEL_MCC_NAME).all()


def test_mcc_risk_is_unknown_sentinel(harmonised_df):
    """mcc_risk must be 'unknown' — distinct from low/medium/high — so
    downstream rules apply conservative treatment (JMLSG Ch.5)."""
    assert (harmonised_df["mcc_risk"] == _SENTINEL_MCC_RISK).all()


# ---------------------------------------------------------------------------
# Email domain tests
# ---------------------------------------------------------------------------


def test_email_domain_payer_mapped(harmonised_df, raw_transaction_df):
    pd.testing.assert_series_equal(
        harmonised_df["email_domain_payer"].reset_index(drop=True),
        raw_transaction_df["P_emaildomain"].reset_index(drop=True),
        check_names=False,
    )


def test_email_domain_payee_mapped(harmonised_df, raw_transaction_df):
    pd.testing.assert_series_equal(
        harmonised_df["email_domain_payee"].reset_index(drop=True),
        raw_transaction_df["R_emaildomain"].reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Identity join tests
# ---------------------------------------------------------------------------


def test_device_type_populated_when_identity_joined(harmonised_df):
    """Rows with TransactionID 1, 2, 4 should have device_type populated."""
    # Rows 0, 1, 3 in the harmonised output (0-indexed after reset)
    assert harmonised_df.loc[0, "device_type"] == "desktop"
    assert harmonised_df.loc[1, "device_type"] == "mobile"
    assert harmonised_df.loc[3, "device_type"] == "desktop"


def test_device_type_nan_when_no_identity_match(harmonised_df):
    """Rows 2 and 4 (TransactionID 3, 5) have no identity record."""
    assert pd.isna(harmonised_df.loc[2, "device_type"])
    assert pd.isna(harmonised_df.loc[4, "device_type"])


def test_device_type_all_nan_without_identity(harmonised_no_identity):
    assert harmonised_no_identity["device_type"].isna().all()


def test_row_count_unchanged_after_identity_join(harmonised_df, harmonised_no_identity):
    assert len(harmonised_df) == len(harmonised_no_identity)


# ---------------------------------------------------------------------------
# Validation / error handling tests
# ---------------------------------------------------------------------------


def test_raises_on_missing_required_column(loader):
    bad_df = pd.DataFrame(
        {
            "TransactionID": [1],
            "isFraud": [0],
            "TransactionDT": [86400],
            # TransactionAmt missing
            "ProductCD": ["W"],
        }
    )
    with pytest.raises(ValueError, match="missing required IEEE-CIS columns"):
        loader.load_from_dataframes(bad_df)


def test_raises_file_not_found_on_bad_transaction_path(loader):
    with pytest.raises(FileNotFoundError, match="transaction file not found"):
        loader.load("/nonexistent/path/train_transaction.csv")


def test_raises_file_not_found_on_bad_identity_path(loader, tmp_path):
    # Create a minimal valid transaction file
    txn_df = pd.DataFrame(
        {
            "TransactionID": [1],
            "isFraud": [0],
            "TransactionDT": [86400],
            "TransactionAmt": [50.0],
            "ProductCD": ["W"],
        }
    )
    txn_file = tmp_path / "train_transaction.csv"
    txn_df.to_csv(txn_file, index=False)
    with pytest.raises(FileNotFoundError, match="identity file not found"):
        loader.load(txn_file, identity_path="/nonexistent/train_identity.csv")


# ---------------------------------------------------------------------------
# Round-trip CSV I/O test
# ---------------------------------------------------------------------------


def test_load_from_csv_matches_load_from_dataframes(loader, raw_transaction_df, tmp_path):
    """load() must produce the same result as load_from_dataframes()."""
    txn_file = tmp_path / "train_transaction.csv"
    raw_transaction_df.to_csv(txn_file, index=False)

    df_from_file = loader.load(txn_file)
    df_from_mem = loader.load_from_dataframes(raw_transaction_df)

    pd.testing.assert_frame_equal(df_from_file, df_from_mem)


# ---------------------------------------------------------------------------
# Compatibility with TypologyInjector
# ---------------------------------------------------------------------------


def test_harmonised_output_compatible_with_typology_injector(harmonised_df):
    """Harmonised output must satisfy _REQUIRED_COLS from typology_injector."""
    from fincrime_ml.core.data.typology_injector import _REQUIRED_COLS

    for col in _REQUIRED_COLS:
        assert (
            col in harmonised_df.columns
        ), f"Harmonised output missing column required by TypologyInjector: {col}"
