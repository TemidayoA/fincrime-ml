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


# ===========================================================================
# PaySimLoader tests
# ===========================================================================

from fincrime_ml.core.data.loaders import (  # noqa: E402
    _PAYSIM_TYPE_CHANNEL_MAP,
    _PAYSIM_TYPE_TXN_TYPE_MAP,
    PAYSIM_AML_COLS,
    PaySimLoader,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal PaySim-format DataFrames
# ---------------------------------------------------------------------------


def _make_paysim_row(
    step: int,
    txn_type: str,
    amount: float,
    name_orig: str,
    name_dest: str,
    is_fraud: int = 0,
    old_bal_orig: float = 10_000.0,
    new_bal_orig: float = 0.0,
    old_bal_dest: float = 0.0,
    new_bal_dest: float = 0.0,
) -> dict:
    return {
        "step": step,
        "type": txn_type,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_bal_orig,
        "newbalanceOrig": new_bal_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": 0,
    }


@pytest.fixture(scope="module")
def raw_paysim_df() -> pd.DataFrame:
    """Minimal PaySim-format DataFrame covering all five transaction types."""
    rows = [
        _make_paysim_row(1, "CASH-IN", 500.0, "C001", "M001"),
        _make_paysim_row(2, "CASH-OUT", 9_200.0, "C002", "M002", is_fraud=1),
        _make_paysim_row(3, "TRANSFER", 15_000.0, "C003", "C004", is_fraud=1),
        _make_paysim_row(4, "PAYMENT", 75.0, "C005", "M003"),
        _make_paysim_row(5, "DEBIT", 200.0, "C006", "M004"),
        _make_paysim_row(6, "TRANSFER", 8_000.0, "C007", "C008"),
        _make_paysim_row(7, "CASH-OUT", 8_700.0, "C009", "M005"),  # structuring amount
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def paysim_loader() -> PaySimLoader:
    return PaySimLoader()


@pytest.fixture(scope="module")
def paysim_harmonised(paysim_loader, raw_paysim_df) -> pd.DataFrame:
    return paysim_loader.load_from_dataframes(raw_paysim_df)


# ---------------------------------------------------------------------------
# Schema / output column tests
# ---------------------------------------------------------------------------


def test_paysim_output_has_all_aml_cols(paysim_harmonised):
    for col in PAYSIM_AML_COLS:
        assert col in paysim_harmonised.columns, f"Missing column: {col}"


def test_paysim_output_column_order(paysim_harmonised):
    assert list(paysim_harmonised.columns) == PAYSIM_AML_COLS


def test_paysim_output_row_count(paysim_harmonised, raw_paysim_df):
    assert len(paysim_harmonised) == len(raw_paysim_df)


# ---------------------------------------------------------------------------
# Identifier tests
# ---------------------------------------------------------------------------


def test_paysim_transaction_id_prefixed(paysim_harmonised):
    assert paysim_harmonised["transaction_id"].str.startswith("PSIM-").all()


def test_paysim_sender_account_id_preserved(paysim_harmonised, raw_paysim_df):
    pd.testing.assert_series_equal(
        paysim_harmonised["sender_account_id"].reset_index(drop=True),
        raw_paysim_df["nameOrig"].astype(str).reset_index(drop=True),
        check_names=False,
    )


def test_paysim_receiver_account_id_preserved(paysim_harmonised, raw_paysim_df):
    pd.testing.assert_series_equal(
        paysim_harmonised["receiver_account_id"].reset_index(drop=True),
        raw_paysim_df["nameDest"].astype(str).reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Amount and currency tests
# ---------------------------------------------------------------------------


def test_paysim_amount_gbp_preserved(paysim_harmonised, raw_paysim_df):
    pd.testing.assert_series_equal(
        paysim_harmonised["amount_gbp"].reset_index(drop=True),
        raw_paysim_df["amount"].round(2).reset_index(drop=True),
        check_names=False,
    )


def test_paysim_currency_is_gbp(paysim_harmonised):
    assert (paysim_harmonised["currency"] == "GBP").all()


# ---------------------------------------------------------------------------
# Channel and transaction type mapping tests
# ---------------------------------------------------------------------------


def test_paysim_cash_in_channel(paysim_harmonised, raw_paysim_df):
    idx = raw_paysim_df[raw_paysim_df["type"] == "CASH-IN"].index
    assert (paysim_harmonised.loc[idx, "channel"] == "MOBILE_APP").all()


def test_paysim_payment_channel(paysim_harmonised, raw_paysim_df):
    idx = raw_paysim_df[raw_paysim_df["type"] == "PAYMENT"].index
    assert (paysim_harmonised.loc[idx, "channel"] == "CNP_ECOM").all()


def test_paysim_cash_out_txn_type(paysim_harmonised, raw_paysim_df):
    idx = raw_paysim_df[raw_paysim_df["type"] == "CASH-OUT"].index
    assert (paysim_harmonised.loc[idx, "transaction_type"] == "WITHDRAWAL").all()


def test_paysim_transfer_txn_type(paysim_harmonised, raw_paysim_df):
    idx = raw_paysim_df[raw_paysim_df["type"] == "TRANSFER"].index
    assert (paysim_harmonised.loc[idx, "transaction_type"] == "TRANSFER").all()


def test_paysim_type_channel_map_coverage():
    """All five PaySim types must be in the channel map."""
    assert set(_PAYSIM_TYPE_CHANNEL_MAP.keys()) == {
        "CASH-IN",
        "CASH-OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    }


def test_paysim_type_txn_map_coverage():
    assert set(_PAYSIM_TYPE_TXN_TYPE_MAP.keys()) == {
        "CASH-IN",
        "CASH-OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    }


# ---------------------------------------------------------------------------
# Temporal feature tests
# ---------------------------------------------------------------------------


def test_paysim_hour_of_day_range(paysim_harmonised):
    assert paysim_harmonised["hour_of_day"].between(0, 23).all()


def test_paysim_day_of_week_range(paysim_harmonised):
    assert paysim_harmonised["day_of_week"].between(0, 6).all()


def test_paysim_timestamp_is_datetime(paysim_harmonised):
    assert paysim_harmonised[
        "timestamp"
    ].dtype == "datetime64[ns]" or pd.api.types.is_datetime64_any_dtype(
        paysim_harmonised["timestamp"]
    )


def test_paysim_hour_of_day_from_step(paysim_harmonised, raw_paysim_df):
    expected = (raw_paysim_df["step"] % 24).reset_index(drop=True)
    pd.testing.assert_series_equal(
        paysim_harmonised["hour_of_day"].reset_index(drop=True),
        expected,
        check_names=False,
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Suspicion label tests
# ---------------------------------------------------------------------------


def test_paysim_is_suspicious_preserved(paysim_harmonised, raw_paysim_df):
    pd.testing.assert_series_equal(
        paysim_harmonised["is_suspicious"].reset_index(drop=True),
        raw_paysim_df["isFraud"].astype(int).reset_index(drop=True),
        check_names=False,
    )


def test_paysim_suspicious_count_matches(paysim_harmonised, raw_paysim_df):
    assert paysim_harmonised["is_suspicious"].sum() == raw_paysim_df["isFraud"].sum()


# ---------------------------------------------------------------------------
# Mule annotation tests
# ---------------------------------------------------------------------------


def test_paysim_mule_sender_is_int(paysim_harmonised):
    assert paysim_harmonised["is_mule_sender"].dtype in (int, "int64", "int32")


def test_paysim_mule_receiver_is_int(paysim_harmonised):
    assert paysim_harmonised["is_mule_receiver"].dtype in (int, "int64", "int32")


def test_paysim_fraud_sender_flagged_as_mule(paysim_harmonised, raw_paysim_df):
    """Senders in isFraud=1 transactions must have is_mule_sender=1."""
    fraud_idx = raw_paysim_df[raw_paysim_df["isFraud"] == 1].index
    assert (paysim_harmonised.loc[fraud_idx, "is_mule_sender"] == 1).all()


def test_paysim_fraud_receiver_flagged_as_mule(paysim_harmonised, raw_paysim_df):
    """Receivers in isFraud=1 transactions must have is_mule_receiver=1."""
    fraud_idx = raw_paysim_df[raw_paysim_df["isFraud"] == 1].index
    assert (paysim_harmonised.loc[fraud_idx, "is_mule_receiver"] == 1).all()


def test_paysim_mule_flags_binary(paysim_harmonised):
    assert set(paysim_harmonised["is_mule_sender"].unique()).issubset({0, 1})
    assert set(paysim_harmonised["is_mule_receiver"].unique()).issubset({0, 1})


def test_paysim_pass_through_mule_detection():
    """Accounts with high outflow/inflow ratio must be flagged as mules."""
    # C_MULE receives 10000 via CASH-IN then sends 9500 via CASH-OUT (95% pass-through)
    rows = [
        _make_paysim_row(1, "CASH-IN", 10_000.0, "SOURCE", "C_MULE"),
        _make_paysim_row(2, "CASH-OUT", 9_500.0, "C_MULE", "SINK"),
        _make_paysim_row(3, "CASH-IN", 500.0, "LEGIT", "CLEAN"),
    ]
    df = pd.DataFrame(rows)
    loader = PaySimLoader()
    result = loader.load_from_dataframes(df, pass_through_threshold=0.80)
    # C_MULE sends on row index 1 — should be flagged as mule_sender
    c_mule_rows = result[result["sender_account_id"] == "C_MULE"]
    assert len(c_mule_rows) > 0
    assert (c_mule_rows["is_mule_sender"] == 1).all()


# ---------------------------------------------------------------------------
# Structuring flag tests
# ---------------------------------------------------------------------------


def test_paysim_structuring_flag_set_for_threshold_amounts(paysim_harmonised, raw_paysim_df):
    """CASH-OUT transactions with amount in [8500, 9950] must have structuring_flag=1."""
    structuring_idx = raw_paysim_df[
        (raw_paysim_df["type"] == "CASH-OUT") & raw_paysim_df["amount"].between(8_500.0, 9_950.0)
    ].index
    if len(structuring_idx) > 0:
        assert (paysim_harmonised.loc[structuring_idx, "structuring_flag"] == 1).all()


def test_paysim_structuring_flag_binary(paysim_harmonised):
    assert set(paysim_harmonised["structuring_flag"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Rapid movement flag tests
# ---------------------------------------------------------------------------


def test_paysim_rapid_movement_flag_binary(paysim_harmonised):
    assert set(paysim_harmonised["rapid_movement_flag"].unique()).issubset({0, 1})


def test_paysim_rapid_movement_detected():
    """Account receiving TRANSFER and sending CASH-OUT within 2 steps is flagged."""
    rows = [
        _make_paysim_row(10, "TRANSFER", 5_000.0, "SENDER", "RELAY"),
        _make_paysim_row(11, "CASH-OUT", 4_800.0, "RELAY", "SINK"),  # 1 step later
        _make_paysim_row(20, "CASH-IN", 200.0, "OTHER", "CLEAN"),
    ]
    df = pd.DataFrame(rows)
    loader = PaySimLoader()
    result = loader.load_from_dataframes(df)
    relay_rows = result[result["sender_account_id"] == "RELAY"]
    assert len(relay_rows) > 0
    assert (relay_rows["rapid_movement_flag"] == 1).all()


# ---------------------------------------------------------------------------
# Typology annotation tests
# ---------------------------------------------------------------------------


def test_paysim_typology_col_exists(paysim_harmonised):
    assert "typology" in paysim_harmonised.columns


def test_paysim_typology_valid_values(paysim_harmonised):
    valid = {"legitimate", "structuring", "layering", "integration"}
    assert set(paysim_harmonised["typology"].unique()).issubset(valid)


def test_paysim_layering_depth_binary_ish(paysim_harmonised):
    """layering_depth should be 0 or 1 for PaySim (no multi-hop chain info)."""
    assert set(paysim_harmonised["layering_depth"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Geography sentinel tests
# ---------------------------------------------------------------------------


def test_paysim_country_origin_sentinel(paysim_harmonised):
    assert (paysim_harmonised["country_origin"] == "OTHER").all()


def test_paysim_country_destination_sentinel(paysim_harmonised):
    assert (paysim_harmonised["country_destination"] == "OTHER").all()


# ---------------------------------------------------------------------------
# Validation / error handling tests
# ---------------------------------------------------------------------------


def test_paysim_raises_on_missing_column(paysim_loader):
    bad_df = pd.DataFrame(
        {
            "step": [1],
            "type": ["TRANSFER"],
            "amount": [100.0],
            "nameOrig": ["C001"],
            # nameDest missing
            "isFraud": [0],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        paysim_loader.load_from_dataframes(bad_df)


def test_paysim_raises_file_not_found(paysim_loader):
    with pytest.raises(FileNotFoundError, match="PaySim CSV file not found"):
        paysim_loader.load("/nonexistent/paysim.csv")


# ---------------------------------------------------------------------------
# Round-trip CSV I/O test
# ---------------------------------------------------------------------------


def test_paysim_load_from_csv_matches_dataframes(paysim_loader, raw_paysim_df, tmp_path):
    csv_file = tmp_path / "paysim.csv"
    raw_paysim_df.to_csv(csv_file, index=False)
    df_from_file = paysim_loader.load(csv_file)
    df_from_mem = paysim_loader.load_from_dataframes(raw_paysim_df)
    pd.testing.assert_frame_equal(df_from_file, df_from_mem)


# ---------------------------------------------------------------------------
# Compatibility with AML pipeline (SyntheticAMLGenerator schema)
# ---------------------------------------------------------------------------


def test_paysim_output_compatible_with_aml_schema(paysim_harmonised):
    """PaySim output must have every column in SyntheticAMLGenerator.AML_SCHEMA_COLS."""
    from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

    for col in SyntheticAMLGenerator.AML_SCHEMA_COLS:
        assert col in paysim_harmonised.columns, f"Missing AML schema column: {col}"
