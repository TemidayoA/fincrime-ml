"""
core/data/loaders.py
====================
Dataset loaders and schema harmonisers for external fraud datasets.

Provides adapter classes that read raw CSV files from published fraud-detection
datasets and harmonise them to the FinCrime-ML internal transaction schema,
making them compatible with fraud pipeline feature engineering and model
training modules.

Supported datasets
------------------
* **IEEE-CIS Fraud Detection** (Kaggle 2019, Vesta Corporation)
  Two-file dataset: ``train_transaction.csv`` + ``train_identity.csv``.
  Binary fraud labels; ~590 k transactions; ~3.5% fraud rate.

No dataset files are bundled with this package. Callers must supply local file
paths. See each loader docstring for download instructions.

Regulatory note
---------------
Column mappings preserve fields relevant to FCA SYSC 6.3 transaction monitoring
— specifically channel, geography, and amount features relied on by rule-based
pre-screening layers. The ``mcc_risk`` sentinel value ``"unknown"`` is flagged
separately from ``"low"``/``"medium"``/``"high"`` so downstream rules can apply
conservative treatment per JMLSG Part I Ch. 5 guidance.

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IEEE-CIS — column mapping constants
# ---------------------------------------------------------------------------

#: ProductCD → FinCrime-ML channel mapping.
#: Interpretation based on public analysis of the Kaggle competition dataset.
#: W = web purchase (CNP e-commerce)
#: H = home / hotel services (CNP)
#: C = card-present retail
#: S = service / telephone order (MOTO)
#: R = retail card-present
_PRODUCT_CD_CHANNEL_MAP: dict[str, str] = {
    "W": "CNP_ECOM",
    "H": "CNP_ECOM",
    "C": "POS",
    "S": "CNP_MOTO",
    "R": "POS",
}

#: Sentinel values for IEEE-CIS columns that have no direct equivalent in the
#: FinCrime-ML schema. Documented here so downstream code can handle them
#: explicitly rather than treating them as normal categorical values.
_SENTINEL_CURRENCY = "USD"  # Dataset is US-centric; amounts are USD not GBP
_SENTINEL_COUNTRY = "US"  # addr2 is encoded; defaulting to US
_SENTINEL_MCC = "0000"  # IEEE-CIS has no MCC field
_SENTINEL_MCC_NAME = "Unknown"
_SENTINEL_MCC_RISK = "unknown"  # ≠ low/medium/high — signals absence of data
_SENTINEL_MERCHANT = "MER-UNKNOWN"

#: Output columns produced by IeeeCisLoader after harmonisation.
#: This is a superset of _REQUIRED_COLS from typology_injector.py, so the
#: harmonised DataFrame can be passed directly to TypologyInjector.
IEEE_CIS_HARMONISED_COLS: list[str] = [
    # FinCrime-ML core schema
    "transaction_id",
    "account_id",
    "merchant_id",
    "mcc",
    "mcc_name",
    "mcc_risk",
    "channel",
    "amount_gbp",
    "currency",
    "country_origin",
    "country_destination",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_international",
    "high_risk_corridor",
    "is_mule_account",
    "swift_bic",
    "iban",
    "is_fraud",
    # IEEE-CIS retained features (useful for downstream feature engineering)
    "transaction_dt_raw",  # original TransactionDT (seconds offset)
    "email_domain_payer",  # P_emaildomain
    "email_domain_payee",  # R_emaildomain
    "device_type",  # from identity join; NaN when identity not provided
]

# High-risk country set used to populate high_risk_corridor flag.
# Mirrors the set in synth_cards.py for consistency.
_HIGH_RISK_COUNTRIES: frozenset[str] = frozenset({"IR", "KP", "AE"})


class IeeeCisLoader:
    """Load and harmonise the IEEE-CIS Fraud Detection dataset.

    The IEEE-CIS dataset was released by Vesta Corporation for the Kaggle
    2019 Fraud Detection competition. It contains real-world e-commerce
    transactions with binary fraud labels (``isFraud``).

    **Dataset download**::

        https://www.kaggle.com/competitions/ieee-fraud-detection/data

    Required files: ``train_transaction.csv``, ``train_identity.csv``

    The loader merges both files on ``TransactionID`` (left join — identity is
    optional) and maps raw columns to the FinCrime-ML internal schema.  Columns
    that have no equivalent are filled with sentinel values documented in this
    module.

    Column mapping summary
    ~~~~~~~~~~~~~~~~~~~~~~
    =================== ====================== ====================================
    IEEE-CIS column     FinCrime-ML column      Notes
    =================== ====================== ====================================
    TransactionID       transaction_id          Prefixed "TXN-"
    TransactionDT       hour_of_day             seconds offset → mod 86400 ÷ 3600
    TransactionDT       day_of_week             seconds offset → div 86400 mod 7
    TransactionAmt      amount_gbp              USD amounts; currency = "USD"
    ProductCD           channel                 W/H→CNP_ECOM, C/R→POS, S→CNP_MOTO
    isFraud             is_fraud                direct map
    card1               account_id              masked card proxy, "ACC{card1:07.0f}"
    P_emaildomain       email_domain_payer      retained as-is
    R_emaildomain       email_domain_payee      retained as-is
    DeviceType          device_type             from identity join; NaN if absent
    =================== ====================== ====================================

    Sentinel values for unmappable columns
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``currency``            = ``"USD"``
    * ``country_origin``      = ``"US"``
    * ``country_destination`` = ``"US"``
    * ``mcc``                 = ``"0000"``
    * ``mcc_name``            = ``"Unknown"``
    * ``mcc_risk``            = ``"unknown"``  (not low/medium/high)
    * ``merchant_id``         = ``"MER-UNKNOWN"``
    * ``is_international``    = ``0``
    * ``high_risk_corridor``  = ``0``
    * ``is_mule_account``     = ``0``
    * ``swift_bic``           = ``None``
    * ``iban``                = ``None``

    Example::

        >>> loader = IeeeCisLoader()
        >>> df = loader.load(
        ...     transaction_path="data/train_transaction.csv",
        ...     identity_path="data/train_identity.csv",
        ... )
        >>> df.shape
        (590540, 20)
        >>> round(df["is_fraud"].mean(), 3)
        0.035
    """

    def load(
        self,
        transaction_path: str | Path,
        identity_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Load IEEE-CIS CSV files and return a harmonised DataFrame.

        Args:
            transaction_path: Path to ``train_transaction.csv``.
            identity_path: Path to ``train_identity.csv``. Optional — when
                omitted, ``device_type`` will be ``NaN`` for all rows.

        Returns:
            Harmonised DataFrame with columns defined in
            ``IEEE_CIS_HARMONISED_COLS``.

        Raises:
            FileNotFoundError: If either supplied path does not exist.
            ValueError: If the transaction file is missing required columns.
        """
        transaction_path = Path(transaction_path)
        if not transaction_path.exists():
            raise FileNotFoundError(
                f"IEEE-CIS transaction file not found: {transaction_path}\n"
                "Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data"
            )

        identity_df: pd.DataFrame | None = None
        if identity_path is not None:
            identity_path = Path(identity_path)
            if not identity_path.exists():
                raise FileNotFoundError(
                    f"IEEE-CIS identity file not found: {identity_path}\n"
                    "Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data"
                )
            logger.info("Reading IEEE-CIS identity file: %s", identity_path)
            identity_df = pd.read_csv(identity_path, low_memory=False)

        logger.info("Reading IEEE-CIS transaction file: %s", transaction_path)
        transaction_df = pd.read_csv(transaction_path, low_memory=False)

        return self.load_from_dataframes(transaction_df, identity_df)

    def load_from_dataframes(
        self,
        transactions_df: pd.DataFrame,
        identity_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Harmonise pre-loaded IEEE-CIS DataFrames.

        Accepts raw IEEE-CIS DataFrames (as loaded directly from CSV) and
        returns a harmonised FinCrime-ML DataFrame. Useful for testing and
        for callers who manage their own I/O.

        Args:
            transactions_df: Raw transaction DataFrame (``train_transaction.csv``
                schema).
            identity_df: Raw identity DataFrame (``train_identity.csv`` schema).
                Optional.

        Returns:
            Harmonised DataFrame with columns defined in
            ``IEEE_CIS_HARMONISED_COLS``.

        Raises:
            ValueError: If ``transactions_df`` is missing required columns.
        """
        self._validate_transaction_df(transactions_df)
        merged = self._merge(transactions_df, identity_df)
        harmonised = self._harmonise(merged)
        logger.info(
            "IeeeCisLoader: harmonised %d rows, fraud rate %.3f%%",
            len(harmonised),
            harmonised["is_fraud"].mean() * 100,
        )
        return harmonised

    # ------------------------------------------------------------------
    # Private — merge and harmonise
    # ------------------------------------------------------------------

    def _merge(
        self,
        transactions_df: pd.DataFrame,
        identity_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Left-join identity onto transactions on TransactionID."""
        if identity_df is None:
            logger.debug("No identity DataFrame provided — device_type will be NaN.")
            return transactions_df.copy()
        merged = transactions_df.merge(
            identity_df[["TransactionID", "DeviceType"]].drop_duplicates("TransactionID"),
            on="TransactionID",
            how="left",
        )
        logger.debug(
            "Merged %d transaction rows with identity; identity coverage: %.1f%%",
            len(merged),
            merged["DeviceType"].notna().mean() * 100,
        )
        return merged

    def _harmonise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw IEEE-CIS columns to the FinCrime-ML internal schema."""
        out = pd.DataFrame()

        # --- identifiers ---
        out["transaction_id"] = "TXN-" + df["TransactionID"].astype(str)
        out["account_id"] = df["card1"].apply(
            lambda x: f"ACC{int(x):07d}" if pd.notna(x) else "ACC_UNKNOWN"
        )
        out["merchant_id"] = _SENTINEL_MERCHANT

        # --- MCC (no equivalent in IEEE-CIS) ---
        out["mcc"] = _SENTINEL_MCC
        out["mcc_name"] = _SENTINEL_MCC_NAME
        out["mcc_risk"] = _SENTINEL_MCC_RISK

        # --- channel ---
        out["channel"] = (
            df["ProductCD"]
            .map(_PRODUCT_CD_CHANNEL_MAP)
            .fillna("CNP_ECOM")  # default for unknown ProductCD values
        )

        # --- amounts and currency ---
        out["amount_gbp"] = pd.to_numeric(df["TransactionAmt"], errors="coerce").round(2)
        out["currency"] = _SENTINEL_CURRENCY

        # --- geography ---
        # addr2 is an encoded numeric country field; we default to "US"
        # (dataset is US-centric). country_destination = country_origin (domestic).
        out["country_origin"] = _SENTINEL_COUNTRY
        out["country_destination"] = _SENTINEL_COUNTRY

        # --- temporal features from TransactionDT (seconds offset) ---
        dt_seconds = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).astype(int)
        out["transaction_dt_raw"] = dt_seconds
        out["hour_of_day"] = (dt_seconds % 86400) // 3600
        out["day_of_week"] = (dt_seconds // 86400) % 7
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

        # --- derived binary flags ---
        out["is_international"] = 0  # geography unavailable; conservative default
        out["high_risk_corridor"] = 0
        out["is_mule_account"] = 0

        # --- wire transfer fields (not applicable for this dataset) ---
        out["swift_bic"] = None
        out["iban"] = None

        # --- fraud label ---
        out["is_fraud"] = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)

        # --- retained IEEE-CIS features ---
        out["email_domain_payer"] = df["P_emaildomain"] if "P_emaildomain" in df.columns else None
        out["email_domain_payee"] = df["R_emaildomain"] if "R_emaildomain" in df.columns else None
        out["device_type"] = df["DeviceType"] if "DeviceType" in df.columns else None

        return out[IEEE_CIS_HARMONISED_COLS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private — validation
    # ------------------------------------------------------------------

    def _validate_transaction_df(self, df: pd.DataFrame) -> None:
        """Check that required IEEE-CIS columns are present.

        Args:
            df: Raw transaction DataFrame to validate.

        Raises:
            ValueError: If any required column is absent.
        """
        required = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"transactions_df is missing required IEEE-CIS columns: {sorted(missing)}"
            )
