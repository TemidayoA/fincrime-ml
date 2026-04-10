"""
core/data/loaders.py
====================
Dataset loaders and schema harmonisers for external financial crime datasets.

Provides adapter classes that read raw CSV files from published datasets and
harmonise them to the FinCrime-ML internal transaction schema, making them
compatible with fraud and AML pipeline feature engineering and model training.

Supported datasets
------------------
* **IEEE-CIS Fraud Detection** (Kaggle 2019, Vesta Corporation)
  Two-file dataset: ``train_transaction.csv`` + ``train_identity.csv``.
  Binary fraud labels; ~590 k transactions; ~3.5% fraud rate.
  Output schema: FinCrime-ML fraud schema (``is_fraud`` label).

* **PaySim Mobile Money Simulator** (Lopez-Rojas et al., 2016)
  Simulated mobile money transactions modelled on real M-Pesa data.
  Binary fraud labels; ~6.3 M transactions; ~0.1% fraud rate.
  Output schema: FinCrime-ML AML schema (``is_suspicious`` label).
  Includes mule chain annotation layer: accounts involved in fraud
  transactions or exhibiting pass-through behaviour are flagged as
  mule senders/receivers, enabling graph-based AML model training.

No dataset files are bundled with this package. Callers must supply local file
paths. See each loader docstring for download instructions.

Regulatory note
---------------
Column mappings preserve fields relevant to FCA SYSC 6.3 transaction monitoring
— specifically channel, geography, and amount features relied on by rule-based
pre-screening layers. The ``mcc_risk`` sentinel value ``"unknown"`` is flagged
separately from ``"low"``/``"medium"``/``"high"`` so downstream rules can apply
conservative treatment per JMLSG Part I Ch. 5 guidance.

For PaySim: the mule annotation layer implements JMLSG Part I para 5.3.17
guidance on identifying pass-through accounts and rapid-movement patterns
as AML red flags. ``is_suspicious`` maps to FATF typology indicators.

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# PaySim — column mapping constants
# ---------------------------------------------------------------------------

#: PaySim transaction type → FinCrime-ML channel.
#: All PaySim types are mobile money channels (M-Pesa model).
_PAYSIM_TYPE_CHANNEL_MAP: dict[str, str] = {
    "CASH-IN": "MOBILE_APP",
    "CASH-OUT": "MOBILE_APP",
    "DEBIT": "MOBILE_APP",
    "PAYMENT": "CNP_ECOM",
    "TRANSFER": "MOBILE_APP",
}

#: PaySim transaction type → FinCrime-ML transaction_type.
_PAYSIM_TYPE_TXN_TYPE_MAP: dict[str, str] = {
    "CASH-IN": "DEPOSIT",
    "CASH-OUT": "WITHDRAWAL",
    "DEBIT": "WITHDRAWAL",
    "PAYMENT": "PAYMENT",
    "TRANSFER": "TRANSFER",
}

#: PaySim simulation epoch — each step represents one hour.
#: Lopez-Rojas et al. (2016) simulate 30 days; step 1 = hour 1 of day 1.
_PAYSIM_EPOCH: datetime = datetime(2024, 1, 1, 0, 0, 0)

#: UK structuring threshold bounds (POCA 2002 s.330).
_STRUCTURING_LOWER: float = 8_500.0
_STRUCTURING_UPPER: float = 9_950.0

#: Pass-through ratio threshold for mule account heuristic.
#: Accounts forwarding more than this fraction of received funds are flagged.
#: Calibrated to JMLSG Part I para 5.3.17 guidance on pass-through behaviour.
_MULE_PASS_THROUGH_THRESHOLD: float = 0.80

#: Rapid movement window in PaySim steps (1 step = 1 hour).
#: Funds moved within this window after receipt are flagged as rapid movement.
_RAPID_MOVEMENT_WINDOW_STEPS: int = 2

#: Required columns in a raw PaySim DataFrame.
_PAYSIM_REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud",
    }
)

#: Output columns produced by PaySimLoader after harmonisation and annotation.
#: Compatible with SyntheticAMLGenerator.AML_SCHEMA_COLS for AML pipeline use.
PAYSIM_AML_COLS: list[str] = [
    "transaction_id",
    "sender_account_id",
    "receiver_account_id",
    "amount_gbp",
    "currency",
    "channel",
    "transaction_type",
    "country_origin",
    "country_destination",
    "timestamp",
    "hour_of_day",
    "day_of_week",
    "is_mule_sender",
    "is_mule_receiver",
    "layering_depth",
    "typology",
    "structuring_flag",
    "rapid_movement_flag",
    "is_suspicious",
]


class PaySimLoader:
    """Load and harmonise the PaySim mobile money dataset with mule chain annotation.

    PaySim simulates mobile financial transactions based on a sample of real
    M-Pesa transaction data from one month in East Africa (Lopez-Rojas et al.,
    2016). It is the primary benchmark dataset for mobile money fraud and AML
    research, and is widely used to evaluate graph-based transaction monitoring.

    **Dataset download**::

        https://www.kaggle.com/datasets/ealaxi/paysim1
        Direct: https://www.kaggle.com/datasets/ealaxi/paysim1/download

    Required file: ``PS_20174392719_1491204439457_log.csv`` (single CSV,
    ~470 MB, ~6.3 M rows).

    Mule chain annotation layer
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Raw PaySim fraud labels (``isFraud``) identify individual fraudulent
    transactions but do not annotate the intermediate mule accounts through
    which funds are layered. This loader applies a two-signal annotation:

    **Signal 1 — Fraud involvement:**
        Any account (sender or receiver) appearing in an ``isFraud=1``
        transaction is flagged as a mule candidate.

    **Signal 2 — Pass-through behaviour:**
        Accounts whose total outflow (TRANSFER + CASH-OUT amounts) exceeds
        ``pass_through_threshold`` (default 80%) of their total inflow are
        flagged as pass-through mule accounts. This implements the JMLSG
        Part I para 5.3.17 red flag for accounts used purely as relay nodes.

    The union of both signals populates ``is_mule_sender`` and
    ``is_mule_receiver`` on every transaction row, enabling graph-based
    AML models (``GraphScorer``) to train on mule-annotated PaySim data.

    Column mapping summary
    ~~~~~~~~~~~~~~~~~~~~~~
    =================== ====================== ====================================
    PaySim column       FinCrime-ML column      Notes
    =================== ====================== ====================================
    step                timestamp               _PAYSIM_EPOCH + step * 1 hour
    step                hour_of_day             step mod 24
    step                day_of_week             (step // 24) mod 7
    type                channel                 CASH-IN/OUT/TRANSFER→MOBILE_APP
    type                transaction_type        CASH-OUT→WITHDRAWAL, etc.
    amount              amount_gbp              Currency unit treated as GBP
    nameOrig            sender_account_id       as-is (C…/M… prefix retained)
    nameDest            receiver_account_id     as-is
    isFraud             is_suspicious           PaySim fraud = AML suspicious
    =================== ====================== ====================================

    Sentinel values for unmappable columns
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``currency``            = ``"GBP"`` (unit-normalised)
    * ``country_origin``      = ``"OTHER"`` (M-Pesa; East Africa, not in UK corridor set)
    * ``country_destination`` = ``"OTHER"``
    * ``layering_depth``      = ``0`` (set to 1 for direct mule-to-mule TRANSFER)

    Example::

        >>> loader = PaySimLoader()
        >>> df = loader.load("data/PS_20174392719_1491204439457_log.csv")
        >>> df.shape[1]  # columns
        19
        >>> round(df["is_suspicious"].mean(), 4)
        0.0013
    """

    def load(
        self,
        csv_path: str | Path,
        pass_through_threshold: float = _MULE_PASS_THROUGH_THRESHOLD,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        """Load PaySim CSV and return a harmonised, annotated AML DataFrame.

        Args:
            csv_path: Path to the PaySim CSV file.
            pass_through_threshold: Outflow/inflow ratio above which an account
                is flagged as a pass-through mule (default: 0.80).
            nrows: If set, only load the first ``nrows`` rows. Useful for
                development and testing with the large (~6.3 M row) PaySim file.

        Returns:
            Harmonised and mule-annotated DataFrame with columns defined in
            ``PAYSIM_AML_COLS``.

        Raises:
            FileNotFoundError: If csv_path does not exist.
            ValueError: If the CSV is missing required PaySim columns.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"PaySim CSV file not found: {csv_path}\n"
                "Download from: https://www.kaggle.com/datasets/ealaxi/paysim1"
            )
        logger.info("Reading PaySim CSV: %s (nrows=%s)", csv_path, nrows)
        raw_df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
        return self.load_from_dataframes(raw_df, pass_through_threshold=pass_through_threshold)

    def load_from_dataframes(
        self,
        df: pd.DataFrame,
        pass_through_threshold: float = _MULE_PASS_THROUGH_THRESHOLD,
    ) -> pd.DataFrame:
        """Harmonise a pre-loaded PaySim DataFrame and apply mule annotation.

        Accepts a raw PaySim DataFrame (as loaded directly from CSV) and returns
        a harmonised, mule-annotated AML DataFrame. Useful for testing and for
        callers managing their own I/O.

        Args:
            df: Raw PaySim DataFrame (``PS_2017…_log.csv`` schema).
            pass_through_threshold: Outflow/inflow ratio above which an account
                is flagged as a pass-through mule (default: 0.80).

        Returns:
            Harmonised DataFrame with columns defined in ``PAYSIM_AML_COLS``.

        Raises:
            ValueError: If df is missing required PaySim columns.
        """
        self._validate(df)

        mule_accounts = self._identify_mule_accounts(df, pass_through_threshold)
        rapid_movers = self._identify_rapid_movers(df)
        harmonised = self._harmonise(df, mule_accounts, rapid_movers)

        logger.info(
            "PaySimLoader: harmonised %d rows, suspicious rate %.4f%%, mule accounts %d",
            len(harmonised),
            harmonised["is_suspicious"].mean() * 100,
            len(mule_accounts),
        )
        return harmonised

    # ------------------------------------------------------------------
    # Private: mule annotation
    # ------------------------------------------------------------------

    def _identify_mule_accounts(self, df: pd.DataFrame, pass_through_threshold: float) -> set[str]:
        """Identify mule accounts via fraud involvement and pass-through behaviour.

        Signal 1 — fraud involvement:
            All accounts (sender or receiver) in isFraud=1 transactions.

        Signal 2 — pass-through behaviour (JMLSG para 5.3.17):
            Accounts whose (TRANSFER + CASH-OUT outflow) / (CASH-IN + TRANSFER
            inflow) exceeds ``pass_through_threshold``. These accounts receive
            funds only to immediately forward them — a structural mule indicator.

        Args:
            df: Raw PaySim DataFrame.
            pass_through_threshold: Outflow/inflow ratio threshold.

        Returns:
            Set of account identifiers flagged as mule accounts.
        """
        # Signal 1: accounts in flagged fraud transactions
        fraud_rows = df[df["isFraud"] == 1]
        fraud_accounts: set[str] = set(fraud_rows["nameOrig"].astype(str)) | set(
            fraud_rows["nameDest"].astype(str)
        )

        # Signal 2: pass-through ratio analysis
        # Inflow: rows where the account is nameDest in CASH-IN or TRANSFER
        inflow_types = {"CASH-IN", "TRANSFER"}
        outflow_types = {"CASH-OUT", "TRANSFER"}

        inflow_df = df[df["type"].isin(inflow_types)][["nameDest", "amount"]]
        inflow_df = inflow_df.rename(columns={"nameDest": "account", "amount": "inflow"})
        total_inflow = inflow_df.groupby("account")["inflow"].sum()

        outflow_df = df[df["type"].isin(outflow_types)][["nameOrig", "amount"]]
        outflow_df = outflow_df.rename(columns={"nameOrig": "account", "amount": "outflow"})
        total_outflow = outflow_df.groupby("account")["outflow"].sum()

        flow = pd.DataFrame({"inflow": total_inflow, "outflow": total_outflow}).fillna(0.0)
        flow["pass_through"] = np.where(flow["inflow"] > 0, flow["outflow"] / flow["inflow"], 0.0)
        pass_through_accounts: set[str] = set(
            flow[flow["pass_through"] >= pass_through_threshold].index.astype(str)
        )

        mule_accounts = fraud_accounts | pass_through_accounts
        logger.debug(
            "_identify_mule_accounts: %d fraud-involved, %d pass-through, %d total",
            len(fraud_accounts),
            len(pass_through_accounts),
            len(mule_accounts),
        )
        return mule_accounts

    def _identify_rapid_movers(self, df: pd.DataFrame) -> set[str]:
        """Identify accounts that receive a TRANSFER and CASH-OUT within a short window.

        A rapid mover receives funds via TRANSFER and exits via CASH-OUT within
        ``_RAPID_MOVEMENT_WINDOW_STEPS`` steps (hours). This is a FATF Recommendation
        R.10 red flag for layering — funds are moved quickly to obscure their origin.

        Args:
            df: Raw PaySim DataFrame.

        Returns:
            Set of account identifiers exhibiting rapid movement behaviour.
        """
        # Accounts receiving TRANSFER
        received = df[df["type"] == "TRANSFER"][["step", "nameDest"]].rename(
            columns={"nameDest": "account", "step": "receive_step"}
        )
        # Accounts sending CASH-OUT
        sent = df[df["type"] == "CASH-OUT"][["step", "nameOrig"]].rename(
            columns={"nameOrig": "account", "step": "send_step"}
        )

        if received.empty or sent.empty:
            return set()

        # Join on account and check if send_step is within window of receive_step
        merged = received.merge(sent, on="account", how="inner")
        step_diff = merged["send_step"] - merged["receive_step"]
        rapid = merged[step_diff.between(0, _RAPID_MOVEMENT_WINDOW_STEPS)]
        return set(rapid["account"].astype(str))

    # ------------------------------------------------------------------
    # Private: harmonisation
    # ------------------------------------------------------------------

    def _harmonise(
        self,
        df: pd.DataFrame,
        mule_accounts: set[str],
        rapid_movers: set[str],
    ) -> pd.DataFrame:
        """Map raw PaySim columns to the FinCrime-ML AML schema."""
        n = len(df)
        out = pd.DataFrame()

        # --- identifiers ---
        out["transaction_id"] = "PSIM-" + df.index.astype(str).str.zfill(10)
        out["sender_account_id"] = df["nameOrig"].astype(str)
        out["receiver_account_id"] = df["nameDest"].astype(str)

        # --- amounts and currency ---
        out["amount_gbp"] = pd.to_numeric(df["amount"], errors="coerce").round(2)
        out["currency"] = "GBP"  # unit-normalised; PaySim uses generic currency units

        # --- channel and transaction type ---
        out["channel"] = df["type"].map(_PAYSIM_TYPE_CHANNEL_MAP).fillna("MOBILE_APP")
        out["transaction_type"] = df["type"].map(_PAYSIM_TYPE_TXN_TYPE_MAP).fillna("TRANSFER")

        # --- geography (PaySim = East Africa mobile money; mapped to OTHER) ---
        out["country_origin"] = "OTHER"
        out["country_destination"] = "OTHER"

        # --- temporal features from step (1 step = 1 hour) ---
        step = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)
        out["timestamp"] = step.apply(lambda s: _PAYSIM_EPOCH + timedelta(hours=int(s)))
        out["hour_of_day"] = step % 24
        out["day_of_week"] = (step // 24) % 7

        # --- mule annotation ---
        sender_ids = out["sender_account_id"]
        receiver_ids = out["receiver_account_id"]

        out["is_mule_sender"] = sender_ids.isin(mule_accounts).astype(int)
        out["is_mule_receiver"] = receiver_ids.isin(mule_accounts).astype(int)

        # --- layering depth: 0 for most; 1 for mule-to-mule TRANSFER ---
        is_mule_transfer = (
            (df["type"] == "TRANSFER")
            & out["is_mule_sender"].astype(bool)
            & out["is_mule_receiver"].astype(bool)
        )
        out["layering_depth"] = is_mule_transfer.astype(int)

        # --- typology annotation ---
        amounts = out["amount_gbp"]
        is_transfer = df["type"] == "TRANSFER"
        is_cashout = df["type"] == "CASH-OUT"
        is_fraud = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(bool)

        structuring = is_cashout & amounts.between(_STRUCTURING_LOWER, _STRUCTURING_UPPER)
        layering = is_transfer & out["is_mule_sender"].astype(bool)
        integration = is_fraud & ~layering & ~structuring

        typology = pd.Series(["legitimate"] * n, index=df.index)
        typology = typology.where(~structuring, "structuring")
        typology = typology.where(~layering, "layering")
        typology = typology.where(~integration, "integration")
        out["typology"] = typology.values

        # --- pattern flags ---
        out["structuring_flag"] = structuring.astype(int)
        out["rapid_movement_flag"] = sender_ids.isin(rapid_movers).astype(int)

        # --- label: PaySim isFraud maps to AML is_suspicious ---
        out["is_suspicious"] = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)

        return out[PAYSIM_AML_COLS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private: validation
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        """Check that required PaySim columns are present.

        Args:
            df: Raw PaySim DataFrame to validate.

        Raises:
            ValueError: If any required column is absent.
        """
        missing = _PAYSIM_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"PaySim DataFrame is missing required columns: {sorted(missing)}")
