"""
core/data/typology_injector.py
==============================
Fraud typology injector for FinCrime-ML.

Injects realistic fraud patterns into a transaction DataFrame by marking and
modifying existing records to exhibit the statistical signatures of specific
fraud typologies. Designed to work with the schema produced by
``SyntheticTransactionGenerator`` but accepts any DataFrame that includes the
minimum required columns.

Typologies implemented (UK Finance Fraud Report 2023 share shown):

**CNP — Card-Not-Present** (~78% of UK card fraud by value)
    E-commerce and MOTO transactions where the physical card is absent.
    Signals: CNP_ECOM/CNP_MOTO channel, high-risk MCC, IP country mismatch,
    3DS bypass, elevated amount vs account baseline.

**ATO — Account Takeover** (~6%)
    Credential compromise followed by fraudulent activity from the legitimate
    account. Signals: channel switch, unusual hour, amount spike, geographic
    jump relative to account home country.

**Bust-out — First-party fraud** (~4%)
    The account holder deliberately maximises credit and disappears.
    Signals: escalating amounts across sequential transactions, concentration
    in cash-advance and gambling MCCs, account tenure is typically short.

**Card-present skimming** (~12%)
    Cloned mag-stripe or shimming attacks at POS terminals / ATMs.
    Signals: POS channel, geographic displacement from home, rapid sequential
    small transactions (card testing) followed by larger extractive spend.

Usage::

    from fincrime_ml.core.data.typology_injector import TypologyInjector

    injector = TypologyInjector(seed=42)
    df_with_fraud = injector.inject_all(df, fraud_rate=0.015)

    # Inject a specific typology only
    df_cnp = injector.inject_cnp(df, n_frauds=200)

Architecture note:
    This module lives in ``core/data/`` and is shared infrastructure.
    It must not import from ``fincrime_ml.fraud`` or ``fincrime_ml.aml``.

Regulatory context:
    Typology definitions are aligned to the FATF 40 Recommendations and
    UK Finance Fraud Report 2023. CNP thresholds reference PSR 2017 / PSD2
    Strong Customer Authentication (SCA) obligations.

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum columns required by the injector (subset of SCHEMA_COLS)
# ---------------------------------------------------------------------------
_REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "transaction_id",
        "account_id",
        "channel",
        "amount_gbp",
        "mcc_risk",
        "country_origin",
        "country_destination",
        "hour_of_day",
        "is_fraud",
    }
)

# MCCs associated with each fraud typology — used when modifying records
_CNP_HIGH_RISK_MCCS: tuple[str, ...] = ("7995", "6051", "4829", "6011", "5094")
_SKIMMING_MCCS: tuple[str, ...] = ("6011", "5411", "4111")  # ATM, grocery, transport
_BUST_OUT_MCCS: tuple[str, ...] = ("6011", "7995", "5999")  # ATM, gambling, misc

# High-risk countries for CNP IP spoofing scenarios (FATF grey/black list)
_HIGH_RISK_IP_COUNTRIES: tuple[str, ...] = ("RU", "IR", "KP", "CN", "OTHER")

# Channels by typology
_CNP_CHANNELS = ("CNP_ECOM", "CNP_MOTO")
_ATO_CHANNELS = ("MOBILE_APP", "CNP_ECOM")
_SKIMMING_CHANNELS = ("POS",)

FraudTypology = Literal["cnp", "ato", "bust_out", "card_skimming"]


class TypologyInjector:
    """Inject fraud typology patterns into a transaction DataFrame.

    The injector operates in-place on a copy of the input DataFrame —
    the original is never mutated. It selects rows (or synthesises new rows)
    that will carry the ``is_fraud = 1`` label, then modifies their feature
    values to exhibit the statistical signatures of the target typology.

    This approach produces more realistic inter-feature correlation structure
    than uniformly random fraud labelling, making it suitable for training
    classifiers and for evaluating typology-specific detection rules.

    Example::

        >>> gen = SyntheticTransactionGenerator(n_accounts=2000, seed=42)
        >>> df = gen.generate(n_transactions=20_000, fraud_rate=0.0)
        >>> injector = TypologyInjector(seed=42)
        >>> df_fraud = injector.inject_all(df, fraud_rate=0.015)
        >>> print(df_fraud["is_fraud"].mean())
        0.015  # approximate

    Attributes:
        seed: Random seed used for all stochastic operations.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        logger.info("TypologyInjector initialised with seed=%d", seed)

    # ------------------------------------------------------------------
    # Public API — individual typologies
    # ------------------------------------------------------------------

    def inject_cnp(self, df: pd.DataFrame, n_frauds: int) -> pd.DataFrame:
        """Inject Card-Not-Present fraud patterns into the DataFrame.

        Modifies ``n_frauds`` rows to exhibit CNP fraud signatures:
            - Channel forced to ``CNP_ECOM`` or ``CNP_MOTO``
            - MCC risk elevated to ``high``
            - ``country_destination`` set to a high-risk corridor
            - Amount inflated (2–5× account average)
            - ``hour_of_day`` biased toward off-hours (0–6 h)
            - ``is_fraud`` set to 1

        Args:
            df: Input transaction DataFrame. Must include _REQUIRED_COLS.
            n_frauds: Number of rows to mark as CNP fraud.

        Returns:
            New DataFrame (copy) with ``n_frauds`` rows marked as CNP fraud.

        Raises:
            ValueError: If ``n_frauds`` exceeds the number of legitimate rows.
        """
        self._validate(df, n_frauds)
        out = df.copy()

        legit_idx = out[out["is_fraud"] == 0].index.tolist()
        chosen = self._rng.choice(legit_idx, size=n_frauds, replace=False).tolist()

        out.loc[chosen, "channel"] = self._rng.choice(_CNP_CHANNELS, size=n_frauds).tolist()
        out.loc[chosen, "mcc_risk"] = "high"
        out.loc[chosen, "country_destination"] = self._rng.choice(
            _HIGH_RISK_IP_COUNTRIES, size=n_frauds
        ).tolist()
        # Inflate amounts: CNP fraud skews 2–5× the row's existing amount
        inflation = self._rng.uniform(2.0, 5.0, size=n_frauds)
        out.loc[chosen, "amount_gbp"] = (out.loc[chosen, "amount_gbp"].values * inflation).clip(
            max=15_000.0
        )
        out.loc[chosen, "amount_gbp"] = np.round(out.loc[chosen, "amount_gbp"], 2)
        # Off-hours bias: 0–6 h local time (PSD2 SCA bypass window)
        self._set_int_col(out, chosen, "hour_of_day", self._rng.integers(0, 7, size=n_frauds))
        out.loc[chosen, "is_fraud"] = 1

        logger.info("inject_cnp: marked %d rows as CNP fraud", n_frauds)
        return out

    def inject_ato(self, df: pd.DataFrame, n_frauds: int) -> pd.DataFrame:
        """Inject Account Takeover fraud patterns into the DataFrame.

        ATO fraud occurs after credential compromise. Signals modelled:
            - Channel switched to ``MOBILE_APP`` or ``CNP_ECOM`` (new device)
            - Amount spike: 3–8× the row's existing amount (urgency to extract)
            - ``hour_of_day`` biased to night / early morning (0–5 h)
            - ``country_origin`` set to a high-risk jurisdiction (IP mismatch)
            - ``is_fraud`` set to 1

        Args:
            df: Input transaction DataFrame. Must include _REQUIRED_COLS.
            n_frauds: Number of rows to mark as ATO fraud.

        Returns:
            New DataFrame (copy) with ``n_frauds`` rows marked as ATO fraud.

        Raises:
            ValueError: If ``n_frauds`` exceeds the number of legitimate rows.
        """
        self._validate(df, n_frauds)
        out = df.copy()

        legit_idx = out[out["is_fraud"] == 0].index.tolist()
        chosen = self._rng.choice(legit_idx, size=n_frauds, replace=False).tolist()

        out.loc[chosen, "channel"] = self._rng.choice(_ATO_CHANNELS, size=n_frauds).tolist()
        # ATO: attacker acts urgently — high amount spike
        inflation = self._rng.uniform(3.0, 8.0, size=n_frauds)
        out.loc[chosen, "amount_gbp"] = (out.loc[chosen, "amount_gbp"].values * inflation).clip(
            max=20_000.0
        )
        out.loc[chosen, "amount_gbp"] = np.round(out.loc[chosen, "amount_gbp"], 2)
        # Night / early-morning session (attacker in different time zone)
        self._set_int_col(out, chosen, "hour_of_day", self._rng.integers(0, 6, size=n_frauds))
        # Origin country mismatch — attacker's IP in high-risk jurisdiction
        out.loc[chosen, "country_origin"] = self._rng.choice(
            _HIGH_RISK_IP_COUNTRIES, size=n_frauds
        ).tolist()
        out.loc[chosen, "is_fraud"] = 1

        logger.info("inject_ato: marked %d rows as ATO fraud", n_frauds)
        return out

    def inject_bust_out(self, df: pd.DataFrame, n_frauds: int) -> pd.DataFrame:
        """Inject bust-out (first-party) fraud patterns into the DataFrame.

        Bust-out fraud involves deliberate credit exhaustion by the account
        holder. Signals modelled:
            - MCC forced to high-risk categories (ATM, gambling, misc retail)
            - Amount set near account maximum (simulating credit exhaustion)
            - Channel biased to POS and CNP_ECOM (rapid multi-channel drain)
            - ``country_origin`` and ``country_destination`` both domestic (GB)
              — bust-out is typically domestic first-party fraud
            - ``is_fraud`` set to 1

        Args:
            df: Input transaction DataFrame. Must include _REQUIRED_COLS.
            n_frauds: Number of rows to mark as bust-out fraud.

        Returns:
            New DataFrame (copy) with ``n_frauds`` rows marked as bust-out fraud.

        Raises:
            ValueError: If ``n_frauds`` exceeds the number of legitimate rows.
        """
        self._validate(df, n_frauds)
        out = df.copy()

        legit_idx = out[out["is_fraud"] == 0].index.tolist()
        chosen = self._rng.choice(legit_idx, size=n_frauds, replace=False).tolist()

        out.loc[chosen, "mcc_risk"] = "high"
        # Bust-out: amounts cluster near credit limit — use lognormal high tail
        bust_amounts = self._rng.lognormal(mean=7.5, sigma=0.6, size=n_frauds)
        bust_amounts = np.clip(bust_amounts, 500.0, 12_000.0)
        out.loc[chosen, "amount_gbp"] = np.round(bust_amounts, 2)
        # Multi-channel drain
        out.loc[chosen, "channel"] = self._rng.choice(
            ["POS", "CNP_ECOM", "MOBILE_APP"], size=n_frauds, p=[0.40, 0.40, 0.20]
        ).tolist()
        # Domestic — bust-out is internal
        out.loc[chosen, "country_origin"] = "GB"
        out.loc[chosen, "country_destination"] = "GB"
        out.loc[chosen, "is_fraud"] = 1

        logger.info("inject_bust_out: marked %d rows as bust-out fraud", n_frauds)
        return out

    def inject_card_skimming(self, df: pd.DataFrame, n_frauds: int) -> pd.DataFrame:
        """Inject card-present skimming patterns into the DataFrame.

        Card-present skimming involves cloned mag-stripe or shimming attacks
        at POS terminals. Signals modelled:
            - Channel forced to ``POS``
            - ``country_destination`` set to a different country from ``country_origin``
              (geographic displacement — attacker uses clone elsewhere)
            - Two-phase amounts: small testing transactions followed by larger
              extractive transactions (modelled via bimodal distribution)
            - MCC biased to ATM, grocery, and transport (testing venues)
            - ``is_fraud`` set to 1

        Args:
            df: Input transaction DataFrame. Must include _REQUIRED_COLS.
            n_frauds: Number of rows to mark as skimming fraud.

        Returns:
            New DataFrame (copy) with ``n_frauds`` rows marked as skimming fraud.

        Raises:
            ValueError: If ``n_frauds`` exceeds the number of legitimate rows.
        """
        self._validate(df, n_frauds)
        out = df.copy()

        legit_idx = out[out["is_fraud"] == 0].index.tolist()
        chosen = self._rng.choice(legit_idx, size=n_frauds, replace=False).tolist()

        out.loc[chosen, "channel"] = "POS"
        out.loc[chosen, "mcc_risk"] = self._rng.choice(
            ["high", "medium"], size=n_frauds, p=[0.60, 0.40]
        ).tolist()

        # Bimodal amounts: ~40% are small testing transactions (£1–£20),
        # ~60% are extractive transactions (£50–£500)
        n_test = int(n_frauds * 0.40)
        n_extract = n_frauds - n_test
        test_amounts = self._rng.uniform(1.0, 20.0, size=n_test)
        extract_amounts = self._rng.uniform(50.0, 500.0, size=n_extract)
        all_amounts = np.concatenate([test_amounts, extract_amounts])
        self._rng.shuffle(all_amounts)
        out.loc[chosen, "amount_gbp"] = np.round(all_amounts, 2)

        # Geographic displacement: clone used abroad
        displacement_countries = ["AE", "US", "DE", "HK", "OTHER"]
        out.loc[chosen, "country_destination"] = self._rng.choice(
            displacement_countries, size=n_frauds
        ).tolist()
        out.loc[chosen, "is_fraud"] = 1

        logger.info("inject_card_skimming: marked %d rows as skimming fraud", n_frauds)
        return out

    def inject_all(
        self,
        df: pd.DataFrame,
        fraud_rate: float = 0.015,
        typology_mix: dict[FraudTypology, float] | None = None,
    ) -> pd.DataFrame:
        """Apply all four fraud typologies using a configurable mix.

        Typology default distribution mirrors UK Finance Fraud Report 2023:
            - cnp:          78%
            - card_skimming: 12%
            - ato:           6%
            - bust_out:      4%

        Args:
            df: Input transaction DataFrame with ``is_fraud = 0`` throughout
                (or an existing partial-fraud DataFrame).
            fraud_rate: Target proportion of transactions to label as fraud.
                Actual rate will be close but may differ by ±1 row due to
                integer rounding.
            typology_mix: Optional dict mapping typology name to its share of
                the fraud budget. Values must sum to 1.0. Defaults to the UK
                Finance distribution above.

        Returns:
            New DataFrame (copy) with fraud injected across all typologies.

        Raises:
            ValueError: If ``fraud_rate`` is not in (0, 1), or if
                ``typology_mix`` values do not sum to approximately 1.0.
        """
        if not 0.0 < fraud_rate < 1.0:
            raise ValueError(f"fraud_rate must be in (0, 1), got {fraud_rate}")

        mix = typology_mix or {
            "cnp": 0.78,
            "card_skimming": 0.12,
            "ato": 0.06,
            "bust_out": 0.04,
        }
        total_share = sum(mix.values())
        if abs(total_share - 1.0) > 1e-6:
            raise ValueError(f"typology_mix values must sum to 1.0, got {total_share:.6f}")

        n_legit = int((df["is_fraud"] == 0).sum())
        n_fraud_total = int(round(len(df) * fraud_rate))
        if n_fraud_total > n_legit:
            raise ValueError(
                f"Cannot inject {n_fraud_total} fraud rows — only {n_legit} legitimate "
                "rows available. Reduce fraud_rate or increase the dataset size."
            )

        # Allocate fraud budget across typologies (ensure integer totals)
        budgets: dict[str, int] = {}
        allocated = 0
        typologies = list(mix.keys())
        for i, typology in enumerate(typologies):
            if i < len(typologies) - 1:
                count = int(round(n_fraud_total * mix[typology]))
                budgets[typology] = count
                allocated += count
            else:
                # Last typology absorbs rounding remainder
                budgets[typology] = n_fraud_total - allocated

        out = df.copy()
        injectors = {
            "cnp": self.inject_cnp,
            "ato": self.inject_ato,
            "bust_out": self.inject_bust_out,
            "card_skimming": self.inject_card_skimming,
        }
        for typology, count in budgets.items():
            if count > 0:
                out = injectors[typology](out, n_frauds=count)

        actual_rate = out["is_fraud"].mean()
        logger.info(
            "inject_all: %d total fraud records injected (target %.3f%%, actual %.3f%%)",
            n_fraud_total,
            fraud_rate * 100,
            actual_rate * 100,
        )
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_int_col(
        self,
        df: pd.DataFrame,
        idx: list,
        col: str,
        values: np.ndarray,
    ) -> None:
        """Assign integer values to a column, preserving its existing dtype.

        Pandas 2.x raises ``LossySetitemError`` when assigning int64 values into
        an int32 column (e.g. ``hour_of_day`` derived from ``dt.hour``). This
        helper casts the values to the column's dtype before assignment.

        Args:
            df: DataFrame to modify in place.
            idx: Row index labels to update.
            col: Column name.
            values: Integer array of length ``len(idx)``.
        """
        col_dtype = df[col].dtype
        df.loc[idx, col] = values.astype(col_dtype)

    def _validate(self, df: pd.DataFrame, n_frauds: int) -> None:
        """Check that the DataFrame has required columns and sufficient rows.

        Args:
            df: Input DataFrame.
            n_frauds: Requested fraud count.

        Raises:
            ValueError: On missing columns or insufficient legitimate rows.
        """
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")
        n_legit = int((df["is_fraud"] == 0).sum())
        if n_frauds > n_legit:
            raise ValueError(
                f"Requested {n_frauds} fraud rows but only {n_legit} legitimate rows available."
            )
        if n_frauds <= 0:
            raise ValueError(f"n_frauds must be a positive integer, got {n_frauds}.")
