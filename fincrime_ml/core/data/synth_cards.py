"""
core/data/synth_cards.py
========================
Synthetic card and payment transaction generator for FinCrime-ML.

Generates realistic transaction data with:
    - Card payment transactions (POS, CNP, contactless)
    - Wire/SWIFT transfers with BIC/IBAN identifiers
    - Digital payment channels (mobile, e-commerce)
    - Configurable fraud typology injection

No real customer data is used or required. All records are synthetically
generated using statistical distributions calibrated to published industry
benchmarks (UK Finance Fraud Report, BIS CPMI payment statistics).

Usage:
    from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator

    gen = SyntheticTransactionGenerator(n_accounts=5000, seed=42)
    df = gen.generate(n_transactions=50_000, fraud_rate=0.015)

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from faker import Faker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Merchant category codes (MCC) with associated base risk tiers
# Source: ISO 18245 / UK Finance MCC risk segmentation
# ---------------------------------------------------------------------------
MCC_REGISTRY: dict[str, dict] = {
    "5411": {"name": "Grocery Stores", "risk": "low"},
    "5812": {"name": "Eating Places / Restaurants", "risk": "low"},
    "5999": {"name": "Miscellaneous Retail", "risk": "medium"},
    "7011": {"name": "Hotels and Lodging", "risk": "medium"},
    "5734": {"name": "Computer and Software Stores", "risk": "medium"},
    "7995": {"name": "Gambling / Betting", "risk": "high"},
    "6051": {"name": "Non-Financial Institutions (Crypto)", "risk": "high"},
    "4829": {"name": "Money Transfer / Remittance", "risk": "high"},
    "5912": {"name": "Drug Stores / Pharmacies", "risk": "low"},
    "5661": {"name": "Shoe Stores", "risk": "low"},
    "5651": {"name": "Family Clothing Stores", "risk": "low"},
    "5045": {"name": "Computers / Peripherals (B2B)", "risk": "medium"},
    "6011": {"name": "ATM / Cash Advance", "risk": "high"},
    "5094": {"name": "Jewellery / Watches", "risk": "high"},
    "4111": {"name": "Transportation", "risk": "low"},
}

MCC_RISK_WEIGHTS = {
    "low": 0.60,
    "medium": 0.30,
    "high": 0.10,
}

# ---------------------------------------------------------------------------
# Country corridors — relative transaction frequency weights
# High-risk corridors reflect FATF grey/black list distributions
# ---------------------------------------------------------------------------
COUNTRY_CORRIDORS: dict[str, float] = {
    "GB": 0.45,
    "US": 0.15,
    "DE": 0.08,
    "FR": 0.06,
    "NG": 0.04,
    "AE": 0.04,
    "HK": 0.03,
    "CN": 0.03,
    "RU": 0.02,
    "IR": 0.01,  # FATF black list — elevated in fraud scenarios
    "KP": 0.01,  # FATF black list
    "OTHER": 0.08,
}

ChannelType = Literal["POS", "CNP_ECOM", "CNP_MOTO", "CONTACTLESS", "WIRE", "MOBILE_APP"]


@dataclass
class GeneratorConfig:
    """Configuration for the synthetic transaction generator.

    Attributes:
        n_accounts: Number of unique customer accounts to simulate.
        start_date: Earliest possible transaction date.
        end_date: Latest possible transaction date.
        seed: Random seed for reproducibility.
        currency: ISO 4217 currency code for card transactions.
        high_risk_mcc_uplift: Fraud probability multiplier for high-risk MCCs.
        include_wire_transfers: Whether to include SWIFT wire transfers.
        mule_account_rate: Proportion of accounts designated as mule accounts
            (used in AML typology injection).
    """

    n_accounts: int = 5_000
    start_date: datetime = datetime(2024, 1, 1)
    end_date: datetime = datetime(2024, 12, 31)
    seed: int = 42
    currency: str = "GBP"
    high_risk_mcc_uplift: float = 3.5
    include_wire_transfers: bool = True
    mule_account_rate: float = 0.008  # ~0.8% mule accounts (industry estimate)


class SyntheticTransactionGenerator:
    """Generate synthetic payment transaction data for model training and evaluation.

    The generator produces a realistic account population with spending
    behaviour drawn from log-normal distributions. Fraud labels are injected
    via configurable typology patterns rather than uniform random assignment,
    producing more realistic cluster structure in the feature space.

    Example:
        >>> gen = SyntheticTransactionGenerator(n_accounts=2000, seed=42)
        >>> df = gen.generate(n_transactions=20_000, fraud_rate=0.012)
        >>> print(df.shape)
        (20000, 24)
        >>> print(df["is_fraud"].mean())
        0.012  # approximate

    Attributes:
        config: GeneratorConfig instance.
        accounts: DataFrame of generated account profiles.
    """

    SCHEMA_COLS = [
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
        "timestamp",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "account_age_days",
        "account_avg_spend",
        "account_spend_stddev",
        "is_international",
        "high_risk_corridor",
        "is_mule_account",
        "swift_bic",
        "iban",
        "is_fraud",
    ]

    def __init__(
        self,
        n_accounts: int = 5_000,
        seed: int = 42,
        config: GeneratorConfig | None = None,
    ) -> None:
        self.config = config or GeneratorConfig(n_accounts=n_accounts, seed=seed)
        self._rng = np.random.default_rng(self.config.seed)
        self._faker = Faker("en_GB")
        Faker.seed(self.config.seed)
        random.seed(self.config.seed)

        self.accounts = self._generate_accounts()
        logger.info(
            "SyntheticTransactionGenerator initialised: %d accounts, seed=%d",
            self.config.n_accounts,
            self.config.seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_transactions: int = 50_000,
        fraud_rate: float = 0.015,
    ) -> pd.DataFrame:
        """Generate a synthetic transaction dataset.

        Args:
            n_transactions: Total number of transactions to generate.
            fraud_rate: Target proportion of fraudulent transactions. The actual
                rate will be close but not exact due to typology-based injection.

        Returns:
            DataFrame conforming to the FinCrime-ML transaction schema with
            ``n_transactions`` rows and columns defined in SCHEMA_COLS.
        """
        logger.info(
            "Generating %d transactions (target fraud rate: %.1f%%)",
            n_transactions,
            fraud_rate * 100,
        )

        n_fraud = int(n_transactions * fraud_rate)
        n_legitimate = n_transactions - n_fraud

        legitimate = self._generate_legitimate_batch(n_legitimate)
        fraudulent = self._generate_fraud_batch(n_fraud)

        df = (
            pd.concat([legitimate, fraudulent], ignore_index=True)
            .sample(frac=1, random_state=self.config.seed)
            .reset_index(drop=True)
        )

        df = self._add_temporal_features(df)
        df = self._add_account_behaviour_features(df)
        df = df[self.SCHEMA_COLS]

        actual_fraud_rate = df["is_fraud"].mean()
        logger.info(
            "Generation complete: %d records, actual fraud rate: %.3f%%",
            len(df),
            actual_fraud_rate * 100,
        )
        return df

    def generate_wire_transfers(self, n: int = 5_000) -> pd.DataFrame:
        """Generate synthetic SWIFT wire transfer records.

        Includes BIC/IBAN identifiers, remittance corridors, and AML-relevant
        fields such as structured amount patterns (smurfing indicators).

        Args:
            n: Number of wire transfer records to generate.

        Returns:
            DataFrame with SWIFT transfer schema compatible with FinCrime-ML AML module.
        """
        records = []
        countries = list(COUNTRY_CORRIDORS.keys())
        weights = list(COUNTRY_CORRIDORS.values())

        for _ in range(n):
            origin = self._rng.choice(countries, p=np.array(weights) / sum(weights))
            destination = self._rng.choice(countries, p=np.array(weights) / sum(weights))

            # Smurfing indicator: amounts just below reporting thresholds
            # UK threshold: £10,000 (POCA 2002 s.330); EU: €10,000
            is_structured = self._rng.random() < 0.03
            if is_structured:
                amount = self._rng.uniform(8_500, 9_950)
            else:
                amount = float(self._rng.lognormal(mean=8.5, sigma=1.8))
                amount = min(amount, 500_000)

            records.append(
                {
                    "transfer_id": self._generate_swift_ref(),
                    "sender_bic": self._generate_bic(origin),
                    "receiver_bic": self._generate_bic(destination),
                    "sender_iban": self._generate_iban(origin),
                    "receiver_iban": self._generate_iban(destination),
                    "amount_gbp": round(amount, 2),
                    "currency": "GBP",
                    "country_origin": origin,
                    "country_destination": destination,
                    "timestamp": self._random_timestamp(),
                    "is_high_risk_corridor": (
                        origin in ("IR", "KP") or destination in ("IR", "KP")
                    ),
                    "is_structured_amount": is_structured,
                    "purpose_code": self._rng.choice(
                        ["SALA", "GDDS", "SVCS", "TRFD", "OTHR"],
                        p=[0.35, 0.25, 0.20, 0.15, 0.05],
                    ),
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Private: account population
    # ------------------------------------------------------------------

    def _generate_accounts(self) -> pd.DataFrame:
        """Create the account population with behavioural profiles."""
        n = self.config.n_accounts
        account_ids = [f"ACC{str(i).zfill(7)}" for i in range(n)]

        # Spending behaviour drawn from log-normal (right-skewed, realistic)
        avg_spend = self._rng.lognormal(mean=3.8, sigma=0.9, size=n)
        spend_std = avg_spend * self._rng.uniform(0.2, 0.8, size=n)

        # Account age in days (uniform 30 days–10 years)
        account_age = self._rng.integers(30, 3_650, size=n)

        # Mule account flags — small subset used in AML scenarios
        n_mules = max(1, int(n * self.config.mule_account_rate))
        mule_flags = [False] * n
        mule_indices = self._rng.choice(n, size=n_mules, replace=False)
        for idx in mule_indices:
            mule_flags[idx] = True

        return pd.DataFrame(
            {
                "account_id": account_ids,
                "account_avg_spend": np.round(avg_spend, 2),
                "account_spend_stddev": np.round(spend_std, 2),
                "account_age_days": account_age,
                "is_mule_account": mule_flags,
                "home_country": self._rng.choice(
                    ["GB", "US", "DE", "FR", "OTHER"],
                    p=[0.70, 0.10, 0.07, 0.07, 0.06],
                    size=n,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Private: transaction generation
    # ------------------------------------------------------------------

    def _generate_legitimate_batch(self, n: int) -> pd.DataFrame:
        """Generate n legitimate (non-fraud) transactions."""
        return self._build_transaction_batch(n, is_fraud=False)

    def _generate_fraud_batch(self, n: int) -> pd.DataFrame:
        """Generate n fraudulent transactions with typology-based patterns.

        Typology distribution (approximate, based on UK Finance 2023):
            - Card-not-present (CNP): 78%
            - Card-present skimming: 12%
            - Account takeover: 6%
            - Bust-out / first-party fraud: 4%
        """
        n_cnp = int(n * 0.78)
        n_skim = int(n * 0.12)
        n_ato = int(n * 0.06)
        n_bustout = n - n_cnp - n_skim - n_ato

        batches = [
            self._build_transaction_batch(n_cnp, is_fraud=True, channel="CNP_ECOM"),
            self._build_transaction_batch(n_skim, is_fraud=True, channel="POS"),
            self._build_transaction_batch(n_ato, is_fraud=True, channel="MOBILE_APP"),
            self._build_transaction_batch(n_bustout, is_fraud=True, channel="CNP_ECOM"),
        ]
        return pd.concat(batches, ignore_index=True)

    def _build_transaction_batch(
        self,
        n: int,
        is_fraud: bool,
        channel: ChannelType | None = None,
    ) -> pd.DataFrame:
        """Core transaction record builder."""
        account_sample = self.accounts.sample(n=n, replace=True, random_state=int(is_fraud))

        mccs = self._sample_mccs(n, is_fraud=is_fraud)
        channels = (
            [channel] * n
            if channel
            else self._rng.choice(
                ["POS", "CNP_ECOM", "CNP_MOTO", "CONTACTLESS", "MOBILE_APP"],
                size=n,
                p=[0.35, 0.30, 0.05, 0.20, 0.10],
            ).tolist()
        )

        countries_origin = self._rng.choice(
            list(COUNTRY_CORRIDORS.keys()),
            size=n,
            p=np.array(list(COUNTRY_CORRIDORS.values())) / sum(COUNTRY_CORRIDORS.values()),
        )
        countries_dest = self._rng.choice(
            list(COUNTRY_CORRIDORS.keys()),
            size=n,
            p=np.array(list(COUNTRY_CORRIDORS.values())) / sum(COUNTRY_CORRIDORS.values()),
        )

        # Fraud transactions: higher amounts, more international, unusual hours
        if is_fraud:
            amounts = (
                account_sample["account_avg_spend"].values
                * self._rng.lognormal(mean=1.2, sigma=0.6, size=n)
            )
            amounts = np.clip(amounts, 10, 15_000)
        else:
            amounts = np.abs(
                account_sample["account_avg_spend"].values
                + account_sample["account_spend_stddev"].values
                * self._rng.standard_normal(size=n)
            )
            amounts = np.clip(amounts, 0.5, 5_000)

        records = {
            "transaction_id": [self._generate_txn_id() for _ in range(n)],
            "account_id": account_sample["account_id"].values,
            "merchant_id": [self._generate_merchant_id() for _ in range(n)],
            "mcc": [m["mcc"] for m in mccs],
            "mcc_name": [m["name"] for m in mccs],
            "mcc_risk": [m["risk"] for m in mccs],
            "channel": channels,
            "amount_gbp": np.round(amounts, 2),
            "currency": self.config.currency,
            "country_origin": countries_origin,
            "country_destination": countries_dest,
            "timestamp": [self._random_timestamp() for _ in range(n)],
            "is_mule_account": account_sample["is_mule_account"].values,
            "account_avg_spend": account_sample["account_avg_spend"].values,
            "account_spend_stddev": account_sample["account_spend_stddev"].values,
            "account_age_days": account_sample["account_age_days"].values,
            "swift_bic": [
                self._generate_bic(c) if ch == "WIRE" else None
                for c, ch in zip(countries_origin, channels)
            ],
            "iban": [
                self._generate_iban(c) if ch == "WIRE" else None
                for c, ch in zip(countries_dest, channels)
            ],
            "is_fraud": int(is_fraud),
        }
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Private: feature enrichment
    # ------------------------------------------------------------------

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        return df

    def _add_account_behaviour_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_international"] = (df["country_origin"] != df["country_destination"]).astype(int)
        high_risk_countries = {"IR", "KP", "AE"}
        df["high_risk_corridor"] = (
            df["country_origin"].isin(high_risk_countries)
            | df["country_destination"].isin(high_risk_countries)
        ).astype(int)
        return df

    # ------------------------------------------------------------------
    # Private: sampling helpers
    # ------------------------------------------------------------------

    def _sample_mccs(self, n: int, is_fraud: bool) -> list[dict]:
        """Sample MCCs with fraud-adjusted risk distribution."""
        mcc_keys = list(MCC_REGISTRY.keys())
        mcc_values = list(MCC_REGISTRY.values())

        if is_fraud:
            # Fraud skews toward high-risk MCCs
            weights = [
                self.config.high_risk_mcc_uplift if v["risk"] == "high" else 1.0
                for v in mcc_values
            ]
        else:
            weights = [
                MCC_RISK_WEIGHTS[v["risk"]] * 10 for v in mcc_values
            ]

        total = sum(weights)
        probs = [w / total for w in weights]
        chosen_keys = self._rng.choice(mcc_keys, size=n, p=probs)

        return [
            {"mcc": k, **MCC_REGISTRY[k]}
            for k in chosen_keys
        ]

    def _random_timestamp(self) -> datetime:
        delta = self.config.end_date - self.config.start_date
        random_seconds = self._rng.integers(0, int(delta.total_seconds()))
        return self.config.start_date + timedelta(seconds=int(random_seconds))

    # ------------------------------------------------------------------
    # Private: identifier generators
    # ------------------------------------------------------------------

    def _generate_txn_id(self) -> str:
        """Generate a unique transaction reference in format TXN-XXXXXXXX."""
        return "TXN-" + "".join(
            self._faker.random_letters(8)
        ).upper()

    def _generate_merchant_id(self) -> str:
        """Generate a merchant identifier."""
        return "MER-" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )

    def _generate_swift_ref(self) -> str:
        """Generate a SWIFT message reference (20-character format)."""
        return "SWIFT" + "".join(
            random.choices(string.digits, k=15)
        )

    def _generate_bic(self, country: str) -> str:
        """Generate a plausible BIC (Bank Identifier Code) for a given country.

        Format: BBBBCCLLXXX where:
            BBBB = institution code (4 alpha)
            CC   = country code (2 alpha, ISO 3166)
            LL   = location code (2 alphanumeric)
            XXX  = branch code (optional, 3 alphanumeric)
        """
        institution = "".join(random.choices(string.ascii_uppercase, k=4))
        location = "".join(random.choices(string.ascii_uppercase + string.digits, k=2))
        country_code = country if country != "OTHER" else "XX"
        return f"{institution}{country_code}{location}XXX"

    def _generate_iban(self, country: str) -> str:
        """Generate a structurally valid synthetic IBAN.

        Note: These are synthetic — check digits are not validated.
        For GB IBANs: GB + 2 check digits + 4 bank code + 6 sort + 8 account.
        """
        country_code = country if country != "OTHER" else "XX"
        check = "".join(random.choices(string.digits, k=2))
        bank = "".join(random.choices(string.ascii_uppercase, k=4))
        rest = "".join(random.choices(string.digits, k=14))
        return f"{country_code}{check}{bank}{rest}"
