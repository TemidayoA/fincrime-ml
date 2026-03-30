"""
core/data/synth_aml.py
=======================
Synthetic AML transaction generator for FinCrime-ML.

Generates realistic anti-money laundering scenarios including:
    - Mule account chain seeding (placement → layering → integration)
    - AML structuring patterns (smurfing — amounts just below £10,000 threshold)
    - Round-trip transaction patterns (integration stage)
    - Mixed legitimate/suspicious datasets for supervised model training

All typologies are aligned to FATF 40 Recommendations and JMLSG Part I Ch.5
guidance on transaction monitoring. Label convention follows the FinCrime-ML
AML standard: ``is_suspicious`` (not ``is_fraud``, which is reserved for the
fraud domain).

No real customer data is used. All identifiers are synthetically generated.

Usage:
    from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

    gen = SyntheticAMLGenerator(n_accounts=3000, seed=42)
    df = gen.generate(n_transactions=20_000, suspicious_rate=0.05)
    chains = gen.generate_mule_chains(n_chains=30)
    structuring = gen.generate_structuring_transactions(n_clusters=50)

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UK SAR/STR reporting threshold — POCA 2002 s.330
# Structuring (smurfing) targets amounts just below this to avoid disclosure
# ---------------------------------------------------------------------------
UK_REPORTING_THRESHOLD_GBP: float = 10_000.0
STRUCTURING_UPPER_GBP: float = 9_950.0
STRUCTURING_LOWER_GBP: float = 8_500.0

# High-risk jurisdiction set for AML corridor weighting
# Sources: FATF grey list, JMLSG Part II sector-specific guidance
HIGH_RISK_JURISDICTIONS: frozenset[str] = frozenset({"IR", "KP", "RU", "AE", "OTHER"})

# Country weights — uplifted for high-risk corridors in suspicious scenarios
AML_COUNTRY_WEIGHTS: dict[str, float] = {
    "GB": 0.40,
    "AE": 0.10,
    "HK": 0.08,
    "CN": 0.06,
    "NG": 0.06,
    "RU": 0.05,
    "DE": 0.05,
    "US": 0.08,
    "IR": 0.04,
    "KP": 0.02,
    "OTHER": 0.06,
}

AMLTypology = Literal["structuring", "layering", "integration", "legitimate"]

AML_CHANNELS = ["WIRE", "MOBILE_APP", "OPEN_BANKING", "CNP_ECOM"]
AML_TRANSACTION_TYPES = ["TRANSFER", "DEPOSIT", "WITHDRAWAL", "EXCHANGE"]


@dataclass
class AMLGeneratorConfig:
    """Configuration for the synthetic AML transaction generator.

    Attributes:
        n_accounts: Total account population size.
        mule_account_rate: Proportion of accounts designated as mule accounts.
            Industry estimates suggest 0.5–2% of UK bank accounts are involved
            in authorised push payment (APP) mule activity (UK Finance 2023).
        n_mule_chains: Number of distinct mule chain sequences to generate
            when calling ``generate_mule_chains()``.
        chain_depth_min: Minimum number of hops in a mule chain (layering depth).
        chain_depth_max: Maximum number of hops in a mule chain.
        start_date: Earliest transaction timestamp.
        end_date: Latest transaction timestamp.
        seed: Random seed for reproducibility.
        currency: ISO 4217 currency code.
    """

    n_accounts: int = 3_000
    mule_account_rate: float = 0.02
    n_mule_chains: int = 20
    chain_depth_min: int = 2
    chain_depth_max: int = 5
    start_date: datetime = datetime(2024, 1, 1)
    end_date: datetime = datetime(2024, 12, 31)
    seed: int = 42
    currency: str = "GBP"


class SyntheticAMLGenerator:
    """Generate synthetic AML transaction data for model training and evaluation.

    Produces three core typology patterns aligned to the FATF money laundering
    lifecycle (placement → layering → integration):

    **Structuring (smurfing)**
        Multiple transactions from the same sender, each just below the
        £10,000 POCA 2002 s.330 reporting threshold, bunched within a short
        time window. The ``structuring_flag`` field marks these records.

    **Mule chain layering**
        Funds deposited into a seed account and rapidly forwarded through a
        chain of mule accounts. Each hop degrades the amount slightly
        (simulating fee extraction). The ``layering_depth`` field records the
        total chain length; ``rapid_movement_flag`` marks funds moved within
        24 hours of receipt.

    **Integration**
        Funds re-entering the legitimate economy via high-value single
        transfers, often to low-risk merchant categories or via property/
        investment channels.

    Example:
        >>> gen = SyntheticAMLGenerator(n_accounts=1000, seed=42)
        >>> df = gen.generate(n_transactions=10_000, suspicious_rate=0.05)
        >>> print(df["is_suspicious"].mean())
        0.05  # approximate

    Attributes:
        config: AMLGeneratorConfig instance.
        accounts: DataFrame of generated account profiles.
    """

    AML_SCHEMA_COLS = [
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

    def __init__(
        self,
        n_accounts: int = 3_000,
        seed: int = 42,
        config: AMLGeneratorConfig | None = None,
    ) -> None:
        self.config = config or AMLGeneratorConfig(n_accounts=n_accounts, seed=seed)
        self._rng = np.random.default_rng(self.config.seed)

        self.accounts = self._generate_accounts()
        logger.info(
            "SyntheticAMLGenerator initialised: %d accounts (%d mule), seed=%d",
            self.config.n_accounts,
            self.accounts["is_mule_account"].sum(),
            self.config.seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_transactions: int = 20_000,
        suspicious_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Generate a mixed AML dataset with suspicious and legitimate transactions.

        Suspicious typology split (approximate, calibrated to JMLSG estimates):
            - Structuring:  50% of suspicious transactions
            - Layering:     35%
            - Integration:  15%

        Args:
            n_transactions: Total number of transaction records.
            suspicious_rate: Target proportion labelled ``is_suspicious = 1``.

        Returns:
            Shuffled DataFrame conforming to AML_SCHEMA_COLS with
            ``n_transactions`` rows.
        """
        n_suspicious = int(n_transactions * suspicious_rate)
        n_legit = n_transactions - n_suspicious

        n_structuring = int(n_suspicious * 0.50)
        n_layering = int(n_suspicious * 0.35)
        n_integration = n_suspicious - n_structuring - n_layering

        batches = [
            self._generate_legitimate_batch(n_legit),
            self._generate_structuring_batch(n_structuring),
            self._generate_layering_batch(n_layering),
            self._generate_integration_batch(n_integration),
        ]

        df = (
            pd.concat(batches, ignore_index=True)
            .sample(frac=1, random_state=self.config.seed)
            .reset_index(drop=True)
        )
        df = self._add_temporal_features(df)
        df = df[self.AML_SCHEMA_COLS]

        actual_rate = df["is_suspicious"].mean()
        logger.info(
            "generate: %d records, actual suspicious rate: %.3f%%",
            len(df),
            actual_rate * 100,
        )
        return df

    def generate_mule_chains(self, n_chains: int | None = None) -> pd.DataFrame:
        """Generate mule chain transaction sequences for graph-based analysis.

        Each chain simulates the layering stage of money laundering: a seed
        amount is deposited into the first mule account and subsequently
        forwarded through a series of intermediary accounts. Each hop
        applies a small deduction (simulating fee extraction or partial
        withdrawal) and occurs within a configurable time window.

        This output is designed for use with the NetworkX graph builder
        (``fincrime_ml.aml.graph``) and graph-based anomaly scorers.

        Args:
            n_chains: Number of chains to generate. Defaults to
                ``config.n_mule_chains``.

        Returns:
            DataFrame with AML_SCHEMA_COLS plus ``chain_id`` and
            ``hop_number`` columns for graph construction.
        """
        n_chains = n_chains or self.config.n_mule_chains
        records: list[dict] = []

        mule_pool = self.accounts[self.accounts["is_mule_account"]]["account_id"].tolist()
        if len(mule_pool) < 2:
            raise ValueError(
                "Insufficient mule accounts for chain generation. "
                "Increase n_accounts or mule_account_rate."
            )

        countries = list(AML_COUNTRY_WEIGHTS.keys())
        country_probs = np.array(list(AML_COUNTRY_WEIGHTS.values()))
        country_probs = country_probs / country_probs.sum()

        for chain_id in range(n_chains):
            depth = int(
                self._rng.integers(self.config.chain_depth_min, self.config.chain_depth_max + 1)
            )
            # Sample distinct accounts for this chain (with replacement across chains)
            chain_size = depth + 1
            chain_accounts = self._rng.choice(
                mule_pool,
                size=min(chain_size, len(mule_pool)),
                replace=False,
            ).tolist()
            # Pad with repeats if chain longer than mule pool
            while len(chain_accounts) < chain_size:
                chain_accounts.append(self._rng.choice(mule_pool))

            # Seed amount — typically large, placed via high-risk corridor
            initial_amount = float(self._rng.lognormal(mean=9.5, sigma=0.8))
            initial_amount = min(initial_amount, 250_000.0)

            base_time = self._random_timestamp()

            for hop in range(depth):
                sender = chain_accounts[hop]
                receiver = chain_accounts[hop + 1]

                # Each hop: small deduction simulating extraction / fees
                deduction = float(self._rng.uniform(0.03, 0.15))
                hop_amount = round(initial_amount * ((1.0 - deduction) ** hop), 2)

                # Rapid movement: hours between hops (< 48h, often < 24h)
                delay_hours = int(self._rng.integers(1, 47))
                tx_time = base_time + timedelta(hours=delay_hours * (hop + 1))

                origin = str(self._rng.choice(countries, p=country_probs))
                destination = str(self._rng.choice(countries, p=country_probs))

                records.append(
                    {
                        "transaction_id": self._generate_txn_id(),
                        "sender_account_id": sender,
                        "receiver_account_id": receiver,
                        "amount_gbp": hop_amount,
                        "currency": self.config.currency,
                        "channel": str(self._rng.choice(["WIRE", "MOBILE_APP", "OPEN_BANKING"])),
                        "transaction_type": "TRANSFER",
                        "country_origin": origin,
                        "country_destination": destination,
                        "timestamp": tx_time,
                        "is_mule_sender": True,
                        "is_mule_receiver": True,
                        "layering_depth": depth,
                        "typology": "layering",
                        "structuring_flag": False,
                        "rapid_movement_flag": delay_hours < 24,
                        "is_suspicious": 1,
                        "chain_id": chain_id,
                        "hop_number": hop,
                    }
                )

        df = pd.DataFrame(records)
        df = self._add_temporal_features(df)
        logger.info("generate_mule_chains: %d records across %d chains", len(df), n_chains)
        return df

    def generate_structuring_transactions(self, n_clusters: int = 50) -> pd.DataFrame:
        """Generate structuring (smurfing) transaction clusters.

        Each cluster represents a single originator making multiple transfers
        just below the £10,000 POCA 2002 s.330 reporting threshold, spread
        over a short time window. This pattern is one of the most common AML
        red flags identified in JMLSG Part I Ch.5 guidance.

        Each cluster produces 3–7 transactions in the £8,500–£9,950 range
        from the same sender to different recipients within 24–72 hours.

        Args:
            n_clusters: Number of distinct structuring clusters.

        Returns:
            DataFrame with AML_SCHEMA_COLS plus ``cluster_id`` for grouping.
        """
        records: list[dict] = []
        all_account_ids = self.accounts["account_id"].tolist()

        for cluster_id in range(n_clusters):
            sender = str(self._rng.choice(all_account_ids))
            n_txns = int(self._rng.integers(3, 8))
            base_time = self._random_timestamp()

            for i in range(n_txns):
                amount = round(
                    float(self._rng.uniform(STRUCTURING_LOWER_GBP, STRUCTURING_UPPER_GBP)), 2
                )
                # Transactions spread over 24–72h window
                delay_hours = int(self._rng.integers(0, 24)) + (i * int(self._rng.integers(2, 12)))
                tx_time = base_time + timedelta(hours=delay_hours)

                receiver = str(self._rng.choice(all_account_ids))
                while receiver == sender:
                    receiver = str(self._rng.choice(all_account_ids))

                records.append(
                    {
                        "transaction_id": self._generate_txn_id(),
                        "sender_account_id": sender,
                        "receiver_account_id": receiver,
                        "amount_gbp": amount,
                        "currency": self.config.currency,
                        "channel": "WIRE",
                        "transaction_type": "TRANSFER",
                        "country_origin": "GB",
                        "country_destination": str(
                            self._rng.choice(
                                ["GB", "AE", "HK", "OTHER"],
                                p=[0.40, 0.25, 0.20, 0.15],
                            )
                        ),
                        "timestamp": tx_time,
                        "is_mule_sender": False,
                        "is_mule_receiver": False,
                        "layering_depth": 0,
                        "typology": "structuring",
                        "structuring_flag": True,
                        "rapid_movement_flag": False,
                        "is_suspicious": 1,
                        "cluster_id": cluster_id,
                    }
                )

        df = pd.DataFrame(records)
        df = self._add_temporal_features(df)
        logger.info(
            "generate_structuring_transactions: %d records across %d clusters",
            len(df),
            n_clusters,
        )
        return df

    # ------------------------------------------------------------------
    # Private: account population
    # ------------------------------------------------------------------

    def _generate_accounts(self) -> pd.DataFrame:
        """Create the account population with mule account flags."""
        n = self.config.n_accounts
        account_ids = [f"AML{str(i).zfill(7)}" for i in range(n)]

        n_mules = max(2, int(n * self.config.mule_account_rate))
        mule_flags = [False] * n
        mule_indices = self._rng.choice(n, size=n_mules, replace=False)
        for idx in mule_indices:
            mule_flags[int(idx)] = True

        avg_balance = self._rng.lognormal(mean=8.5, sigma=1.2, size=n)
        account_age = self._rng.integers(30, 2_500, size=n)

        return pd.DataFrame(
            {
                "account_id": account_ids,
                "is_mule_account": mule_flags,
                "avg_monthly_balance": np.round(avg_balance, 2),
                "account_age_days": account_age,
                "home_country": self._rng.choice(
                    ["GB", "US", "DE", "NG", "OTHER"],
                    p=[0.65, 0.10, 0.08, 0.07, 0.10],
                    size=n,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Private: batch generators
    # ------------------------------------------------------------------

    def _generate_legitimate_batch(self, n: int) -> pd.DataFrame:
        """Generate n legitimate (non-suspicious) transactions."""
        return self._build_transaction_batch(n, typology="legitimate", is_suspicious=False)

    def _generate_structuring_batch(self, n: int) -> pd.DataFrame:
        """Generate n structuring transactions for the mixed dataset."""
        records: list[dict] = []
        all_ids = self.accounts["account_id"].tolist()

        # Distribute n transactions across ~n/5 clusters (avg 5 per cluster)
        approx_clusters = max(1, n // 5)
        per_cluster = max(1, n // approx_clusters)
        remainder = n

        for _ in range(approx_clusters):
            batch_size = min(per_cluster, remainder)
            if batch_size <= 0:
                break
            sender = str(self._rng.choice(all_ids))
            base_time = self._random_timestamp()

            for i in range(batch_size):
                amount = round(
                    float(self._rng.uniform(STRUCTURING_LOWER_GBP, STRUCTURING_UPPER_GBP)), 2
                )
                delay_hours = i * int(self._rng.integers(2, 12))
                tx_time = base_time + timedelta(hours=delay_hours)
                receiver = str(self._rng.choice(all_ids))

                records.append(
                    {
                        "transaction_id": self._generate_txn_id(),
                        "sender_account_id": sender,
                        "receiver_account_id": receiver,
                        "amount_gbp": amount,
                        "currency": self.config.currency,
                        "channel": "WIRE",
                        "transaction_type": "TRANSFER",
                        "country_origin": "GB",
                        "country_destination": str(
                            self._rng.choice(
                                ["GB", "AE", "HK", "OTHER"],
                                p=[0.40, 0.25, 0.20, 0.15],
                            )
                        ),
                        "timestamp": tx_time,
                        "is_mule_sender": False,
                        "is_mule_receiver": False,
                        "layering_depth": 0,
                        "typology": "structuring",
                        "structuring_flag": True,
                        "rapid_movement_flag": False,
                        "is_suspicious": 1,
                    }
                )
            remainder -= batch_size

        return pd.DataFrame(records)

    def _generate_layering_batch(self, n: int) -> pd.DataFrame:
        """Generate n layering (mule chain) transactions for the mixed dataset."""
        mule_pool = self.accounts[self.accounts["is_mule_account"]]["account_id"].tolist()
        all_ids = self.accounts["account_id"].tolist()

        countries = list(AML_COUNTRY_WEIGHTS.keys())
        country_probs = np.array(list(AML_COUNTRY_WEIGHTS.values()))
        country_probs = country_probs / country_probs.sum()

        records: list[dict] = []
        while len(records) < n:
            sender = str(self._rng.choice(mule_pool if mule_pool else all_ids))
            receiver = str(self._rng.choice(all_ids))
            amount = float(self._rng.lognormal(mean=8.5, sigma=1.0))
            amount = min(round(amount, 2), 100_000.0)
            delay_hours = int(self._rng.integers(1, 47))
            origin = str(self._rng.choice(countries, p=country_probs))
            destination = str(self._rng.choice(countries, p=country_probs))

            records.append(
                {
                    "transaction_id": self._generate_txn_id(),
                    "sender_account_id": sender,
                    "receiver_account_id": receiver,
                    "amount_gbp": amount,
                    "currency": self.config.currency,
                    "channel": str(self._rng.choice(["WIRE", "MOBILE_APP", "OPEN_BANKING"])),
                    "transaction_type": "TRANSFER",
                    "country_origin": origin,
                    "country_destination": destination,
                    "timestamp": self._random_timestamp(),
                    "is_mule_sender": sender in mule_pool,
                    "is_mule_receiver": receiver in mule_pool,
                    "layering_depth": int(self._rng.integers(2, 6)),
                    "typology": "layering",
                    "structuring_flag": False,
                    "rapid_movement_flag": delay_hours < 24,
                    "is_suspicious": 1,
                }
            )

        return pd.DataFrame(records[:n])

    def _generate_integration_batch(self, n: int) -> pd.DataFrame:
        """Generate n integration-stage transactions for the mixed dataset.

        Integration transactions are typically large, single-hop transfers
        re-entering the legitimate economy — property purchases, investment
        accounts, or high-value retail. They are harder to detect in isolation
        but flagged by prior layering context.
        """
        return self._build_transaction_batch(
            n, typology="integration", is_suspicious=True, high_value=True
        )

    def _build_transaction_batch(
        self,
        n: int,
        typology: AMLTypology,
        is_suspicious: bool,
        high_value: bool = False,
    ) -> pd.DataFrame:
        """Core builder for legitimate and integration batches."""
        all_ids = self.accounts["account_id"].tolist()
        mule_ids = set(self.accounts[self.accounts["is_mule_account"]]["account_id"].tolist())

        countries = list(AML_COUNTRY_WEIGHTS.keys())
        country_probs = np.array(list(AML_COUNTRY_WEIGHTS.values()))
        country_probs = country_probs / country_probs.sum()

        records: list[dict] = []
        for _ in range(n):
            sender = str(self._rng.choice(all_ids))
            receiver = str(self._rng.choice(all_ids))

            if high_value:
                amount = float(self._rng.lognormal(mean=10.5, sigma=0.8))
                amount = min(round(amount, 2), 500_000.0)
            else:
                amount = float(self._rng.lognormal(mean=5.5, sigma=1.0))
                amount = min(round(amount, 2), 50_000.0)

            origin = str(self._rng.choice(countries, p=country_probs))
            destination = str(self._rng.choice(countries, p=country_probs))
            channel = str(self._rng.choice(AML_CHANNELS, p=[0.40, 0.30, 0.20, 0.10]))
            tx_type = str(self._rng.choice(AML_TRANSACTION_TYPES, p=[0.55, 0.20, 0.15, 0.10]))

            records.append(
                {
                    "transaction_id": self._generate_txn_id(),
                    "sender_account_id": sender,
                    "receiver_account_id": receiver,
                    "amount_gbp": amount,
                    "currency": self.config.currency,
                    "channel": channel,
                    "transaction_type": tx_type,
                    "country_origin": origin,
                    "country_destination": destination,
                    "timestamp": self._random_timestamp(),
                    "is_mule_sender": sender in mule_ids,
                    "is_mule_receiver": receiver in mule_ids,
                    "layering_depth": 0,
                    "typology": typology,
                    "structuring_flag": False,
                    "rapid_movement_flag": False,
                    "is_suspicious": int(is_suspicious),
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Private: feature enrichment
    # ------------------------------------------------------------------

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp and derive hour_of_day, day_of_week."""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        return df

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _random_timestamp(self) -> datetime:
        """Sample a random datetime within the configured date range."""
        delta = self.config.end_date - self.config.start_date
        random_seconds = int(self._rng.integers(0, int(delta.total_seconds())))
        return self.config.start_date + timedelta(seconds=random_seconds)

    def _generate_txn_id(self) -> str:
        """Generate a unique AML transaction reference in format AML-XXXXXXXXXX."""
        _alphanum = list(string.ascii_uppercase + string.digits)
        return "AML-" + "".join(self._rng.choice(_alphanum, size=10))
