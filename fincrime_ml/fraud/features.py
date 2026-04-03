"""
fraud/features.py
=================
Feature engineering for the fraud detection pipeline.

Computes three families of features that are the primary predictors of
payment fraud in UK card transaction data (UK Finance Fraud Report 2023):

Velocity features
    Per-account transaction counts and total amounts over rolling time
    windows (1 hour, 24 hours, 7 days). High velocity relative to the
    account baseline is the strongest single indicator of CNP fraud and
    account takeover. Windows are computed as backward-looking counts from
    each transaction timestamp.

Amount deviation features
    Z-score of the transaction amount against the account's historical
    mean and standard deviation. A large positive z-score signals an
    amount spike, characteristic of ATO fraud where the attacker acts
    quickly to extract maximum value before the account is locked.

MCC risk features
    Numeric encoding of the merchant category code risk tier and a
    binary flag for high-risk MCC combinations. High-risk MCCs (ATM,
    crypto, gambling, remittance) are over-represented in all major
    fraud typologies.

Regulatory context
    Velocity monitoring is explicitly required by JMLSG Part I Ch. 5.3
    as a transaction monitoring control. Amount deviation aligns to the
    MLR 2017 Reg. 28(11) obligation to monitor against the customer's
    established transaction profile.

Architecture note
    This module imports only from fincrime_ml.core. No imports from
    fincrime_ml.aml are permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: MCC risk tier to numeric score mapping.
#: Used as an ordinal feature; unknown MCCs are assigned 0 (conservative neutral).
MCC_RISK_SCORE: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "unknown": 0,
}

#: Rolling window sizes in hours.
VELOCITY_WINDOWS_HOURS: tuple[int, ...] = (1, 24, 168)  # 1h, 24h, 7d

#: Minimum standard deviation applied when computing z-scores to avoid
#: division by zero for accounts with a single historical transaction.
_MIN_STD: float = 1e-6


class FraudFeatureEngineer:
    """Compute fraud detection features from a transaction DataFrame.

    Operates on a DataFrame conforming to the FinCrime-ML core transaction
    schema. All feature methods return a new DataFrame with added columns;
    the input is never mutated.

    The primary entry point is ``transform``, which applies all three feature
    families in sequence and returns the enriched DataFrame.

    Example::

        from fincrime_ml.fraud.features import FraudFeatureEngineer

        engineer = FraudFeatureEngineer()
        df_features = engineer.transform(df)

    Attributes:
        velocity_windows: Rolling window sizes in hours (default: 1, 24, 168).
    """

    def __init__(
        self,
        velocity_windows: tuple[int, ...] = VELOCITY_WINDOWS_HOURS,
    ) -> None:
        self.velocity_windows = velocity_windows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all fraud feature families to the DataFrame.

        Applies velocity, amount deviation, and MCC risk features in sequence.
        Requires the FinCrime-ML core transaction schema columns:
        ``account_id``, ``timestamp``, ``amount_gbp``, ``account_avg_spend``,
        ``account_spend_stddev``, and ``mcc_risk``.

        Args:
            df: Input transaction DataFrame in FinCrime-ML schema.

        Returns:
            New DataFrame with all fraud feature columns appended. Row count
            and row order are preserved.

        Raises:
            ValueError: If any required column is absent.
        """
        self._validate(df)
        out = df.copy()
        out = self.add_velocity_features(out)
        out = self.add_amount_deviation_features(out)
        out = self.add_mcc_risk_features(out)
        logger.info(
            "FraudFeatureEngineer.transform: %d rows, %d feature columns added",
            len(out),
            len(out.columns) - len(df.columns),
        )
        return out

    def add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-account rolling transaction velocity features.

        For each window W in ``velocity_windows``, adds:
            - ``velocity_count_{W}h``: number of transactions by the same
              account in the W hours preceding (exclusive) this transaction.
            - ``velocity_amount_{W}h``: total amount transacted by the same
              account in the W hours preceding this transaction.

        Transactions are sorted by timestamp within each account group.
        The window is strictly backward-looking (the current transaction is
        excluded from its own window count).

        Args:
            df: DataFrame with ``account_id``, ``timestamp``, ``amount_gbp``.

        Returns:
            New DataFrame with velocity columns appended.
        """
        required = {"account_id", "timestamp", "amount_gbp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"add_velocity_features: missing columns {sorted(missing)}")

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"])

        # Sort by account and time to enable backward-looking window logic
        original_index = out.index
        out = out.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

        for window_h in self.velocity_windows:
            count_col = f"velocity_count_{window_h}h"
            amount_col = f"velocity_amount_{window_h}h"

            counts = np.zeros(len(out), dtype=np.int32)
            amounts = np.zeros(len(out), dtype=np.float64)

            # Process per account group
            for _, group in out.groupby("account_id", sort=False):
                idx = group.index.to_numpy()
                timestamps = group["timestamp"].to_numpy(dtype="datetime64[ns]")
                txn_amounts = group["amount_gbp"].to_numpy(dtype=np.float64)
                window_ns = np.timedelta64(window_h, "h")

                for i, (ts, pos) in enumerate(zip(timestamps, idx)):
                    cutoff = ts - window_ns
                    # Count/sum transactions strictly before current ts within window
                    mask = (timestamps[:i] >= cutoff) & (timestamps[:i] < ts)
                    counts[pos] = int(mask.sum())
                    amounts[pos] = float(txn_amounts[:i][mask].sum())

            out[count_col] = counts
            out[amount_col] = np.round(amounts, 2)

        # Restore original row order
        out = out.loc[original_index].reset_index(drop=True)
        logger.debug(
            "add_velocity_features: added %d velocity columns",
            len(self.velocity_windows) * 2,
        )
        return out

    def add_amount_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute amount deviation from the account's historical spend profile.

        Adds two columns:
            - ``amount_zscore``: (amount_gbp - account_avg_spend) / account_spend_stddev.
              A z-score above 3 indicates a significant amount spike relative to the
              account's normal behaviour. Capped at [-10, 10] to limit the influence
              of extreme outliers on tree-based models.
            - ``amount_over_avg_ratio``: amount_gbp / account_avg_spend. Provides a
              ratio-based alternative to the z-score; more interpretable for rule-based
              pre-screening (e.g. "amount is 5x the account average").

        Args:
            df: DataFrame with ``amount_gbp``, ``account_avg_spend``,
                ``account_spend_stddev``.

        Returns:
            New DataFrame with deviation columns appended.
        """
        required = {"amount_gbp", "account_avg_spend", "account_spend_stddev"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"add_amount_deviation_features: missing columns {sorted(missing)}")

        out = df.copy()

        std = out["account_spend_stddev"].clip(lower=_MIN_STD)
        z = (out["amount_gbp"] - out["account_avg_spend"]) / std
        out["amount_zscore"] = z.clip(-10.0, 10.0).round(4)

        avg = out["account_avg_spend"].clip(lower=_MIN_STD)
        out["amount_over_avg_ratio"] = (out["amount_gbp"] / avg).round(4)

        logger.debug("add_amount_deviation_features: added amount_zscore, amount_over_avg_ratio")
        return out

    def add_mcc_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode MCC risk tier as numeric features.

        Adds two columns:
            - ``mcc_risk_score``: ordinal encoding of ``mcc_risk``.
              low=1, medium=2, high=3, unknown=0.
            - ``is_high_risk_mcc``: binary flag; 1 if ``mcc_risk == "high"``,
              else 0. Provides a direct binary signal without requiring the model
              to learn the ordinal structure.

        Args:
            df: DataFrame with ``mcc_risk``.

        Returns:
            New DataFrame with MCC risk feature columns appended.
        """
        if "mcc_risk" not in df.columns:
            raise ValueError("add_mcc_risk_features: missing column 'mcc_risk'")

        out = df.copy()
        out["mcc_risk_score"] = out["mcc_risk"].map(MCC_RISK_SCORE).fillna(0).astype(int)
        out["is_high_risk_mcc"] = (out["mcc_risk"] == "high").astype(int)

        logger.debug("add_mcc_risk_features: added mcc_risk_score, is_high_risk_mcc")
        return out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        """Check all required columns are present for the full transform.

        Args:
            df: Input DataFrame.

        Raises:
            ValueError: If any required column is absent.
        """
        required = {
            "account_id",
            "timestamp",
            "amount_gbp",
            "account_avg_spend",
            "account_spend_stddev",
            "mcc_risk",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"FraudFeatureEngineer.transform: missing required columns: {sorted(missing)}"
            )


def compute_velocity_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = VELOCITY_WINDOWS_HOURS,
) -> pd.DataFrame:
    """Functional wrapper for velocity feature computation.

    Convenience function for callers who do not need the full
    FraudFeatureEngineer class interface.

    Args:
        df: Input transaction DataFrame.
        windows: Rolling window sizes in hours.

    Returns:
        DataFrame with velocity feature columns appended.
    """
    return FraudFeatureEngineer(velocity_windows=windows).add_velocity_features(df)


def compute_amount_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Functional wrapper for amount deviation feature computation.

    Args:
        df: Input transaction DataFrame.

    Returns:
        DataFrame with amount deviation columns appended.
    """
    return FraudFeatureEngineer().add_amount_deviation_features(df)


def compute_mcc_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Functional wrapper for MCC risk feature computation.

    Args:
        df: Input transaction DataFrame.

    Returns:
        DataFrame with MCC risk feature columns appended.
    """
    return FraudFeatureEngineer().add_mcc_risk_features(df)
