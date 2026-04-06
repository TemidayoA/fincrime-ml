"""
aml/typologies.py
==================
Rule-based AML typology detection engine for FinCrime-ML.

Purpose
    Before deploying a supervised ML model for AML transaction monitoring,
    firms typically implement a rule-based typology engine. Rule engines are
    required by most regulators as the first line of detection, and they
    provide the labelled training data from which ML models are subsequently
    trained. This module implements three FATF-defined typologies as
    deterministic, auditable detection rules.

Typologies implemented
    Structuring (smurfing)
        Breaking a large transaction into multiple smaller transactions to
        stay below the POCA 2002 s.330 reporting threshold of £10,000.
        JMLSG Part I para 5.3.11 flags bunched transactions from the same
        sender within a short time window, each individually below the
        threshold, as a primary structuring indicator.

    Layering
        Rapid movement of funds through intermediate accounts to obscure
        the money trail (placement -> layering stage of the FATF lifecycle).
        Detected by identifying accounts that receive funds and rapidly
        forward them onward within a configurable time window.

    Integration
        Re-entry of laundered funds into the legitimate economy via
        high-value single transfers, often to low-risk merchant or
        investment channels. Detected by identifying accounts that receive
        aggregate inflows materially above their historical norm and then
        execute a large single outbound transfer.

Output
    All detection methods return one row per (account, window) match with
    a confidence score, evidence dictionary, and typology label. The combined
    score() method returns one row per transaction with per-typology scores
    and a composite AML risk score.

Regulatory alignment
    FATF 40 Recommendations R.10 (customer due diligence) and R.20
        (suspicious transaction reporting) underpin all three typologies.
    JMLSG Part I Ch.5 paras 5.3.10-5.3.14: specific transaction
        monitoring indicators for structuring and rapid movement.
    POCA 2002 s.330: failure to disclose — the £10,000 threshold that
        structuring is designed to evade.
    MLR 2017 Reg.28: enhanced due diligence triggers that the typology
        engine feeds.

Architecture note
    This module imports only from fincrime_ml.core. No imports from
    fincrime_ml.fraud are permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UK regulatory constants
# ---------------------------------------------------------------------------

#: POCA 2002 s.330 reporting threshold. Structuring targets amounts just below.
UK_REPORTING_THRESHOLD_GBP: float = 10_000.0

#: Default structuring window: JMLSG guidance flags bunched transactions
#: within 24 hours as a primary indicator.
DEFAULT_STRUCTURING_WINDOW_HOURS: int = 24

#: Default minimum transaction count to constitute a structuring cluster.
DEFAULT_STRUCTURING_MIN_TXN: int = 2

#: Default rapid movement window for layering detection.
DEFAULT_LAYERING_WINDOW_HOURS: int = 24

#: High-risk country pairs that elevate integration detection confidence.
HIGH_RISK_COUNTRIES: frozenset[str] = frozenset({"IR", "KP", "RU", "AE", "OTHER"})

#: Percentile of account-level inflow above which a single large outflow
#: triggers an integration flag.
DEFAULT_INTEGRATION_INFLOW_PERCENTILE: float = 90.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TypologyMatch:
    """A single typology detection match for one account and time window.

    Attributes:
        account_id: The flagged account identifier.
        typology: One of "structuring", "layering", "integration".
        confidence: Rule-based confidence score in [0, 1]. Higher confidence
            reflects stronger evidence (more transactions, tighter window, etc.).
        n_transactions: Number of transactions involved in the match.
        total_amount_gbp: Total GBP value of transactions in the cluster.
        window_start: Earliest timestamp in the detection window.
        window_end: Latest timestamp in the detection window.
        evidence: Dict of supporting detection facts for audit purposes.
        transaction_ids: List of transaction IDs in the match.
    """

    account_id: str
    typology: str
    confidence: float
    n_transactions: int
    total_amount_gbp: float
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    evidence: dict = field(default_factory=dict)
    transaction_ids: list[str] = field(default_factory=list)


@dataclass
class TypologyScore:
    """Per-transaction typology risk scores.

    Attributes:
        transaction_id: Transaction identifier.
        structuring_score: Structuring detection score in [0, 1].
        layering_score: Layering detection score in [0, 1].
        integration_score: Integration detection score in [0, 1].
        composite_score: Weighted combination of all typology scores.
        dominant_typology: The typology with the highest individual score,
            or "none" if all scores are below the flag threshold.
        is_flagged: Whether the composite score exceeds the flag threshold.
    """

    transaction_id: str
    structuring_score: float
    layering_score: float
    integration_score: float
    composite_score: float
    dominant_typology: str
    is_flagged: bool


# ---------------------------------------------------------------------------
# Typology engine
# ---------------------------------------------------------------------------


class TypologyEngine:
    """Rule-based AML typology detection engine.

    Implements three FATF-defined money laundering typologies as deterministic
    detection rules: structuring (smurfing), layering (rapid fund movement),
    and integration (re-entry into the legitimate economy).

    All detection methods return a list of TypologyMatch objects. The combined
    score() method returns a DataFrame with one row per transaction, suitable
    for downstream model training or alert generation.

    Example::

        from fincrime_ml.aml.typologies import TypologyEngine
        from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

        gen = SyntheticAMLGenerator(seed=42)
        df = gen.generate(n_transactions=10_000, suspicious_rate=0.05)

        engine = TypologyEngine()
        matches = engine.detect_structuring(df)
        scores = engine.score(df)

    Attributes:
        reporting_threshold_gbp: Amount above which a single transaction
            would require disclosure (default: £10,000, POCA 2002 s.330).
        structuring_window_hours: Time window for structuring detection.
        structuring_min_txn: Minimum number of transactions in a cluster.
        layering_window_hours: Time window for rapid movement detection.
        flag_threshold: Composite score above which a transaction is flagged.
    """

    def __init__(
        self,
        reporting_threshold_gbp: float = UK_REPORTING_THRESHOLD_GBP,
        structuring_window_hours: int = DEFAULT_STRUCTURING_WINDOW_HOURS,
        structuring_min_txn: int = DEFAULT_STRUCTURING_MIN_TXN,
        layering_window_hours: int = DEFAULT_LAYERING_WINDOW_HOURS,
        flag_threshold: float = 0.5,
    ) -> None:
        self.reporting_threshold_gbp = reporting_threshold_gbp
        self.structuring_window_hours = structuring_window_hours
        self.structuring_min_txn = structuring_min_txn
        self.layering_window_hours = layering_window_hours
        self.flag_threshold = flag_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_structuring(
        self,
        df: pd.DataFrame,
        sender_col: str = "sender_account_id",
        amount_col: str = "amount_gbp",
        timestamp_col: str = "timestamp",
        txn_id_col: str = "transaction_id",
    ) -> list[TypologyMatch]:
        """Detect structuring (smurfing) patterns in transaction data.

        Identifies accounts that send multiple transactions within a rolling
        window, each individually below the reporting threshold, where the
        aggregate sum approaches or exceeds the threshold. This is the primary
        structuring indicator under JMLSG Part I para 5.3.11.

        Detection criteria:
            1. Each transaction in the cluster is below the reporting threshold.
            2. The cluster contains at least ``structuring_min_txn`` transactions.
            3. All transactions occur within ``structuring_window_hours``.
            4. The aggregate amount exceeds 50% of the reporting threshold
               (to exclude genuinely small, unrelated transactions).

        Args:
            df: Transaction DataFrame. Must contain sender_col, amount_col,
                timestamp_col, and txn_id_col.
            sender_col: Column name for the sending account identifier.
            amount_col: Column name for transaction amount in GBP.
            timestamp_col: Column name for transaction timestamp.
            txn_id_col: Column name for transaction identifier.

        Returns:
            List of TypologyMatch objects, one per detected structuring cluster.

        Raises:
            KeyError: If required columns are absent from df.
        """
        self._check_columns(df, [sender_col, amount_col, timestamp_col, txn_id_col])

        df_work = df.copy()
        df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])

        # Pre-filter: only transactions below the reporting threshold
        below_threshold = df_work[df_work[amount_col] < self.reporting_threshold_gbp].copy()

        matches: list[TypologyMatch] = []
        window_td = pd.Timedelta(hours=self.structuring_window_hours)
        min_aggregate = self.reporting_threshold_gbp * 0.50

        for account_id, group in below_threshold.groupby(sender_col):
            group = group.sort_values(timestamp_col)
            timestamps = group[timestamp_col].values
            amounts = group[amount_col].values
            txn_ids = group[txn_id_col].values

            # Sliding window over transactions sorted by time
            n = len(timestamps)
            i = 0
            while i < n:
                window_start = pd.Timestamp(timestamps[i])
                window_end = window_start + window_td

                # Collect all transactions within the window starting at i
                mask = (pd.to_datetime(timestamps) >= window_start) & (
                    pd.to_datetime(timestamps) <= window_end
                )
                window_txns = np.where(mask)[0]

                if len(window_txns) >= self.structuring_min_txn:
                    window_amounts = amounts[window_txns]
                    total = float(window_amounts.sum())

                    if total >= min_aggregate:
                        # Confidence: scales with how close the total is to the threshold
                        # and how many transactions are in the cluster
                        threshold_ratio = min(total / self.reporting_threshold_gbp, 1.0)
                        count_factor = min(len(window_txns) / 5.0, 1.0)
                        confidence = round(0.6 * threshold_ratio + 0.4 * count_factor, 4)

                        match = TypologyMatch(
                            account_id=str(account_id),
                            typology="structuring",
                            confidence=confidence,
                            n_transactions=len(window_txns),
                            total_amount_gbp=round(total, 2),
                            window_start=window_start,
                            window_end=pd.Timestamp(timestamps[window_txns[-1]]),
                            evidence={
                                "threshold_gbp": self.reporting_threshold_gbp,
                                "window_hours": self.structuring_window_hours,
                                "aggregate_pct_of_threshold": round(
                                    total / self.reporting_threshold_gbp * 100, 1
                                ),
                                "max_single_amount": round(float(window_amounts.max()), 2),
                            },
                            transaction_ids=[str(txn_ids[j]) for j in window_txns],
                        )
                        matches.append(match)

                        # Advance past this window to avoid overlapping clusters
                        i = int(window_txns[-1]) + 1
                        continue

                i += 1

        logger.info(
            "detect_structuring: %d clusters detected from %d accounts",
            len(matches),
            df_work[sender_col].nunique(),
        )
        return matches

    def detect_layering(
        self,
        df: pd.DataFrame,
        sender_col: str = "sender_account_id",
        receiver_col: str = "receiver_account_id",
        amount_col: str = "amount_gbp",
        timestamp_col: str = "timestamp",
        txn_id_col: str = "transaction_id",
    ) -> list[TypologyMatch]:
        """Detect layering patterns via rapid fund movement.

        Identifies accounts that receive funds and rapidly forward them to
        another account within the layering window. This is consistent with
        the mule account behaviour described in JMLSG Part I para 5.3.13
        and FATF Recommendation R.10 enhanced monitoring triggers.

        Detection criteria:
            1. An account receives a credit (appears as receiver_account_id).
            2. The same account sends a debit within ``layering_window_hours``.
            3. The outbound amount is at least 50% of the inbound amount
               (simulating fee extraction common in mule chain layering).

        Args:
            df: Transaction DataFrame.
            sender_col: Column name for the sending account.
            receiver_col: Column name for the receiving account.
            amount_col: Column name for transaction amount in GBP.
            timestamp_col: Column name for transaction timestamp.
            txn_id_col: Column name for transaction identifier.

        Returns:
            List of TypologyMatch objects, one per detected layering event.

        Raises:
            KeyError: If required columns are absent from df.
        """
        self._check_columns(df, [sender_col, receiver_col, amount_col, timestamp_col, txn_id_col])

        df_work = df.copy()
        df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])
        window_td = pd.Timedelta(hours=self.layering_window_hours)

        matches: list[TypologyMatch] = []

        # Build an inbound index: for each account, when did they receive funds?
        inbound = (
            df_work[[receiver_col, amount_col, timestamp_col, txn_id_col]]
            .rename(columns={receiver_col: "account_id", txn_id_col: "inbound_txn_id"})
            .copy()
        )
        outbound = (
            df_work[[sender_col, amount_col, timestamp_col, txn_id_col]]
            .rename(columns={sender_col: "account_id", txn_id_col: "outbound_txn_id"})
            .copy()
        )

        all_accounts = set(inbound["account_id"].unique()) & set(outbound["account_id"].unique())

        for account_id in all_accounts:
            acc_in = inbound[inbound["account_id"] == account_id].sort_values(timestamp_col)
            acc_out = outbound[outbound["account_id"] == account_id].sort_values(timestamp_col)

            for _, inrow in acc_in.iterrows():
                in_time = inrow[timestamp_col]
                in_amount = inrow[amount_col]
                min_out = in_amount * 0.50

                # Find outbound transactions within the layering window
                rapid = acc_out[
                    (acc_out[timestamp_col] > in_time)
                    & (acc_out[timestamp_col] <= in_time + window_td)
                    & (acc_out[amount_col] >= min_out)
                ]

                if len(rapid) > 0:
                    total_out = float(rapid[amount_col].sum())
                    pass_through_ratio = total_out / in_amount if in_amount > 0 else 0.0
                    time_delta_hours = float(
                        (rapid[timestamp_col].min() - in_time).total_seconds() / 3600
                    )

                    # Confidence: higher for faster movement and higher pass-through ratio
                    speed_factor = max(0.0, 1.0 - time_delta_hours / self.layering_window_hours)
                    ratio_factor = min(pass_through_ratio, 1.0)
                    confidence = round(0.5 * speed_factor + 0.5 * ratio_factor, 4)

                    match = TypologyMatch(
                        account_id=str(account_id),
                        typology="layering",
                        confidence=confidence,
                        n_transactions=1 + len(rapid),
                        total_amount_gbp=round(in_amount + total_out, 2),
                        window_start=in_time,
                        window_end=rapid[timestamp_col].max(),
                        evidence={
                            "inbound_amount_gbp": round(float(in_amount), 2),
                            "outbound_amount_gbp": round(total_out, 2),
                            "pass_through_ratio": round(pass_through_ratio, 4),
                            "hours_to_forward": round(time_delta_hours, 2),
                            "layering_window_hours": self.layering_window_hours,
                        },
                        transaction_ids=[str(inrow["inbound_txn_id"])]
                        + list(rapid["outbound_txn_id"].astype(str)),
                    )
                    matches.append(match)

        logger.info(
            "detect_layering: %d rapid movement events detected across %d accounts",
            len(matches),
            len(all_accounts),
        )
        return matches

    def detect_integration(
        self,
        df: pd.DataFrame,
        sender_col: str = "sender_account_id",
        receiver_col: str = "receiver_account_id",
        amount_col: str = "amount_gbp",
        timestamp_col: str = "timestamp",
        txn_id_col: str = "transaction_id",
        country_origin_col: str | None = "country_origin",
        country_dest_col: str | None = "country_destination",
    ) -> list[TypologyMatch]:
        """Detect integration patterns via large single outflows after aggregate inflows.

        Identifies accounts where accumulated inbound transactions are followed
        by a single large outbound transfer, consistent with the integration
        stage of the FATF money laundering lifecycle. High-risk country corridors
        (FATF grey/black list) elevate the confidence score.

        Detection criteria:
            1. An account's total inflows over the observation period exceed the
               ``integration_inflow_percentile`` (default: 90th percentile of
               all accounts).
            2. The account executes at least one outbound transaction with
               amount >= 70% of its total inbound value.

        Args:
            df: Transaction DataFrame.
            sender_col: Sending account column.
            receiver_col: Receiving account column.
            amount_col: Amount column in GBP.
            timestamp_col: Timestamp column.
            txn_id_col: Transaction identifier column.
            country_origin_col: Origin country column (ISO 3166-1 alpha-2).
                Pass None if not present in df.
            country_dest_col: Destination country column. Pass None if absent.

        Returns:
            List of TypologyMatch objects for detected integration events.

        Raises:
            KeyError: If required columns are absent from df.
        """
        required = [sender_col, receiver_col, amount_col, timestamp_col, txn_id_col]
        self._check_columns(df, required)

        df_work = df.copy()
        df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])

        # Compute per-account total inflows
        inflow_totals = df_work.groupby(receiver_col)[amount_col].sum().rename("total_inflow")
        if len(inflow_totals) == 0:
            return []
        threshold_inflow = float(
            np.percentile(inflow_totals.values, DEFAULT_INTEGRATION_INFLOW_PERCENTILE)
        )
        high_inflow_accounts = set(inflow_totals[inflow_totals >= threshold_inflow].index)

        matches: list[TypologyMatch] = []

        for account_id in high_inflow_accounts:
            total_inflow = float(inflow_totals[account_id])
            min_integration_amount = total_inflow * 0.70

            # Outbound transactions from this account
            outbound = df_work[
                (df_work[sender_col] == account_id)
                & (df_work[amount_col] >= min_integration_amount)
            ]

            for _, row in outbound.iterrows():
                out_amount = float(row[amount_col])
                integration_ratio = out_amount / total_inflow if total_inflow > 0 else 0.0

                # Base confidence from integration ratio
                confidence = round(min(integration_ratio, 1.0) * 0.8, 4)

                # Uplift for high-risk country corridors
                high_risk = False
                if country_origin_col and country_origin_col in df_work.columns:
                    orig = str(row.get(country_origin_col, ""))
                    dest_val = (
                        str(row.get(country_dest_col, ""))
                        if country_dest_col and country_dest_col in df_work.columns
                        else ""
                    )
                    if orig in HIGH_RISK_COUNTRIES or dest_val in HIGH_RISK_COUNTRIES:
                        high_risk = True
                        confidence = round(min(confidence + 0.15, 1.0), 4)

                match = TypologyMatch(
                    account_id=str(account_id),
                    typology="integration",
                    confidence=confidence,
                    n_transactions=1,
                    total_amount_gbp=round(out_amount, 2),
                    window_start=row[timestamp_col],
                    window_end=row[timestamp_col],
                    evidence={
                        "total_inflow_gbp": round(total_inflow, 2),
                        "outbound_amount_gbp": round(out_amount, 2),
                        "integration_ratio": round(integration_ratio, 4),
                        "high_risk_corridor": high_risk,
                        "inflow_percentile_threshold": round(threshold_inflow, 2),
                    },
                    transaction_ids=[str(row[txn_id_col])],
                )
                matches.append(match)

        logger.info(
            "detect_integration: %d integration events detected from %d high-inflow accounts",
            len(matches),
            len(high_inflow_accounts),
        )
        return matches

    def score(
        self,
        df: pd.DataFrame,
        sender_col: str = "sender_account_id",
        receiver_col: str = "receiver_account_id",
        amount_col: str = "amount_gbp",
        timestamp_col: str = "timestamp",
        txn_id_col: str = "transaction_id",
        country_origin_col: str | None = "country_origin",
        country_dest_col: str | None = "country_destination",
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Run all typology detectors and return per-transaction AML scores.

        Combines structuring, layering, and integration detection into a
        single composite AML risk score per transaction. Transactions not
        matched by any detector receive a score of 0.0.

        Args:
            df: Transaction DataFrame.
            sender_col: Sending account column.
            receiver_col: Receiving account column.
            amount_col: Amount column in GBP.
            timestamp_col: Timestamp column.
            txn_id_col: Transaction identifier column.
            country_origin_col: Origin country column. Pass None if absent.
            country_dest_col: Destination country column. Pass None if absent.
            weights: Optional weighting dict with keys "structuring", "layering",
                "integration". Values must sum to 1.0. Defaults to equal weights.

        Returns:
            DataFrame with columns: transaction_id, structuring_score,
            layering_score, integration_score, composite_score,
            dominant_typology, is_flagged. One row per input transaction.

        Raises:
            ValueError: If weights do not sum to 1.0 (within tolerance).
            KeyError: If required columns are absent from df.
        """
        if weights is None:
            weights = {"structuring": 1 / 3, "layering": 1 / 3, "integration": 1 / 3}

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"score: weights must sum to 1.0, got {total_weight:.6f}. " f"Weights: {weights}"
            )

        # Run all three detectors
        structuring_matches = self.detect_structuring(
            df,
            sender_col=sender_col,
            amount_col=amount_col,
            timestamp_col=timestamp_col,
            txn_id_col=txn_id_col,
        )
        layering_matches = self.detect_layering(
            df,
            sender_col=sender_col,
            receiver_col=receiver_col,
            amount_col=amount_col,
            timestamp_col=timestamp_col,
            txn_id_col=txn_id_col,
        )
        integration_matches = self.detect_integration(
            df,
            sender_col=sender_col,
            receiver_col=receiver_col,
            amount_col=amount_col,
            timestamp_col=timestamp_col,
            txn_id_col=txn_id_col,
            country_origin_col=country_origin_col,
            country_dest_col=country_dest_col,
        )

        # Build transaction-level score maps (take max confidence if a txn appears
        # in multiple matches of the same typology)
        struct_scores: dict[str, float] = {}
        for m in structuring_matches:
            for tid in m.transaction_ids:
                struct_scores[tid] = max(struct_scores.get(tid, 0.0), m.confidence)

        layer_scores: dict[str, float] = {}
        for m in layering_matches:
            for tid in m.transaction_ids:
                layer_scores[tid] = max(layer_scores.get(tid, 0.0), m.confidence)

        integ_scores: dict[str, float] = {}
        for m in integration_matches:
            for tid in m.transaction_ids:
                integ_scores[tid] = max(integ_scores.get(tid, 0.0), m.confidence)

        rows = []
        for txn_id in df[txn_id_col].astype(str):
            s = struct_scores.get(txn_id, 0.0)
            l = layer_scores.get(txn_id, 0.0)  # noqa: E741
            i = integ_scores.get(txn_id, 0.0)

            composite = round(
                weights["structuring"] * s + weights["layering"] * l + weights["integration"] * i,
                4,
            )

            scores_map = {"structuring": s, "layering": l, "integration": i}
            dominant = max(scores_map, key=lambda k: scores_map[k])
            if scores_map[dominant] == 0.0:
                dominant = "none"

            rows.append(
                {
                    "transaction_id": txn_id,
                    "structuring_score": round(s, 4),
                    "layering_score": round(l, 4),
                    "integration_score": round(i, 4),
                    "composite_score": composite,
                    "dominant_typology": dominant,
                    "is_flagged": composite >= self.flag_threshold,
                }
            )

        result = pd.DataFrame(rows)
        n_flagged = int(result["is_flagged"].sum())
        logger.info(
            "score: %d/%d transactions flagged (threshold=%.2f)",
            n_flagged,
            len(df),
            self.flag_threshold,
        )
        return result

    def matches_to_dataframe(self, matches: list[TypologyMatch]) -> pd.DataFrame:
        """Convert a list of TypologyMatch objects to a flat DataFrame.

        Useful for exporting detection results to a reporting database or
        MLRO review queue. Each row represents one typology cluster.

        Args:
            matches: List of TypologyMatch objects from any detect_* method.

        Returns:
            DataFrame with one row per match. Columns: account_id, typology,
            confidence, n_transactions, total_amount_gbp, window_start,
            window_end, transaction_ids (as a pipe-separated string).
        """
        if not matches:
            return pd.DataFrame(
                columns=[
                    "account_id",
                    "typology",
                    "confidence",
                    "n_transactions",
                    "total_amount_gbp",
                    "window_start",
                    "window_end",
                    "transaction_ids",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "account_id": m.account_id,
                    "typology": m.typology,
                    "confidence": m.confidence,
                    "n_transactions": m.n_transactions,
                    "total_amount_gbp": m.total_amount_gbp,
                    "window_start": m.window_start,
                    "window_end": m.window_end,
                    "transaction_ids": "|".join(m.transaction_ids),
                }
                for m in matches
            ]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"TypologyEngine: required columns missing from DataFrame: {missing}")
