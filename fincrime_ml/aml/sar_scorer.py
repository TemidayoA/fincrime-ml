"""
aml/sar_scorer.py
==================
SAR trigger scoring — alert prioritisation with MLRO-ready audit output.

Purpose
    ML risk scores and rule-based typology flags must be translated into
    actionable alerts before they reach an MLRO (Money Laundering Reporting
    Officer). This module bridges the gap: it ingests transaction-level risk
    scores (from GraphScorer, AMLIsolationForest, or any scoring pipeline),
    applies SAR trigger rules aligned to JMLSG Part I Ch.5, and returns a
    prioritised alert queue with regulatory references and MLRO-ready
    summary text.

    The output is deliberately structured for operational use: each alert
    row contains everything an MLRO needs to decide whether to file a SAR
    under POCA 2002 s.330, without needing to cross-reference the underlying
    model outputs.

SAR trigger rules
    Each rule is evaluated independently; a transaction is escalated to the
    alert queue if its risk_score meets the minimum threshold. Multiple
    matched rules escalate the priority level.

    ==================== ================================================
    Trigger              Condition
    ==================== ================================================
    HIGH_RISK_SCORE      risk_score >= sar_score_threshold (default 0.65)
    STRUCTURING_AMOUNT   amount_gbp in [8500, 9950] (POCA 2002 s.330)
    MULE_INVOLVEMENT     is_mule_sender=1 or is_mule_receiver=1
    RAPID_MOVEMENT       rapid_movement_flag=1
    CHAIN_LAYERING       layering_depth > 0
    SUSPICIOUS_TYPOLOGY  typology in {structuring, layering, integration}
    ==================== ================================================

Priority levels
    1 — CRITICAL: risk_tier=CRITICAL, or 3+ trigger rules matched.
        Immediate MLRO referral; SAR filing recommended.
    2 — HIGH: risk_tier=HIGH, or 2 trigger rules matched.
        Review within 24 hours; SAR filing considered.
    3 — MEDIUM: risk_tier=MEDIUM, or 1 trigger rule matched.
        Review within 5 business days; enhanced monitoring.

Regulatory alignment
    POCA 2002 s.330
        Failure to disclose — criminal offence for regulated firms that
        know or suspect money laundering and fail to report. The SAR
        recommendation field directly maps to this obligation.

    JMLSG Part I Ch.5 para 5.3.1-5.3.17
        Transaction monitoring indicators — each trigger rule maps to a
        specific JMLSG paragraph documented in ``TRIGGER_REGULATORY_REFS``.

    FATF Recommendations R.10, R.16, R.20
        R.10: customer due diligence; R.16: wire transfer transparency;
        R.20: suspicious transaction reporting obligation.

    MLR 2017 Reg 28
        Enhanced Due Diligence requirements. Mule involvement and
        high-risk-score alerts feed EDD review workflows.

    FCA SYSC 10A
        Record-keeping for automated decision systems. The ``audit_log``
        property provides a timestamped, immutable record of every alert
        batch produced by this scorer instance.

Architecture note
    SARScorer is a rule-based orchestration layer — it does not train or
    predict in the ML sense, and therefore does not inherit BasePipeline
    or BaseScorer. It maintains its own audit log conforming to the same
    FCA SYSC 10A structure used by BasePipeline._log_audit().

    Imports only from fincrime_ml.aml and fincrime_ml.core. No imports
    from fincrime_ml.fraud permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_int(val: Any, default: int = 0) -> int:
    """Convert *val* to int, returning *default* for None or NaN values.

    Used when reading optional flag columns from a Pandas Series where the
    column may be absent (defaulting to NaN via mixed-column DataFrames).
    """
    try:
        fv = float(val)
        return default if math.isnan(fv) else int(fv)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Regulatory reference table — maps each trigger to its statutory basis
# ---------------------------------------------------------------------------

#: Per-trigger regulatory references included in MLRO alert output.
TRIGGER_REGULATORY_REFS: dict[str, str] = {
    "HIGH_RISK_SCORE": "JMLSG Part I para 5.3.1; FCA FCG 3.2",
    "STRUCTURING_AMOUNT": "POCA 2002 s.330; JMLSG Part I para 5.3.11",
    "MULE_INVOLVEMENT": "MLR 2017 Reg 28; JMLSG Part I para 5.3.17",
    "RAPID_MOVEMENT": "FATF R.10; JMLSG Part I para 5.3.7",
    "CHAIN_LAYERING": "FATF R.16; JMLSG Part I para 5.3.17",
    "SUSPICIOUS_TYPOLOGY": "FATF 40 Recommendations R.20; JMLSG Part I Ch.5",
}

#: Typology values that trigger SUSPICIOUS_TYPOLOGY rule.
SUSPICIOUS_TYPOLOGIES: frozenset[str] = frozenset({"structuring", "layering", "integration"})

#: Structuring amount band (POCA 2002 s.330 threshold avoidance window).
STRUCTURING_LOWER_GBP: float = 8_500.0
STRUCTURING_UPPER_GBP: float = 9_950.0

#: Columns produced in the alert queue DataFrame.
SAR_ALERT_COLS: list[str] = [
    "alert_id",
    "transaction_id",
    "risk_score",
    "risk_tier",
    "priority",
    "n_triggers",
    "trigger_reasons",
    "sar_recommended",
    "regulatory_refs",
    "mlro_summary",
    "amount_gbp",
    "typology",
    "scored_at",
]


@dataclass
class SARScorerConfig:
    """Configuration for the SAR trigger scorer.

    Attributes:
        alert_score_threshold: Minimum risk_score to include a transaction
            in the alert queue at all. Below this, transactions are silently
            excluded (default: 0.30 = MEDIUM tier floor).
        sar_score_threshold: Minimum risk_score to fire the HIGH_RISK_SCORE
            trigger and recommend SAR filing (default: 0.65 = HIGH tier).
        structuring_lower: Lower bound of the structuring amount band (GBP).
        structuring_upper: Upper bound of the structuring amount band (GBP).
        version: Scorer version string, included in audit log entries.
        audit_log_enabled: Whether to write a decision audit trail.
    """

    alert_score_threshold: float = 0.30
    sar_score_threshold: float = 0.65
    structuring_lower: float = STRUCTURING_LOWER_GBP
    structuring_upper: float = STRUCTURING_UPPER_GBP
    version: str = "0.1.0"
    audit_log_enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class SARScorer:
    """SAR trigger scorer — prioritised alert queue with MLRO-ready output.

    Evaluates a set of SAR trigger rules against transaction risk scores and
    returns a prioritised alert DataFrame. Each alert row contains the trigger
    reasons, mapped regulatory references, a plain-English MLRO summary, and
    a SAR filing recommendation.

    The scorer is stateless between calls to ``score()`` — it can be called
    repeatedly on different batches without re-initialisation.

    Example::

        from fincrime_ml.aml.sar_scorer import SARScorer, SARScorerConfig

        # Assume scored_df is the output of GraphScorer.predict() merged
        # with the original transaction DataFrame
        scorer = SARScorer()
        alerts = scorer.score(scored_df)

        # Review top priority alerts
        critical = alerts[alerts["priority"] == 1]
        print(critical[["alert_id", "trigger_reasons", "mlro_summary"]])

    Attributes:
        config: SARScorerConfig instance.
    """

    def __init__(self, config: SARScorerConfig | None = None) -> None:
        self.config = config or SARScorerConfig()
        self._audit_log: list[dict] = []
        logger.info(
            "SARScorer initialised v%s (alert_threshold=%.2f, sar_threshold=%.2f)",
            self.config.version,
            self.config.alert_score_threshold,
            self.config.sar_score_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SAR trigger rules and return a prioritised alert queue.

        Evaluates all six trigger rules against each transaction. Transactions
        whose ``risk_score`` falls below ``config.alert_score_threshold`` are
        excluded from the output. All retained transactions receive an
        ``alert_id``, priority level, trigger reasons, regulatory references,
        and MLRO summary text.

        Input DataFrame must contain ``risk_score`` and ``risk_tier``. All
        other columns (``amount_gbp``, ``typology``, ``structuring_flag``,
        ``rapid_movement_flag``, ``layering_depth``, ``is_mule_sender``,
        ``is_mule_receiver``) are used when present and defaulted to neutral
        values when absent.

        Args:
            df: Transaction DataFrame with at minimum ``risk_score`` and
                ``risk_tier`` columns.

        Returns:
            Alert queue DataFrame with columns defined in ``SAR_ALERT_COLS``,
            sorted by priority ascending then risk_score descending
            (highest-priority, highest-risk alerts first).

        Raises:
            KeyError: If ``risk_score`` or ``risk_tier`` is absent from df.
        """
        self._check_required_columns(df)

        alerts = []
        scored_at = datetime.utcnow().isoformat()

        for idx, row in df.iterrows():
            risk_score = float(row["risk_score"])
            if risk_score < self.config.alert_score_threshold:
                continue

            triggers = self._evaluate_triggers(row, risk_score)
            if not triggers:
                continue

            priority = self._assign_priority(row["risk_tier"], len(triggers))
            sar_recommended = int(risk_score >= self.config.sar_score_threshold or priority == 1)
            reg_refs = self._collect_regulatory_refs(triggers)
            mlro_summary = self._build_mlro_summary(row, risk_score, triggers, priority)

            alerts.append(
                {
                    "alert_id": f"SAR-{uuid.uuid4().hex[:12].upper()}",
                    "transaction_id": row.get("transaction_id", str(idx)),
                    "risk_score": round(risk_score, 4),
                    "risk_tier": str(row["risk_tier"]),
                    "priority": priority,
                    "n_triggers": len(triggers),
                    "trigger_reasons": "|".join(triggers),
                    "sar_recommended": sar_recommended,
                    "regulatory_refs": reg_refs,
                    "mlro_summary": mlro_summary,
                    "amount_gbp": float(row.get("amount_gbp", 0.0)),
                    "typology": str(row.get("typology", "unknown")),
                    "scored_at": scored_at,
                }
            )

        if not alerts:
            result = pd.DataFrame(columns=SAR_ALERT_COLS)
        else:
            result = (
                pd.DataFrame(alerts)[SAR_ALERT_COLS]
                .sort_values(
                    ["priority", "risk_score"],
                    ascending=[True, False],
                )
                .reset_index(drop=True)
            )

        n_sar = int(result["sar_recommended"].sum()) if len(result) else 0
        self._log_audit(
            "score",
            {
                "n_input_transactions": len(df),
                "n_alerts_generated": len(result),
                "n_sar_recommended": n_sar,
                "n_priority_1": int((result["priority"] == 1).sum()) if len(result) else 0,
            },
        )
        logger.info(
            "SARScorer.score: %d transactions → %d alerts (%d SAR recommended, %d P1)",
            len(df),
            len(result),
            n_sar,
            int((result["priority"] == 1).sum()) if len(result) else 0,
        )
        return result

    def summary_report(self, alerts: pd.DataFrame) -> dict[str, Any]:
        """Produce a summary statistics report over a scored alert queue.

        Suitable for management information (MI) reporting to MLRO or
        compliance committee. Covers alert volume, SAR recommendation rate,
        priority breakdown, and top trigger frequencies.

        Args:
            alerts: Output DataFrame from ``score()``.

        Returns:
            Dict with keys: n_alerts, n_sar_recommended, sar_rate,
            priority_counts, top_triggers, mean_risk_score.
        """
        if len(alerts) == 0:
            return {
                "n_alerts": 0,
                "n_sar_recommended": 0,
                "sar_rate": 0.0,
                "priority_counts": {1: 0, 2: 0, 3: 0},
                "top_triggers": {},
                "mean_risk_score": 0.0,
            }

        # Trigger frequency across all alerts
        all_triggers: list[str] = []
        for reasons in alerts["trigger_reasons"]:
            all_triggers.extend(reasons.split("|"))
        trigger_counts: dict[str, int] = {}
        for t in all_triggers:
            trigger_counts[t] = trigger_counts.get(t, 0) + 1
        top_triggers = dict(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True))

        return {
            "n_alerts": len(alerts),
            "n_sar_recommended": int(alerts["sar_recommended"].sum()),
            "sar_rate": round(float(alerts["sar_recommended"].mean()), 4),
            "priority_counts": {p: int((alerts["priority"] == p).sum()) for p in (1, 2, 3)},
            "top_triggers": top_triggers,
            "mean_risk_score": round(float(alerts["risk_score"].mean()), 4),
        }

    @property
    def audit_log(self) -> list[dict]:
        """Return the immutable audit log for this scorer instance."""
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # Private: trigger evaluation
    # ------------------------------------------------------------------

    def _evaluate_triggers(self, row: Any, risk_score: float) -> list[str]:
        """Evaluate all SAR trigger rules against a single transaction row.

        Args:
            row: Pandas Series (one transaction row).
            risk_score: Pre-extracted float risk score.

        Returns:
            List of triggered rule names, in a stable order.
        """
        triggers: list[str] = []

        # Rule 1: high risk score
        if risk_score >= self.config.sar_score_threshold:
            triggers.append("HIGH_RISK_SCORE")

        # Rule 2: structuring amount band (POCA 2002 s.330)
        amount = float(row.get("amount_gbp", 0.0))
        if self.config.structuring_lower <= amount <= self.config.structuring_upper:
            triggers.append("STRUCTURING_AMOUNT")

        # Rule 3: mule account involvement
        is_mule_sender = _safe_int(row.get("is_mule_sender", 0))
        is_mule_receiver = _safe_int(row.get("is_mule_receiver", 0))
        if is_mule_sender or is_mule_receiver:
            triggers.append("MULE_INVOLVEMENT")

        # Rule 4: rapid fund movement (FATF R.10)
        if _safe_int(row.get("rapid_movement_flag", 0)):
            triggers.append("RAPID_MOVEMENT")

        # Rule 5: chain layering
        if _safe_int(row.get("layering_depth", 0)) > 0:
            triggers.append("CHAIN_LAYERING")

        # Rule 6: known suspicious typology (FATF R.20)
        typology = str(row.get("typology", ""))
        if typology in SUSPICIOUS_TYPOLOGIES:
            triggers.append("SUSPICIOUS_TYPOLOGY")

        return triggers

    # ------------------------------------------------------------------
    # Private: priority and output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_priority(risk_tier: str, n_triggers: int) -> int:
        """Map risk tier and trigger count to a 1/2/3 priority level.

        Args:
            risk_tier: Risk tier string (CRITICAL/HIGH/MEDIUM/LOW).
            n_triggers: Number of SAR trigger rules matched.

        Returns:
            Priority integer: 1=Critical, 2=High, 3=Medium.
        """
        if risk_tier == "CRITICAL" or n_triggers >= 3:
            return 1
        if risk_tier == "HIGH" or n_triggers >= 2:
            return 2
        return 3

    @staticmethod
    def _collect_regulatory_refs(triggers: list[str]) -> str:
        """Collect unique regulatory references for all matched triggers.

        Args:
            triggers: List of trigger rule names.

        Returns:
            Pipe-separated string of unique regulatory references.
        """
        refs: list[str] = []
        seen: set[str] = set()
        for trigger in triggers:
            ref = TRIGGER_REGULATORY_REFS.get(trigger, "")
            if ref and ref not in seen:
                refs.append(ref)
                seen.add(ref)
        return " | ".join(refs)

    @staticmethod
    def _build_mlro_summary(
        row: Any,
        risk_score: float,
        triggers: list[str],
        priority: int,
    ) -> str:
        """Build a plain-English MLRO summary for an alert.

        Produces a concise, structured summary suitable for insertion into
        a SAR workflow system or compliance case management tool. The text
        format is deterministic — identical inputs produce identical output —
        so it can be stored and compared across runs.

        Args:
            row: Pandas Series (one transaction row).
            risk_score: Normalised risk score.
            triggers: List of matched trigger rule names.
            priority: Assigned priority level.

        Returns:
            Formatted plain-English summary string.
        """
        txn_id = row.get("transaction_id", "UNKNOWN")
        amount = float(row.get("amount_gbp", 0.0))
        typology = str(row.get("typology", "unknown"))
        risk_tier = str(row.get("risk_tier", "UNKNOWN"))

        priority_label = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM"}.get(priority, "MEDIUM")
        trigger_text = ", ".join(t.replace("_", " ").title() for t in triggers)

        lines = [
            f"Alert Priority {priority} ({priority_label})",
            f"Transaction: {txn_id} | Amount: GBP {amount:,.2f} | "
            f"Risk Score: {risk_score:.4f} ({risk_tier})",
            f"Typology: {typology.title()}",
            f"Trigger rules matched ({len(triggers)}): {trigger_text}",
        ]

        if "STRUCTURING_AMOUNT" in triggers:
            lines.append(
                f"Structuring indicator: amount GBP {amount:,.2f} falls within "
                "the POCA 2002 s.330 threshold avoidance band (GBP 8,500-9,950)."
            )
        if "MULE_INVOLVEMENT" in triggers:
            lines.append(
                "Mule account involvement detected on sender or receiver. "
                "Enhanced Due Diligence required under MLR 2017 Reg 28."
            )
        if "RAPID_MOVEMENT" in triggers:
            lines.append(
                "Rapid fund movement detected: funds transferred within 2 hours "
                "of receipt. FATF R.10 layering indicator."
            )
        if "CHAIN_LAYERING" in triggers:
            lines.append(
                "Transaction is part of a multi-hop mule chain. "
                "FATF R.16 wire transfer transparency applies."
            )

        return " | ".join(lines)

    # ------------------------------------------------------------------
    # Private: validation and audit
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required_columns(df: pd.DataFrame) -> None:
        """Raise KeyError if risk_score or risk_tier are absent.

        Args:
            df: Input DataFrame to validate.

        Raises:
            KeyError: If any required column is missing.
        """
        required = {"risk_score", "risk_tier"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"SARScorer.score: required columns missing from DataFrame: {sorted(missing)}"
            )

    def _log_audit(self, event: str, metadata: dict | None = None) -> None:
        """Append an entry to the in-memory audit log.

        Conforms to the same structure used by BasePipeline._log_audit() so
        audit records are uniform across the FinCrime-ML framework.

        Args:
            event: Human-readable event description.
            metadata: Optional additional context.
        """
        if not self.config.audit_log_enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "scorer": self.__class__.__name__,
            "version": self.config.version,
            "event": event,
            **(metadata or {}),
        }
        self._audit_log.append(entry)
        logger.debug("Audit: %s", entry)
