# ADR 001 — Dual-domain architecture: fraud detection and AML as separate modules

**Status:** Accepted  
**Date:** 2024-01-02  
**Author:** Temidayo Akindahunsi  
**Deciders:** Core maintainers

---

## Context

Financial crime within banking and payments institutions manifests across two
distinct but related domains: **payment fraud** and **anti-money laundering (AML)**.
While both involve anomalous financial activity, they differ materially in:

- the supervisory frameworks that govern them,
- the data signals that predict them,
- the operational workflows that follow a detection event, and
- the model performance metrics that matter most.

This document records the decision to treat fraud and AML as separate pipeline
modules within FinCrime-ML, rather than a single unified detection model.

---

## Decision

FinCrime-ML implements fraud and AML as **independent pipeline domains** under
`fincrime_ml/fraud/` and `fincrime_ml/aml/`, with a **shared integration layer**
(`fincrime_ml/core/`) providing common base classes, data generators, and a
unified risk scorer that optionally combines both signals.

---

## Rationale

### 1. Regulatory separation

UK FCA rules treat fraud prevention (PSR 2017, PSD2) and AML/CFT (POCA 2002,
MLR 2017) as distinct compliance obligations with separate governance tracks.
A fraud alert triggers a block-and-reverse workflow. An AML alert triggers a
Suspicious Activity Report (SAR) submitted to the National Crime Agency (NCA)
under s.330 POCA 2002, with **tipping-off** constraints that fundamentally
change how the alert can be communicated.

Conflating the two in a single model would make compliance mapping ambiguous
and risk a tipping-off breach in production deployment.

### 2. Feature space divergence

Fraud detection relies primarily on **transaction-level features**: velocity
patterns, amount deviation from account baseline, merchant risk tier, channel
mismatch, and geographic anomaly within a single session.

AML monitoring relies on **network-level and temporal-aggregate features**:
beneficial ownership chains, remittance corridor risk, structuring patterns
(amounts clustered below reporting thresholds), layering through multiple
institutions, and integration into legitimate-appearing businesses.

A single feature engineering pipeline would require significant branching logic
to serve both domains, reducing maintainability and making the code harder to
validate — a concern explicitly raised in PRA SS1/23 model risk guidance.

### 3. Class imbalance characteristics differ

Fraud labels in card transaction datasets (IEEE-CIS, synthetic) typically sit
at 0.5–3% positive rate. AML ground truth is far sparser and less reliable —
SAR conversion rates in UK banking are estimated at below 1 in 1,000 alerts
(FATF Mutual Evaluation UK 2018). This drives different choices of:
- primary evaluation metric (AUC-PR for fraud; recall-at-threshold for AML),
- sampling strategy (SMOTE viable for fraud; isolation forest / unsupervised
  methods more appropriate when AML labels are absent or unreliable), and
- decision threshold (operational false-positive costs differ substantially).

### 4. Operational integration targets

Fraud pipelines typically need sub-second inference latency to support
real-time card authorisation decisioning. AML batch monitoring runs overnight
or intraday on complete transaction sets. These operational requirements drive
different architecture choices downstream (online vs batch scoring, feature
store requirements, model serving infrastructure).

---

## Alternatives considered

### A. Single unified model

A single multi-label classifier predicting both fraud and AML risk was
considered. Rejected because:
- The regulatory tipping-off constraint for AML makes a single alert workflow
  legally problematic.
- Feature importance patterns differ enough that a joint model would require
  multi-task learning architecture, adding significant complexity without
  clear performance benefit given the label sparsity asymmetry.

### B. Fully independent repositories

Maintaining fraud and AML as entirely separate repositories was considered.
Rejected because:
- Synthetic data generation, base classes, evaluation harness, and SQL schema
  are materially shared, creating duplication and version drift risk.
- A unified `core/` module provides a clean separation of concerns without
  requiring consumers to install two separate packages.

---

## Consequences

- All fraud pipeline classes must inherit from `fincrime_ml.core.base.BasePipeline`
  with `label_col="is_fraud"`.
- All AML pipeline classes must inherit from `BasePipeline` with
  `label_col="is_suspicious"`.
- The unified scorer in `core/scorer.py` accepts both pipeline outputs and
  produces a fused risk score — this is additive functionality, not a
  replacement for domain-specific alerting.
- Documentation, notebooks, and tests must be organised by domain. No
  cross-domain imports from `fraud/` into `aml/` or vice versa — only
  imports from `core/` are permitted across domain boundaries.
- The regulatory alignment guide (`docs/regulatory.md`) must map each module
  to its governing framework explicitly.

---

## References

- FCA: [SYSC 6.3 — Financial crime](https://www.handbook.fca.org.uk/handbook/SYSC/6/3.html)
- FCA / PSR: [Payment Services Regulations 2017](https://www.legislation.gov.uk/uksi/2017/752/contents)
- HM Treasury: [Proceeds of Crime Act 2002 s.330 — Failure to disclose](https://www.legislation.gov.uk/ukpga/2002/29/section/330)
- HM Treasury: [Money Laundering Regulations 2017](https://www.legislation.gov.uk/uksi/2017/692/contents)
- PRA: [SS1/23 — Model risk management principles](https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss)
- FATF: [40 Recommendations](https://www.fatf-gafi.org/en/topics/fatf-recommendations.html)
- Federal Reserve / OCC: [SR 11-7 — Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- UK Finance: [Annual Fraud Report 2023](https://www.ukfinance.org.uk/policy-and-guidance/reports-and-publications)
- FATF: [Mutual Evaluation Report — United Kingdom, 2018](https://www.fatf-gafi.org/en/publications/Mutualevaluations/Mer-united-kingdom-2018.html)
