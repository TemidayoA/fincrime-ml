# Regulatory Alignment Guide

**Document reference:** RA-2026-001
**Version:** 1.0
**Date:** May 2026
**Classification:** Internal — Portfolio Reference

---

## 1. Introduction

This document maps every module in the FinCrime-ML framework to the specific regulatory
obligations that informed its design. The framework operates across two distinct supervisory
regimes: the fraud domain is governed primarily by the Payment Services Regulations 2017
(PSR 2017) and PSD2, whilst the AML domain falls under the Proceeds of Crime Act 2002
(POCA 2002), the Money Laundering Regulations 2017 (MLR 2017), and the Joint Money
Laundering Steering Group (JMLSG) guidance. The core infrastructure layer is subject to
the Financial Conduct Authority's Senior Management Arrangements, Systems and Controls
sourcebook (SYSC) and the Prudential Regulation Authority's model risk supervisory statement
SS1/23.

The purpose of this guide is threefold: to demonstrate that design decisions were not made
in an engineering vacuum; to enable compliance colleagues to trace any system capability
back to a specific rule or supervisory expectation; and to support model validation activity
by providing a coherent account of the regulatory rationale behind each threshold, metric,
and workflow.

---

## 2. Regulatory Framework Summary

| Instrument | Jurisdiction | Key Obligations Relevant to This Framework |
|---|---|---|
| POCA 2002 ss.330–332 | UK | SAR filing obligation; mandatory disclosure to NCA |
| MLR 2017 Regs 19, 28, 40 | UK | Staff oversight; EDD; five-year record retention |
| JMLSG Part I Ch.5 | UK (guidance) | Transaction monitoring indicators; typology coverage |
| PSR 2017 / PSD2 | UK/EU | Fraud detection obligation for payment service providers |
| FCA SYSC 6.3 | UK | Transaction monitoring system design and effectiveness |
| FCA SYSC 10A | UK | Automated decision audit trail requirements |
| FCA FCG 3.2 | UK | Financial crime systems and controls guidance |
| PRA SS1/23 | UK | Model risk management; champion/challenger; governance |
| FATF Recommendations R.10, R.16, R.20 | International | CDD; wire transfer monitoring; suspicious transaction reporting |
| Basel Committee BCBS 239 | International | Risk data aggregation; reporting accuracy |

---

## 3. Fraud Domain Modules

### 3.1 `fincrime_ml/fraud/features.py` — Feature Engineering

**Primary obligation:** PSR 2017 Reg 98 requires payment service providers to maintain
effective fraud detection measures proportionate to the risk.

**Regulatory mapping:**

The velocity window features (transaction count and amount sum over 1h, 6h, 24h, 7d, 30d)
directly implement the monitoring indicators enumerated in FCA FCG 3.2, which requires
institutions to monitor for unusual frequency and volume of transactions relative to
established customer behaviour. The 30-day baseline window is used to compute z-score
deviations; this approach is consistent with JMLSG Part I para 5.3.7, which identifies
unusual velocity as a primary detection indicator.

The merchant category code (MCC) risk scoring implements risk-based customer due diligence
profiling as required under MLR 2017 Reg 28, applied here to transaction-level risk rather
than customer-level EDD. Hour-of-day and day-of-week features capture temporal anomalies
consistent with JMLSG para 5.3.9 (transactions outside normal business hours).

### 3.2 `fincrime_ml/fraud/models/xgb_classifier.py` — XGBoost Classifier

**Primary obligation:** PRA SS1/23 s.3 requires that model design choices be documented
and that performance be monitored against pre-agreed thresholds on a regular basis.

**Regulatory mapping:**

AUC-PR is used as the primary performance metric in preference to ROC-AUC because the
dataset is highly imbalanced (fraud prevalence approximately 1–3 percent). This choice is
consistent with PRA SS1/23's expectation that performance metrics be appropriate to the
specific use case and not misleading. The model uses five-fold stratified cross-validation
to ensure that fraud prevalence is maintained across folds, which is a requirement of
robust model validation under SS1/23 s.4.

The champion/challenger structure (XGBoost as champion, logistic regression as challenger)
implements the supervisory expectation in PRA SS1/23 s.5 that institutions maintain a
comparison baseline and have a documented promotion and demotion process.

### 3.3 `fincrime_ml/fraud/models/logistic_baseline.py` — Logistic Baseline

**Primary obligation:** PRA SS1/23 s.5 — challenger model requirement.

**Regulatory mapping:**

A logistic regression baseline serves as the challenger model and also provides an
interpretable benchmark against which the XGBoost champion can be compared. The linear
decision boundary makes coefficients directly auditable, supporting the FCA's expectation
under SYSC 10A that automated decisions be explainable to regulators on request.

### 3.4 `fincrime_ml/fraud/explain.py` — SHAP Explainability

**Primary obligation:** FCA SR11-7 (US origin, adopted as best practice by FCA); FCA
Discussion Paper DP5/22 on AI and Machine Learning in Financial Services.

**Regulatory mapping:**

SHAP (SHapley Additive exPlanations) values are computed for every scored transaction to
produce per-prediction reason codes. This supports three distinct regulatory requirements.
First, SYSC 10A requires that automated decisions be accompanied by a documented audit
trail that can be presented to regulators; the `shap_json` column in the `fraud_scores`
table stores the full SHAP vector for this purpose. Second, the FCA's expectations around
model explainability in DP5/22 require that institutions be able to articulate why a
particular transaction was flagged; the top-three reason code columns (`top_reason_1` to
`top_reason_3`) are designed for direct use in analyst-facing alert interfaces. Third,
consumer protection obligations under the Consumer Duty (FCA PS22/9) require that adverse
decisions (such as payment blocking) be explainable to the customer if challenged.

### 3.5 `fincrime_ml/fraud/imbalance.py` — Class Imbalance Handler

**Primary obligation:** PRA SS1/23 s.3 — model design documentation.

**Regulatory mapping:**

SMOTE oversampling and cost-sensitive weighting are offered as alternatives. The module
benchmarks both approaches and selects based on AUC-PR improvement. The regulatory
rationale is that an unaddressed class imbalance would produce a model that appears
performant on overall accuracy whilst generating an unacceptably high false-negative rate
on the minority fraud class, which would constitute a failure of the institution's fraud
detection obligation under PSR 2017 Reg 98.

### 3.6 `fincrime_ml/fraud/evaluation.py` — Fraud Evaluation Suite

**Primary obligation:** PRA SS1/23 s.4 — model performance monitoring.

**Regulatory mapping:**

The threshold analysis and false positive cost matrix implement the operational review
requirements of PRA SS1/23. The cost matrix makes explicit the asymmetry between the
cost of a missed fraud (false negative — direct financial loss plus potential regulatory
sanction) and the cost of a false positive (customer friction, potential Consumer Duty
breach). Documenting this trade-off is a supervisory expectation under SS1/23 s.3.

---

## 4. AML Domain Modules

### 4.1 `fincrime_ml/aml/typologies.py` — Typology Engine

**Primary obligation:** JMLSG Part I para 5.3 requires that transaction monitoring systems
be calibrated to detect the typologies prevalent in the firm's specific business and
customer base.

**Regulatory mapping:**

The four typologies implemented (structuring, layering, integration, mule account activity)
correspond directly to the categories enumerated in JMLSG Part I paras 5.3.11, 5.3.13,
5.3.15, and 5.3.16 respectively. The structuring detection threshold of GBP 8,500 to
GBP 9,950 reflects the POCA 2002 s.330 reporting band; transactions deliberately clustered
just below GBP 10,000 indicate an intent to avoid the mandatory disclosure threshold.

FATF Recommendation 20 requires that jurisdictions ensure suspicious transaction reports
are filed where there are reasonable grounds to suspect that funds are the proceeds of
crime. The typology engine produces the `is_suspicious` label that feeds the SAR
recommendation logic in `sar_scorer.py`, creating a direct link from pattern detection
to the POCA 2002 s.330 disclosure obligation.

### 4.2 `fincrime_ml/aml/graph.py` — Transaction Network Graph Builder

**Primary obligation:** FATF R.10 (customer due diligence on unusual patterns); FATF
Guidance on Digital Identity (2020) — network-based entity resolution.

**Regulatory mapping:**

Graph-based anomaly scoring captures network-level risk that is invisible to
transaction-level models. Centrality metrics (degree, betweenness, PageRank) identify
accounts that function as hubs in a money movement network, which is a primary indicator
of mule account activity and layering structures under JMLSG para 5.3.15. The pass-through
ratio (ratio of outbound to inbound value) detects accounts used purely for value
transmission, which FATF R.10 identifies as a basis for enhanced due diligence.

The FCA's Financial Crime Guide (FCG 3.2) states that institutions should use all available
data to build a complete picture of customer behaviour. Graph analytics over the full
transaction network, rather than account-level analysis in isolation, directly satisfies
this expectation.

### 4.3 `fincrime_ml/aml/models/isolation_forest.py` — AML Isolation Forest

**Primary obligation:** JMLSG Part I para 5.3.1 — monitoring system design; MLR 2017
Reg 19 — systems and controls proportionate to risk.

**Regulatory mapping:**

The Isolation Forest is deployed as an unsupervised anomaly detector for the AML domain
because labelled AML data is sparse and subject to survivorship bias (only previously
detected cases carry confirmed labels). This design choice is documented in the model
card in accordance with PRA SS1/23 s.3, which requires that the absence of labels and
the consequent reliance on unsupervised methods be explicitly acknowledged and the
limitations assessed.

MLR 2017 Reg 19 requires that monitoring systems be calibrated so that the volume of
alerts generated does not exceed the firm's review capacity. The contamination parameter
controls alert volume; the alert fatigue evaluation module (`evaluation.py`) provides
the FPR-at-sensitivity analysis needed to set this parameter in a defensible,
regulatorily documented manner.

### 4.4 `fincrime_ml/aml/sar_scorer.py` — SAR Trigger Scorer

**Primary obligation:** POCA 2002 s.330 — the primary obligation to disclose knowledge
or suspicion of money laundering to the National Crime Agency.

**Regulatory mapping:**

This is the most directly regulated module in the framework. Section 330 of POCA 2002
creates a criminal offence for a person in the regulated sector who knows or suspects,
or has reasonable grounds for knowing or suspecting, that another person is engaged in
money laundering, and who fails to disclose that knowledge or suspicion to the NCA.

The six trigger rules implemented in `_evaluate_triggers()` map to specific JMLSG
indicators: `STRUCTURING_AMOUNT` (para 5.3.11), `RAPID_MOVEMENT` (para 5.3.13),
`HIGH_RISK_TYPOLOGY` (para 5.3.15), `MULE_INVOLVEMENT` (para 5.3.16),
`CROSS_BORDER_HIGH_RISK` (FATF R.16), and `VELOCITY_SPIKE` (para 5.3.7). The
three-tier priority system (CRITICAL / HIGH / MEDIUM) is calibrated so that Priority 1
alerts require MLRO review within 24 hours, consistent with the firm's documented
policy aligned to JMLSG Part I Ch.5.

The `regulatory_refs` field on each alert surfaces the specific POCA and JMLSG references
that apply to that alert, enabling the MLRO to document the legal basis for any SAR filing
without additional research. The `mlro_summary` field provides a plain-English narrative
suitable for direct inclusion in a SAR submission to the NCA.

### 4.5 `fincrime_ml/aml/evaluation.py` — Alert Fatigue Evaluator

**Primary obligation:** MLR 2017 Reg 19 — requirement that the number of alerts generated
be manageable relative to available staff resource; FCA SYSC 6.3 — monitoring system
effectiveness review.

**Regulatory mapping:**

The alert fatigue index (1 minus precision at a given operating threshold) quantifies the
proportion of analyst time consumed by false positives. FCA SYSC 6.3.6R requires that
firms regularly review the effectiveness of their transaction monitoring arrangements;
the `evaluate()` method produces the metric suite needed for this review, including FPR
at five sensitivity targets (80, 85, 90, 95, 99 percent), the fatigue index profile, and
the precision-recall AUC.

MLR 2017 Reg 19 is operationalised through the analyst workload calculation in the model
validation report (MV-2026-001, s.5.2), which derives required FTE from alert volume at
a given sensitivity target. This creates a documented link between model threshold choice
and staffing adequacy, which is a direct input to the compliance committee MI pack.

---

## 5. Core Infrastructure Modules

### 5.1 `fincrime_ml/core/scorer.py` — FinCrime Unified Scorer

**Primary obligation:** FCA SYSC 6.3 — requires that firms maintain a coherent, integrated
view of financial crime risk; PRA SS1/23 s.3 — model design documentation.

**Regulatory mapping:**

The three fusion strategies (weighted average, maximum, harmonic mean) each represent a
different supervisory posture. The weighted average strategy allows the institution to
reflect its specific risk appetite by allocating more weight to whichever domain presents
a higher regulatory risk (for instance, upweighting AML signals where the customer base
has higher ML exposure). The maximum strategy implements a conservative, risk-averse
posture in which either domain signal alone is sufficient to elevate the unified score;
this is appropriate where the regulatory cost of a missed alert outweighs the operational
cost of elevated false positive volumes. The harmonic mean strategy penalises cases where
one signal is near zero, producing a conservative score only where both domains indicate
risk.

The `FusionConfig` dataclass documents the chosen strategy and weights in a form that
is directly auditable by model validators and regulators, consistent with SYSC 10A.

### 5.2 `sql/schema.sql` — Transaction Monitoring Schema

**Primary obligation:** FCA SYSC 6.3; MLR 2017 Reg 40 — five-year record retention;
FCA SYSC 10A — audit trail.

**Regulatory mapping:**

MLR 2017 Reg 40(1)(a) requires that records of customer due diligence measures be retained
for five years from the end of the business relationship. The schema does not implement
physical deletion (`deleted_at` soft-delete pattern) to ensure that records cannot be
inadvertently removed before the retention period expires. The `audit_log` table carries
a comment explicitly stating that `UPDATE` and `DELETE` operations are not permitted;
this comment reflects the SYSC 10A requirement for an immutable audit trail of automated
decisions.

The `sar_referrals` table implements the consent regime under POCA 2002 ss.335–336, which
requires that the NCA either grant consent or allow the seven-day moratorium period to
expire before a transaction can proceed where an authorised disclosure has been made. The
`filing_status` column tracks the consent workflow states (PENDING, SUBMITTED,
CONSENT_REQUESTED, CONSENT_GRANTED, REFUSED).

### 5.3 `sql/queries/velocity.sql` — Velocity Window Queries

**Primary obligation:** JMLSG Part I para 5.3.7 — unusual transaction velocity; FATF R.16
— cross-border wire transfer monitoring.

**Regulatory mapping:**

The parameterised time windows (1h, 6h, 24h, 7d, 30d) map to the JMLSG indicators for
short-term spikes (intraday), medium-term patterns (weekly), and baseline deviation
(monthly). Query Q3 implements POCA 2002 s.330 structuring detection over the GBP 8,500
to GBP 9,950 band. Query Q4 implements FATF R.10 rapid-movement (layering) detection
through a self-join that identifies receive-then-send patterns within a two-hour window.
Query Q5 implements FATF R.16 cross-border monitoring. All queries use bind parameters
(`:account_id`, `:as_of_datetime`) to prevent SQL injection, consistent with secure coding
standards.

### 5.4 `fincrime_ml/core/data/` — Synthetic Data Generators

**Primary obligation:** PRA SS1/23 s.4 — model validation requires representative test
data; GDPR Art. 25 — data protection by design.

**Regulatory mapping:**

Production transaction data cannot be used for model development and testing without
appropriate privacy engineering controls. The synthetic data generators produce statistically
representative transaction populations without containing personal data attributable to
real individuals, which satisfies the data protection by design requirement under GDPR
Art. 25 and the data minimisation principle under Art. 5(1)(c). The generators reproduce
the statistical properties of fraud and AML typologies documented in JMLSG Part I Ch.5,
ensuring that models trained on synthetic data generalise to production distributions.

---

## 6. Audit Trail and Record-Keeping

### 6.1 FCA SYSC 10A Compliance

FCA SYSC 10A.1.6R requires that firms using automated or semi-automated decision-making
systems maintain a record of decisions made by those systems in a form that can be
reproduced and audited. Every scoring event in FinCrime-ML writes a record to the
`audit_log` table, capturing: the scorer class name; the model version; the event type
(train, predict, score, explain, evaluate); the transaction identifier where applicable;
the number of records processed; and a JSON metadata payload containing any additional
context relevant to the event. The `event_id` column is generated as a UUID, providing
a globally unique reference that can be cited in supervisory correspondence.

### 6.2 MLR 2017 Reg 40 Record Retention

All tables use `created_at` timestamps and, where applicable, `updated_at` with
`ON UPDATE CURRENT_TIMESTAMP`. No hard deletes are permitted on any table that contains
transaction monitoring records; the soft-delete pattern (`deleted_at IS NULL = active`)
ensures that the five-year retention obligation can be met without architectural changes
to the application layer.

---

## 7. Model Risk Governance (PRA SS1/23)

PRA SS1/23 establishes a three-tier model risk classification (Tier 1 critical, Tier 2
material, Tier 3 non-material). The FinCrime-ML models are classified as follows:

| Model | Classification | Rationale |
|---|---|---|
| XGBFraudClassifier (champion) | Tier 2 — Material | Direct input to payment blocking decisions |
| LogisticFraudBaseline (challenger) | Tier 3 — Non-material | Benchmark only; not used in production decisions |
| AMLIsolationForest (champion) | Tier 2 — Material | Direct input to SAR trigger queue |
| GraphScorer (challenger) | Tier 2 — Material | Supplementary signal; influences SAR priority |
| FinCrimeScorer | Tier 2 — Material | Aggregates both domains; unified alert trigger |

Tier 2 classification triggers the following governance requirements under SS1/23: annual
full model validation; quarterly performance monitoring against approved thresholds;
documented champion/challenger comparison at each retraining cycle; and a model owner
attestation that the model continues to perform within approved bounds.

---

## 8. Regulatory Change Monitoring

The following regulatory developments are anticipated to affect this framework and should
be monitored:

**FCA AI and Machine Learning Discussion Paper (DP5/22) outcomes.** The FCA's consultation
on AI use in financial services may result in binding rules requiring enhanced explainability
obligations, mandatory model documentation standards, or new governance requirements for
automated decision-making systems.

**PSR APP Fraud Reimbursement Mandate (effective October 2024).** The mandatory
reimbursement scheme for authorised push payment fraud increases the financial consequence
of false negatives in the fraud model. This may require a review of the operating threshold
to reduce false-negative rate, with corresponding impact on false positive volume and the
analyst workload calculation.

**FATF Mutual Evaluation of the United Kingdom (next cycle).** FATF mutual evaluations
assess the technical compliance and effectiveness of the UK's AML/CFT regime. An adverse
finding on transaction monitoring effectiveness could trigger FCA supervisory activity
requiring firms to enhance their systems.

---

## 9. Glossary

| Term | Definition |
|---|---|
| AML | Anti-Money Laundering |
| AUC-PR | Area Under the Precision-Recall Curve |
| CDD | Customer Due Diligence |
| EDD | Enhanced Due Diligence |
| FATF | Financial Action Task Force |
| FCG | FCA Financial Crime Guide |
| FPR | False Positive Rate |
| JMLSG | Joint Money Laundering Steering Group |
| MLR 2017 | Money Laundering, Terrorist Financing and Transfer of Funds (Information on the Payer) Regulations 2017 |
| MLRO | Money Laundering Reporting Officer |
| NCA | National Crime Agency |
| POCA 2002 | Proceeds of Crime Act 2002 |
| PRA | Prudential Regulation Authority |
| PSR 2017 | Payment Services Regulations 2017 |
| SAR | Suspicious Activity Report |
| SHAP | SHapley Additive exPlanations |
| SS1/23 | PRA Supervisory Statement 1/23: Model Risk Management Principles for Banks |
| SYSC | Senior Management Arrangements, Systems and Controls sourcebook (FCA) |
