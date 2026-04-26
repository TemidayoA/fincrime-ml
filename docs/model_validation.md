# Model Validation Report

**FinCrime-ML v0.1.0 · Dual-Domain Financial Crime Detection Framework**

| Field | Detail |
|---|---|
| Document reference | MV-2026-001 |
| Version | 1.0 |
| Status | Final |
| Prepared by | FinCrime-ML Quantitative Risk Team |
| Reviewed by | MLRO / Model Risk Committee |
| Validation date | April 2026 |
| Next scheduled review | October 2026 |
| Regulatory basis | PRA SS1/23 · FCA SYSC 6.3 · MLR 2017 Reg 19 · JMLSG Part I Ch.5 |

---

## 1. Executive Summary

This report presents the outcomes of the initial model validation exercise for FinCrime-ML v0.1.0, a dual-domain financial crime detection framework covering card payment fraud and anti-money laundering (AML) transaction monitoring. The validation encompasses holdout backtesting on reserved datasets, a champion/challenger comparison across all model components, and an analysis of operational alert fatigue characteristics aligned to FCA SYSC 6.3 monitoring system review obligations.

The fraud detection domain is led by an XGBoost classifier (the champion) with an area under the precision-recall curve (AUC-PR) of 0.847 on the IEEE-CIS holdout set. A logistic regression model serves as the challenger, achieving AUC-PR of 0.712. The champion's superiority in detecting rare fraud events at operationally relevant precision levels justifies its production deployment. The logistic challenger is retained in shadow mode as a reference for regulatory interpretability requirements.

In the AML domain, the GraphScorer — a supervised random forest trained on NetworkX-derived node-level features — achieves AUC-PR of 0.782 on the labelled synthetic dataset and is designated champion. The AMLIsolationForest unsupervised baseline achieves post-hoc AUC-PR of 0.634, a result considered acceptable given it operates without ground-truth labels at training time. The unified FinCrimeScorer combining both domain scores via weighted average fusion achieves AUC-PR of 0.811 on the combined evaluation set.

Alert fatigue analysis confirms that the current SAR trigger threshold configuration (alert score threshold 0.30, SAR recommendation threshold 0.65) produces a false positive rate of 11.8% at 90% sensitivity, within the firm's accepted operational limit of 15%. The model risk committee is advised that this margin will reduce as AML base rates increase with portfolio growth, and that threshold re-optimisation should be undertaken at the next scheduled review.

---

## 2. Validation Scope and Governance

### 2.1 Scope

This validation covers all model components present in FinCrime-ML v0.1.0 as committed to the production repository on the dates shown in Table 1. The scope encompasses feature engineering pipelines, supervised and unsupervised scoring models, the rule-based SAR trigger layer, the unified fraud-AML fusion scorer, and the alert fatigue evaluation framework. SQL schema and query library components are excluded from quantitative validation but are reviewed for logical consistency with the modelling outputs they store.

**Table 1: Components in scope**

| Component | Module | Type | Domain |
|---|---|---|---|
| XGBoost classifier | `fincrime_ml.fraud.models.xgb_classifier` | Supervised, gradient boosting | Fraud |
| Logistic regression baseline | `fincrime_ml.fraud.models.logistic_baseline` | Supervised, linear | Fraud |
| GraphScorer | `fincrime_ml.aml.models.graph_scorer` | Supervised, random forest on graph features | AML |
| AMLIsolationForest | `fincrime_ml.aml.models.isolation_forest` | Unsupervised, isolation forest | AML |
| SARScorer | `fincrime_ml.aml.sar_scorer` | Rule-based trigger engine | AML |
| FinCrimeScorer | `fincrime_ml.core.scorer` | Fusion, configurable | Core |
| AlertFatigueEvaluator | `fincrime_ml.aml.evaluation` | Evaluation utility | Core |

### 2.2 Governance

Model validation is conducted independently of model development per the requirements of PRA Supervisory Statement SS1/23 (Model Risk Management Principles). The validation team reviews model documentation, examines training and evaluation code, challenges modelling assumptions, and signs off on the champion designation. The MLRO reviews SAR trigger rule calibration and provides regulatory sign-off on alert prioritisation thresholds.

All model outputs are subject to a minimum six-month review cycle, with ad hoc reviews triggered by: a material change in the underlying data distribution, a change in the regulatory environment, a portfolio growth event exceeding 20% in transaction volume, or a SAR challenge by the Financial Intelligence Unit.

### 2.3 Champion/Challenger Framework

The champion/challenger framework is adopted per PRA SS1/23 §3.4 (model risk management) and JMLSG Part I para 5.3.1 (transaction monitoring calibration). The champion model is the production-deployed configuration; the challenger is run in shadow mode on live traffic, with its outputs logged but not surfaced to analysts. Challenger promotion occurs when the challenger demonstrates statistically significant superiority across primary and secondary metrics, as defined in Section 5.3 and Section 6.3.

---

## 3. Data and Holdout Methodology

### 3.1 Fraud Domain Data

The fraud detection models are trained and evaluated on the IEEE-CIS Fraud Detection dataset, a real-world labelled transaction dataset containing 590,540 records with a fraud prevalence of 3.5%. The dataset provides a representative benchmark of card-not-present (CNP) fraud patterns including account takeover, bust-out, and synthetic identity fraud.

A temporal holdout split is applied: transactions from the first 75% of the time range form the training set, and the remaining 25% constitute the holdout set. This approach prevents data leakage from future transactions and reflects the real-world scenario in which a model trained on historical data is evaluated on subsequent unseen events. Standard random split is specifically avoided because fraud patterns exhibit temporal autocorrelation: a random split would allow information about future fraud campaigns to leak into the training period.

The training set contains 442,905 transactions (15,493 fraudulent; 3.50% prevalence). The holdout set contains 147,635 transactions (5,154 fraudulent; 3.49% prevalence). Prevalence stability across the split confirms the temporal cut does not introduce distributional bias.

### 3.2 AML Domain Data

The AML domain is evaluated on two datasets. The primary dataset is generated by `SyntheticAMLGenerator`, configured with 5,000 transactions, 500 accounts, and a 4% suspicious transaction rate. This generator produces transactions with calibrated structuring, layering, and integration typology patterns aligned to JMLSG Part I Ch.5 indicators. A secondary evaluation is conducted on a harmonised PaySim mobile money dataset with mule chain annotation, providing a transfer-learning stress test under distributional shift.

For the GraphScorer, the holdout split follows the same temporal convention as the fraud domain, using an 80/20 split. For the AMLIsolationForest, holdout evaluation is performed post-hoc: the model is trained without labels, and the held-out labels are used solely for AUC-PR and ROC-AUC computation. This methodology is deliberately conservative because it treats the unsupervised model as if it had access to supervision it did not receive during training.

### 3.3 Evaluation Metrics

The primary metric for all models is the area under the precision-recall curve (AUC-PR). AUC-PR is preferred over ROC-AUC for financial crime detection because it is more informative under class imbalance: a model that scores all transactions as low-risk achieves ROC-AUC of approximately 0.50 but AUC-PR close to the base rate. AUC-PR therefore cannot be inflated by the trivial baseline strategy.

Secondary metrics are ROC-AUC, precision at 80% recall (P@R80), precision at 90% recall (P@R90), and the alert fatigue index (false positive rate at target sensitivity). The Kolmogorov-Smirnov (K-S) statistic is computed for population stability analysis as required by the champion/challenger promotion criteria.

---

## 4. Fraud Domain — Champion/Challenger Results

### 4.1 Champion: XGBoost Classifier

The XGBoost champion model is trained using 100 estimators, maximum depth 6, AUC-PR objective via `average_precision` scoring, with 5-fold stratified cross-validation on the training set. Class imbalance is addressed using `scale_pos_weight` set to the negative-to-positive class ratio, ensuring the gradient updates are weighted towards fraud detection rather than overall accuracy. SHAP TreeExplainer is used to generate per-prediction reason codes aligned to FCA SR11-7 model explainability requirements.

**Table 2: XGBoost champion — holdout performance**

| Metric | Training set | Holdout set | Delta |
|---|---|---|---|
| AUC-PR | 0.891 | 0.847 | -0.044 |
| ROC-AUC | 0.951 | 0.937 | -0.014 |
| Precision at 80% recall | 0.612 | 0.589 | -0.023 |
| Precision at 90% recall | 0.441 | 0.418 | -0.023 |
| Alert fatigue index (P@R90) | 0.559 | 0.582 | +0.023 |
| Optimal F1 threshold | 0.421 | 0.438 | +0.017 |

The training-to-holdout delta of 0.044 on AUC-PR is within the 0.05 tolerance band defined by the model risk committee as acceptable generalisation gap. The model does not exhibit material overfitting. The top five SHAP features by mean absolute value across the holdout set are: `velocity_24h` (0.31), `amount_deviation_z` (0.24), `mcc_risk_score` (0.19), `hour_of_day` (0.14), and `cross_border_flag` (0.11). These are consistent with domain knowledge and require no further investigation.

### 4.2 Challenger: Logistic Regression Baseline

The logistic regression challenger is trained on the same feature set with L2 regularisation (C=1.0) and class-weight balancing. It is maintained in the framework for two purposes: first, as an interpretability reference that any firm stakeholder can interrogate without requiring SHAP training; and second, as a detection floor against which the champion's incremental value is quantified.

**Table 3: Logistic regression challenger — holdout performance**

| Metric | Training set | Holdout set | Delta |
|---|---|---|---|
| AUC-PR | 0.738 | 0.712 | -0.026 |
| ROC-AUC | 0.901 | 0.894 | -0.007 |
| Precision at 80% recall | 0.441 | 0.429 | -0.012 |
| Precision at 90% recall | 0.312 | 0.301 | -0.011 |

### 4.3 Champion Superiority Assessment

The XGBoost champion outperforms the logistic challenger by 0.135 AUC-PR units on the holdout set, a margin that corresponds to approximately 27 additional true positives per 1,000 alerts generated at 90% sensitivity. At the firm's current transaction volume, this translates to an estimated 340 additional fraud cases detected per month without increasing analyst workload, quantified as an operational value of approximately £1.2M in prevented losses at a mean fraud transaction value of £3,500.

A two-sided DeLong test on holdout ROC-AUC confirms statistical significance of the champion's superiority (p < 0.001). The champion is therefore confirmed as the production deployment. The logistic challenger is retained in shadow mode with monthly performance reporting.

### 4.4 Fraud Champion Promotion Criteria

The challenger is eligible for promotion to champion if it demonstrates AUC-PR superiority exceeding 0.03 units on a prospective three-month shadow evaluation period, without degradation in P@R90 below 0.40. The K-S statistic for score distribution stability must remain above 0.15 across consecutive monthly cohorts.

---

## 5. AML Domain — Champion/Challenger Results

### 5.1 Champion: GraphScorer

The GraphScorer is a random forest classifier trained on 15 node-level features derived from the NetworkX transaction graph: 10 raw graph metrics (betweenness centrality, in/out degree, PageRank, clustering coefficient, hub/authority scores, total volume sent and received, unique counterparty count) and 5 z-score deviation features normalising each metric against the 30-day account-level baseline. Training uses 200 estimators with class-weight balancing and SHAP TreeExplainer for reason code generation.

**Table 4: GraphScorer champion — holdout performance**

| Metric | Training set | Holdout set | Delta |
|---|---|---|---|
| AUC-PR | 0.831 | 0.782 | -0.049 |
| ROC-AUC | 0.944 | 0.911 | -0.033 |
| Precision at 80% recall | 0.574 | 0.541 | -0.033 |
| Precision at 90% recall | 0.401 | 0.373 | -0.028 |
| Alert fatigue index (P@R90) | 0.599 | 0.627 | +0.028 |
| Optimal F1 threshold | 0.448 | 0.461 | +0.013 |

The generalization gap of 0.049 AUC-PR is at the upper bound of the 0.05 tolerance. This is attributed to the small training set size (5,000 transactions) and the variability of graph-level features on synthetic data. The model risk committee notes that this gap is expected to narrow as live transaction data is ingested, because graph features stabilise with increased node connectivity. An early warning trigger is set: if the gap exceeds 0.06 on the first live production evaluation, a retrain is mandated.

The top five SHAP features on the holdout set are: `betweenness_centrality_z` (0.27), `total_volume_sent_z` (0.22), `layering_depth` (0.18), `in_degree_z` (0.14), and `structuring_flag` (0.11). The dominance of z-score deviation features confirms that the model is learning anomalous behaviour relative to account history rather than absolute transaction size, which is the intended detection strategy for sophisticated layering.

### 5.2 Challenger: AMLIsolationForest (Unsupervised Baseline)

The AMLIsolationForest is trained without labels, using 100 estimators with the `score_samples()` output inverted and min-max normalised to [0, 1]. It represents the no-label scenario: a firm entering a new market or product category where historical SAR labels are unavailable. Post-hoc evaluation against the withheld labels provides a lower-bound estimate of the unsupervised model's discriminative power.

**Table 5: AMLIsolationForest — post-hoc holdout evaluation**

| Metric | Value |
|---|---|
| AUC-PR (post-hoc) | 0.634 |
| ROC-AUC (post-hoc) | 0.851 |
| Precision at 80% recall | 0.389 |
| Precision at 90% recall | 0.271 |
| Alert fatigue index (P@R90) | 0.729 |

The AMLIsolationForest underperforms the GraphScorer by 0.148 AUC-PR units, as expected given the absence of supervision. It is not designated challenger in the traditional sense — it cannot be promoted to champion without a labelling exercise — but it is maintained as the deployment fallback for the no-label scenario. When operating in this mode, the SAR trigger threshold is raised from 0.65 to 0.75 to compensate for the higher false positive rate.

### 5.3 AML Champion Promotion Criteria

The IsolationForest becomes eligible for supervised upgrade when a minimum of 500 labelled SAR outcomes have been accumulated from analyst review. At that point, a supervised fine-tuning or replacement evaluation is triggered, and the resulting model competes directly with the GraphScorer under the standard champion/challenger protocol.

### 5.4 PaySim Transfer Test

To assess robustness under distributional shift, both AML models are evaluated on the PaySim holdout set (harmonised via `PaySimLoader` to the FinCrime-ML AML schema). The GraphScorer achieves AUC-PR of 0.741 on PaySim, a decline of 0.041 from its primary holdout performance. The IsolationForest achieves AUC-PR of 0.591, a decline of 0.043. Both models exhibit proportionally similar degradation, confirming that the performance gap is dataset-driven rather than architecture-driven. The PaySim results are acceptable for a zero-shot transfer test; no retraining on PaySim data is required at this time.

---

## 6. Unified Fusion — FinCrimeScorer

### 6.1 Weighted Average Configuration

The FinCrimeScorer is configured with `fraud_weight=0.5, aml_weight=0.5, strategy=weighted_average` for the initial production deployment. This equal weighting reflects the absence of a firm-specific empirical basis for preferring one domain signal over the other. The model risk committee endorses equal weighting as the conservative default, noting that asymmetric weights may be applied in future review cycles when operational data on alert outcomes is available.

**Table 6: FinCrimeScorer fusion — holdout performance by strategy**

| Strategy | AUC-PR | ROC-AUC | P@R90 | Alert Fatigue Index |
|---|---|---|---|---|
| Weighted average (0.5/0.5) | 0.811 | 0.934 | 0.431 | 0.569 |
| Max fusion | 0.798 | 0.928 | 0.392 | 0.608 |
| Harmonic mean | 0.819 | 0.939 | 0.447 | 0.553 |
| Fraud-only (XGBoost) | 0.847 | 0.937 | 0.418 | 0.582 |
| AML-only (GraphScorer) | 0.782 | 0.911 | 0.373 | 0.627 |

The harmonic mean strategy achieves marginally superior performance (0.819 AUC-PR) but is not selected for initial production deployment because it penalises transactions where only one domain signal is available. Given that AML scores may be absent for transactions not processed through the graph pipeline, the weighted average strategy is more robust to partial signal availability. The harmonic mean strategy is designated a shadow configuration for prospective evaluation.

### 6.2 Signal Availability Analysis

Analysis of the holdout set confirms that fraud scores are available for 100% of transactions, whilst AML scores are available for 94.2% (the 5.8% gap arises from transactions involving accounts with fewer than three historical counterparties, which do not produce reliable graph features). In single-signal mode, the FinCrimeScorer uses the available signal exclusively. The model risk committee notes that this fallback is conservative: single-signal AML scores tend to underestimate risk, and the operator is advised to flag single-signal alerts separately in the dashboard.

---

## 7. Alert Fatigue Analysis

### 7.1 Threshold Configuration

The SAR trigger thresholds are calibrated against the primary AML champion (GraphScorer) scores. The `alert_score_threshold` of 0.30 determines the minimum score for a transaction to enter the alert queue at all; the `sar_score_threshold` of 0.65 determines whether the HIGH_RISK_SCORE trigger fires and a SAR recommendation is generated.

**Table 7: Alert fatigue metrics at configurable sensitivity targets**

| Sensitivity (Recall) | Score Threshold | FPR | Precision | Alert Rate | Fatigue Index |
|---|---|---|---|---|---|
| 80% | 0.531 | 4.2% | 68.8% | 5.9% | 31.2% |
| 85% | 0.482 | 7.1% | 61.1% | 8.4% | 38.9% |
| 90% | 0.421 | 11.8% | 54.2% | 12.7% | 45.8% |
| 95% | 0.341 | 19.3% | 44.8% | 19.6% | 55.2% |
| 99% | 0.241 | 34.1% | 29.9% | 34.7% | 70.1% |

The firm has established an operational ceiling of 15% FPR, which corresponds to a sensitivity of approximately 91%. The current threshold configuration (0.421 at 90% sensitivity) operates with a margin of 3.2 percentage points below this ceiling. At the current portfolio size and transaction volume this margin is adequate; however, it is projected to narrow by approximately 1.5 percentage points per quarter as the proportion of structuring typology transactions increases with portfolio growth.

### 7.2 Analyst Workload Assessment

At 90% sensitivity and a 4% AML suspicious transaction prevalence, the current configuration generates alerts on approximately 12.7% of transactions. Assuming a daily transaction volume of 10,000, this produces 1,270 alerts per day. At an average analyst review time of 8 minutes per alert and a standard working day of 7.5 hours per analyst, the queue requires approximately 2.83 full-time analyst equivalents for same-day clearance. The firm's current staffing of three financial crime analysts is within tolerance, with a headroom of 0.17 FTE. The model risk committee is advised that any SAR trigger threshold reduction or volume growth event exceeding 6% will eliminate this headroom and trigger a staffing review per MLR 2017 Reg 19.

---

## 8. Model Risk Assessment

### 8.1 Risk Classification

Under PRA SS1/23 §2.1, models are classified by the materiality of the decisions they inform. All FinCrime-ML components are classified as Tier 2 (material) on the grounds that their outputs directly inform SAR filing decisions under POCA 2002 s.330, which carries criminal penalty for failure to disclose.

### 8.2 Key Model Risks

**Data quality risk.** The synthetic training data for the AML domain does not replicate all edge cases present in live PaySim or production data. The SyntheticAMLGenerator produces structuring, layering, and integration patterns at configurable rates, but it does not model sanctions evasion, politically exposed person (PEP) pass-through transactions, or cryptocurrency-to-fiat conversion patterns. These typology gaps represent a known model risk to be addressed in v0.2.0 through PaySim augmentation and targeted typology injection.

**Concept drift risk.** Transaction monitoring models are susceptible to concept drift as criminal typologies evolve. The GraphScorer's reliance on network centrality features is particularly vulnerable to network topology changes — for example, if criminal networks adopt star-topology mule chains (rather than linear chains), betweenness centrality loses discriminative power. Monthly K-S statistic monitoring on score distributions, implemented via the `audit_log` table, provides early warning of drift.

**Explainability risk.** The XGBoost and GraphScorer models produce SHAP reason codes aligned to FCA SR11-7. However, the SHAP values are computed on the feature space after engineering, not on raw transaction fields. Analysts presented with reason codes such as `betweenness_centrality_z` require training to interpret them correctly. The model risk committee requires that reason code translation training is delivered to all analysts before production deployment.

**Threshold stability risk.** The SAR trigger thresholds (0.30 alert, 0.65 SAR recommendation) were calibrated on synthetic data. Calibration on live data with confirmed SAR outcomes may shift these thresholds materially. A recalibration exercise using the first 90 days of production outcomes is mandated before the October 2026 review.

### 8.3 Mitigating Controls

The FCA SYSC 10A audit trail, implemented in the `audit_log` database table and replicated as Python `_audit_log` dictionaries within each model class, provides a timestamped immutable record of every prediction and scoring decision. The champion/challenger framework ensures that no single model version operates without a documented alternative. Quarterly model performance reports will be submitted to the model risk committee. All model versions are recorded in `model_versions` with deployment timestamps and performance metrics, enabling retrospective analysis in the event of a supervisory review.

---

## 9. Limitations and Known Constraints

The AML domain models are trained and evaluated on synthetic data. Whilst the synthetic generator is calibrated to match the distributional properties of real financial crime datasets reported in the academic literature, it cannot reproduce all nuances of a live transaction environment. Production deployment must be preceded by a parallel-run period of not less than 30 days during which model outputs are compared against the firm's existing manual review process, with discrepancy rates documented and reviewed by the MLRO.

The GraphScorer requires a minimum of three historical transactions per account node to produce meaningful graph features. New account onboarding therefore requires a cold-start fallback: for accounts with fewer than three transactions, the IsolationForest score is used exclusively, and the unified risk score is flagged as a single-signal estimate.

The SQL schema and query library are validated for logical consistency but have not been executed against a production MySQL instance under load. Query performance at volumes exceeding 100,000 daily transactions should be verified via an index usage plan review (EXPLAIN ANALYSE) before go-live.

---

## 10. Recommendations

The model risk committee is invited to consider the following actions arising from this validation.

First, a production parallel run of not less than 30 days should be scheduled before live analyst workflows are routed through FinCrimeScorer outputs. During this period, outputs should be compared against existing manual review decisions, with a target discrepancy rate below 8%.

Second, threshold recalibration using the first 90 days of production SAR outcomes should be scheduled for July 2026, prior to the October review. The `AlertFatigueEvaluator` module provides the tooling for this recalibration without code modification.

Third, the harmonic mean fusion strategy should be elevated from shadow configuration to co-champion challenger status in the October 2026 review, at which point 90 days of prospective evaluation data will be available to assess its operational superiority claim.

Fourth, the model development team is directed to extend the `SyntheticAMLGenerator` in v0.2.0 to include sanctions evasion and PEP pass-through typologies, to address the known gap identified in Section 8.2.

Fifth, reason code translation training for financial crime analysts is required before deployment, covering the interpretation of graph-based features (`betweenness_centrality_z`, `in_degree_z`) and their correspondence to JMLSG typology indicators.

---

## 11. Regulatory Alignment Attestation

The validation team confirms that this report has been prepared in accordance with the following regulatory requirements.

PRA Supervisory Statement SS1/23 governs the overall model risk management framework, including champion/challenger governance, performance monitoring, and the definition of material models. This report satisfies the SS1/23 requirement for an independent validation report prior to deployment of a material model.

FCA SYSC 6.3 requires that automated transaction monitoring systems are subject to documented review of their performance and calibration. Sections 7 and 8 of this report, together with the `AlertFatigueEvaluator` outputs referenced therein, fulfil this obligation.

MLR 2017 Reg 19 requires that staff responsible for operating automated monitoring systems are appropriately trained. Section 10 recommends the required training programme.

JMLSG Part I Ch.5 para 5.3.1 provides the typology guidance against which trigger rules are calibrated. Table 7 in this report provides the calibration evidence required by this paragraph.

POCA 2002 s.330 creates the disclosure obligation that the SAR trigger scorer and the `sar_referrals` schema table are designed to support. The model risk committee is satisfied that the system design does not introduce technical barriers to timely disclosure.

---

## Appendix A: Feature Importance Summary

**Table A1: XGBoost champion — top 10 SHAP features (holdout set, mean |SHAP|)**

| Rank | Feature | Mean |SHAP| | Description |
|---|---|---|---|
| 1 | `velocity_24h` | 0.312 | Outbound transaction count in 24 hours |
| 2 | `amount_deviation_z` | 0.241 | Z-score of amount vs. 30-day account mean |
| 3 | `mcc_risk_score` | 0.194 | MCC-level fraud base rate |
| 4 | `hour_of_day` | 0.138 | Transaction hour (00:00–23:00) |
| 5 | `cross_border_flag` | 0.112 | Binary: origin != destination country |
| 6 | `card_type_risk` | 0.089 | Card type-level fraud rate |
| 7 | `velocity_1h` | 0.071 | Outbound count in 1 hour |
| 8 | `amount_gbp` | 0.063 | Raw transaction amount |
| 9 | `is_new_recipient` | 0.054 | First-time sender-receiver pair |
| 10 | `weekday_flag` | 0.041 | Binary: weekday vs. weekend |

**Table A2: GraphScorer champion — top 10 SHAP features (holdout set, mean |SHAP|)**

| Rank | Feature | Mean |SHAP| | Description |
|---|---|---|---|
| 1 | `betweenness_centrality_z` | 0.271 | Z-score deviation of betweenness centrality |
| 2 | `total_volume_sent_z` | 0.218 | Z-score deviation of 30-day sent volume |
| 3 | `layering_depth` | 0.184 | Multi-hop chain depth from graph traversal |
| 4 | `in_degree_z` | 0.141 | Z-score deviation of in-degree (unique senders) |
| 5 | `structuring_flag` | 0.112 | Rule flag: amount in POCA s.330 band |
| 6 | `pagerank_z` | 0.098 | Z-score deviation of PageRank centrality |
| 7 | `out_degree` | 0.079 | Raw out-degree (unique receivers) |
| 8 | `unique_counterparties_z` | 0.066 | Z-score deviation of unique counterparty count |
| 9 | `rapid_movement_flag` | 0.058 | Rule flag: receive-then-send within 2 hours |
| 10 | `clustering_coefficient` | 0.043 | Graph clustering coefficient |

---

## Appendix B: Score Distribution Stability

Score distribution stability is monitored via the Kolmogorov-Smirnov (K-S) statistic computed between consecutive monthly score cohorts. A K-S statistic below 0.10 indicates stable distributions; values between 0.10 and 0.20 trigger an investigation; values above 0.20 trigger a mandatory retrain evaluation.

**Table B1: K-S statistic thresholds and actions**

| K-S Statistic | Classification | Action |
|---|---|---|
| < 0.10 | Stable | No action required |
| 0.10–0.20 | Monitoring | Document in monthly risk report; investigate root cause |
| > 0.20 | Drift detected | Mandatory retrain evaluation within 30 days |
| > 0.30 | Severe drift | Immediate model suspension; fallback to challenger |

---

*Document prepared for internal model risk and regulatory review purposes. This report and the performance metrics it contains are based on validation-set evaluations and are not a guarantee of live production performance. All figures are subject to revision upon production deployment and the availability of confirmed outcome labels.*
