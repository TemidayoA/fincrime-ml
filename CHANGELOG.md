# Changelog

All notable changes to FinCrime-ML are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-15

Initial public release of the FinCrime-ML dual-domain financial crime detection framework.

### Added

**Core infrastructure**

- `BasePipeline` and `BaseScorer` abstract interfaces establishing the shared contract
  between fraud and AML domain modules.
- `SyntheticTransactionGenerator`: configurable card and SWIFT transaction generator
  with realistic fraud and AML typology injection, producing reproducible synthetic
  datasets without requiring proprietary data.
- `SyntheticAMLGenerator`: PaySim-style mobile money generator with mule chain
  annotation and structuring pattern seeding.
- `TypologyInjector`: post-hoc AML typology flag injector for labelling synthetic
  and real transaction datasets.
- IEEE-CIS fraud dataset loader and PaySim AML dataset adapter with schema harmonisation.
- `FinCrimeScorer`: unified fraud and AML risk fusion with three configurable strategies
  (weighted average, maximum, harmonic mean) and graceful single-signal degradation.
- Architecture decision record: `docs/adr/001-dual-domain.md` documenting the domain
  separation rationale and the regulatory tipping-off basis for the constraint.

**Fraud domain**

- `FraudFeatureEngineer`: velocity windows (1h, 6h, 24h, 7d, 30d), amount z-score
  deviation, MCC risk tier scoring, temporal features.
- `ImbalanceHandler`: SMOTE oversampling versus cost-sensitive weighting benchmark,
  with AUC-PR-based selection.
- `XGBFraudClassifier`: XGBoost champion model with five-fold stratified cross-validation,
  AUC-PR optimisation, and threshold analysis. AUC-PR 0.847 on held-out test set.
- `LogisticFraudBaseline`: interpretable logistic regression challenger with feature
  importance comparison.
- `FraudExplainer`: SHAP TreeExplainer integration producing per-transaction reason codes
  and full SHAP vector for FCA SYSC 10A audit logging.
- `FraudEvaluator`: precision-recall analysis, false positive cost matrix, and
  champion/challenger comparison tooling.

**AML domain**

- `TypologyEngine`: rule-based detection of structuring (POCA 2002 s.330 band
  GBP 8,500–9,950), layering, integration, and mule account typologies aligned to
  JMLSG Part I Ch.5.
- `TransactionGraphBuilder`: NetworkX entity relationship graph over the transaction
  network with centrality and community detection features.
- `AMLIsolationForest`: unsupervised anomaly baseline for the no-label scenario,
  with contamination parameter calibration guide.
- `GraphScorer`: supervised graph-based AML model using centrality, flow deviation,
  and pass-through ratio features. AUC-PR 0.782.
- `SARScorer`: SAR trigger scorer implementing six JMLSG indicator rules with
  three-tier priority assignment (CRITICAL / HIGH / MEDIUM), MLRO-ready narrative
  summary generation, and regulatory reference annotation. Designed to directly
  support the POCA 2002 s.330 mandatory disclosure obligation.
- `AlertFatigueEvaluator`: FPR at configurable sensitivity targets (80–99%),
  fatigue index (1 minus precision), and precision-recall AUC for monitoring system
  effectiveness review per FCA SYSC 6.3.

**Infrastructure and tooling**

- MySQL 8.0+ transaction monitoring schema: nine InnoDB tables including immutable
  FCA SYSC 10A audit log, SAR referral tracker with POCA 2002 ss.335–336 consent
  workflow, and soft-delete pattern for MLR 2017 Reg 40 five-year retention.
- SQL velocity query library: seven parameterised rolling-window queries for feature
  engineering pipelines, including structuring detection, rapid movement, cross-border
  velocity, and fan-out detection.
- SQL AML alert query library: ten MLRO operational queries covering the priority work
  queue, SAR filing pipeline, trigger frequency analysis, typology breakdown, alert
  fatigue MI, mule concentration, structuring pattern detection, and daily MI summary.
- `dashboard/index.html`: single-file, zero-dependency HTML monitoring dashboard
  for the MLRO alert queue with SHAP reason codes, priority filtering, sortable table,
  Chart.js visualisations, and click-to-expand alert detail panels.
- Pre-commit hook configuration enforcing black, ruff, and pytest coverage gate on
  every commit.

**Documentation**

- `docs/model_validation.md`: complete PRA SS1/23 model validation report covering
  holdout backtesting, champion/challenger comparison, alert fatigue analysis, analyst
  workload calculations, drift detection thresholds, and regulatory attestation.
- `docs/regulatory.md`: comprehensive FCA / JMLSG / FATF per-module regulatory
  alignment mapping.
- Comprehensive README with ASCII architecture diagram, quickstart, use case guide,
  and performance benchmarks.
- Jupyter notebooks: `fraud_detection.ipynb` and `aml_monitoring.ipynb` with
  regulatory commentary embedded throughout.

### Technical notes

- AUC-PR is used as the primary performance metric throughout in preference to ROC-AUC,
  appropriate to the highly imbalanced class distributions in fraud and AML datasets.
- NumPy 2.0 compatibility: `np.trapezoid` used in preference to the removed `np.trapz`.
- SHAP 0.44+ compatibility: `ndim == 3` check for the modern TreeExplainer API.
- Test suite: 850 tests, 97% coverage. Coverage gate set at 80% minimum in CI.
- Domain separation enforced: no cross-imports between `fincrime_ml.fraud` and
  `fincrime_ml.aml`; all shared code lives in `fincrime_ml.core`.

---

[0.1.0]: https://github.com/TemidayoA/fincrime-ml/releases/tag/v0.1.0
