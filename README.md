# FinCrime-ML

[![CI](https://github.com/TemidayoA/fincrime-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/TemidayoA/fincrime-ml/actions)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)](https://github.com/TemidayoA/fincrime-ml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)

A dual-domain Python framework for financial crime detection: **payment fraud** and **AML
transaction monitoring**, built for UK regulatory context with production-grade ML engineering.

---

## Contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Performance](#performance)
- [Quickstart](#quickstart)
- [Use case guide](#use-case-guide)
- [Dashboard](#dashboard)
- [Database schema](#database-schema)
- [Regulatory alignment](#regulatory-alignment)
- [Running tests](#running-tests)
- [Project structure](#project-structure)
- [Contributing](#contributing)

---

## Why this exists

Most financial crime ML tutorials treat fraud detection as a single binary classification
problem on a Kaggle dataset. In practice, UK-regulated institutions operate under two
distinct supervisory regimes: PSR 2017 / PSD2 for fraud, and POCA 2002 / MLR 2017 for AML.
These regimes impose different alert workflows, operational latency constraints, model
explainability obligations, and regulatory reporting requirements.

FinCrime-ML models this separation explicitly:

- The fraud and AML pipelines are independent modules that share no cross-domain imports
  (enforced by the dual-domain architecture decision record at `docs/adr/001-dual-domain.md`).
- A unified risk scorer fuses both signals with configurable weighting strategies.
- Every design decision traces back to a specific regulatory reference (FCA SYSC 6.3,
  POCA 2002 s.330, JMLSG Part I Ch.5, PRA SS1/23).
- AUC-PR is used as the primary metric throughout, not ROC-AUC, because the fraud/AML
  class imbalance makes ROC-AUC misleading in operational contexts.

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              Transaction Ingest              │
                        │   Raw payment events → schema harmonisation  │
                        └────────────────────┬────────────────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    │                  core/data/                      │
                    │   SyntheticTransactionGenerator · IEEE-CIS       │
                    │   loader · PaySim loader · TypologyInjector      │
                    └────────────┬───────────────────────┬────────────┘
                                 │                       │
              ┌──────────────────▼──────────┐  ┌────────▼──────────────────┐
              │       FRAUD DOMAIN          │  │       AML DOMAIN           │
              │                             │  │                            │
              │  FraudFeatureEngineer       │  │  TypologyEngine            │
              │  ├─ velocity windows        │  │  ├─ structuring detection  │
              │  ├─ amount z-score          │  │  ├─ layering patterns      │
              │  └─ MCC risk scoring        │  │  └─ integration signals    │
              │                             │  │                            │
              │  ImbalanceHandler           │  │  TransactionGraphBuilder   │
              │  ├─ SMOTE oversampling      │  │  ├─ NetworkX entity graph  │
              │  └─ cost-sensitive weights  │  │  ├─ centrality features    │
              │                             │  │  └─ pass-through ratio     │
              │  XGBFraudClassifier ────────│  │                            │
              │  LogisticFraudBaseline      │  │  AMLIsolationForest ───────│
              │  (champion / challenger)    │  │  GraphScorer               │
              │                             │  │  (champion / challenger)   │
              │  FraudExplainer             │  │                            │
              │  └─ SHAP reason codes       │  │  SARScorer                 │
              │                             │  │  └─ 6 JMLSG trigger rules  │
              │  FraudEvaluator             │  │     SAR priority queue     │
              │  └─ AUC-PR · cost matrix    │  │                            │
              │     threshold analysis      │  │  AlertFatigueEvaluator     │
              └──────────────┬──────────────┘  └────────────┬──────────────┘
                             │                              │
                             └──────────────┬───────────────┘
                                            │
                              ┌─────────────▼─────────────┐
                              │      core/scorer.py        │
                              │   FinCrimeScorer           │
                              │   ├─ weighted_average      │
                              │   ├─ max                   │
                              │   └─ harmonic_mean         │
                              └─────────────┬─────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                  Outputs                       │
                    │                                                │
                    │  MySQL schema  ·  MLRO alert queue            │
                    │  SAR referrals ·  FCA SYSC 10A audit log      │
                    │  HTML dashboard · Model validation report      │
                    └────────────────────────────────────────────────┘
```

### Domain separation rule

The fraud and AML modules must not import from each other. All cross-domain communication
passes through `fincrime_ml/core/`. This constraint is enforced at code review and mirrors
the regulatory tipping-off prohibition under POCA 2002 s.333A: the fraud investigation
workflow and the SAR disclosure workflow are kept operationally separate.

---

## Performance

Results on held-out test sets (20% stratified split). Full methodology in
[`docs/model_validation.md`](docs/model_validation.md).

### Fraud domain

| Model | AUC-PR | ROC-AUC | F1 @ 0.5 | Precision @ 90% Recall |
|---|---|---|---|---|
| XGBoost (champion) | **0.847** | 0.964 | 0.791 | 0.683 |
| Logistic Regression (challenger) | 0.712 | 0.941 | 0.734 | 0.521 |

### AML domain

| Model | AUC-PR | ROC-AUC | Alert Rate @ 90% Sensitivity |
|---|---|---|---|
| GraphScorer (champion) | **0.782** | 0.931 | 8.3% |
| IsolationForest (challenger) | 0.634 | 0.887 | 12.1% |

### Unified scorer

| Fusion strategy | AUC-PR | Alert fatigue index |
|---|---|---|
| Weighted average (fraud 0.6, AML 0.4) | **0.811** | 0.217 |
| Maximum | 0.798 | 0.284 |
| Harmonic mean | 0.779 | 0.193 |

---

## Quickstart

### Installation

```bash
git clone https://github.com/TemidayoA/fincrime-ml.git
cd fincrime-ml
pip install -e ".[dev]"
pre-commit install     # optional: enforces black + ruff + tests on commit
```

Requires Python 3.11+. All dependencies are pure-Python or wheel-distributed.

### Generate synthetic transaction data

```python
from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator

gen = SyntheticTransactionGenerator(n_accounts=5_000, seed=42)

# 50,000 card transactions with approximately 1.5% fraud rate
df = gen.generate(n_transactions=50_000, fraud_rate=0.015)
print(df.shape)                        # (50000, 24)
print(df["is_fraud"].value_counts())   # 49250 / 750

# Wire transfers with AML typologies injected
wires = gen.generate_wire_transfers(n=5_000)
print(wires["typology"].value_counts())
# normal          4312
# structuring      341
# layering         248
# integration       99
```

### Train and evaluate the fraud model

```python
from fincrime_ml.fraud.models.xgb_classifier import XGBFraudClassifier
from fincrime_ml.fraud.evaluation import FraudEvaluator

clf = XGBFraudClassifier()
clf.train(df_train, label_col="is_fraud")

# Score a holdout set
scores = clf.predict(df_holdout)
print(scores[["transaction_id", "fraud_score", "risk_tier"]].head())
#    transaction_id  fraud_score risk_tier
# 0     TXN-0001234       0.9231  CRITICAL
# 1     TXN-0001235       0.0142       LOW

# Evaluate with precision-recall analysis
evaluator = FraudEvaluator()
report = evaluator.evaluate(scores, df_holdout["is_fraud"])
print(f"AUC-PR: {report['auc_pr']:.3f}")
print(f"Optimal threshold (max F1): {report['optimal_threshold']:.3f}")
```

### SHAP reason codes per transaction

```python
from fincrime_ml.fraud.explain import FraudExplainer

explainer = FraudExplainer(clf)
explanations = explainer.explain(df_holdout.head(200))

# Top three driver features per transaction
print(explanations[["transaction_id", "top_reason_1", "top_reason_2", "top_reason_3"]].head())
#    transaction_id         top_reason_1         top_reason_2       top_reason_3
# 0     TXN-0001234    txn_count_1h=12.0  amount_z_score=4.1  mcc_risk=high_risk
```

### Run AML typology detection and SAR scoring

```python
from fincrime_ml.aml.typologies import TypologyEngine
from fincrime_ml.aml.sar_scorer import SARScorer, SARScorerConfig

engine = TypologyEngine()
flagged = engine.detect(wires)
print(f"Transactions with AML signals: {len(flagged)}")

config = SARScorerConfig(min_triggers_for_sar=2)
scorer = SARScorer(config=config)
alerts = scorer.score(flagged)

# Priority 1 alerts — immediate MLRO referral (POCA 2002 s.330)
critical = alerts[alerts["priority"] == 1]
print(critical[["alert_id", "risk_score", "trigger_reasons", "sar_recommended", "mlro_summary"]])
```

### Unified fraud + AML scoring

```python
from fincrime_ml.core.scorer import FinCrimeScorer, FusionConfig

config = FusionConfig(
    strategy="weighted_average",
    fraud_weight=0.6,
    aml_weight=0.4,
)
scorer = FinCrimeScorer(config=config)

unified = scorer.score(fraud_scores=fraud_df, aml_scores=aml_df)
print(unified[["transaction_id", "unified_risk_score", "risk_tier"]].head())
```

### Alert fatigue analysis

```python
from fincrime_ml.aml.evaluation import AlertFatigueEvaluator

evaluator = AlertFatigueEvaluator()
report = evaluator.evaluate(y_true=labels, y_score=aml_scores)

print(f"AUC-PR: {report.pr_auc:.3f}")
print(f"FPR at 90% sensitivity: {report.fpr_at_sensitivity(0.90):.1%}")
print(f"Fatigue index at optimal threshold: {report.fatigue_index():.3f}")
```

---

## Use case guide

### UK MLRO / Compliance analyst

The SAR alert queue in `sql/queries/aml_alerts.sql` provides ten pre-built operational
queries for the MLRO work queue, SAR filing pipeline, typology breakdown reports, and
daily MI summaries. The HTML dashboard at `dashboard/index.html` renders this queue
with SHAP reason codes and filter/sort controls, and requires no server: open it directly
in a browser.

### Fraud operations team

The `FraudEvaluator` threshold analysis produces a false positive cost matrix that
quantifies the operational cost of each threshold choice. The `FraudExplainer` reason
codes are designed for direct use in customer-facing decline messages and analyst
review queues. The champion/challenger comparison at `docs/model_validation.md` provides
the retraining sign-off documentation required by PRA SS1/23.

### Model validator / model risk officer

`docs/model_validation.md` is the complete model risk governance document: holdout
backtesting results, champion/challenger comparison, alert fatigue analysis, analyst
workload calculations, drift detection thresholds, and regulatory attestation. It is
written to PRA SS1/23 standards and ready for submission to model risk committees.

### Data scientist / ML engineer (portfolio use)

The project demonstrates production engineering practices across a complete ML lifecycle:
data generation, feature engineering, class imbalance handling, model training, SHAP
explainability, evaluation, AML-specific unsupervised scoring, graph analytics, alert
prioritisation, SQL schema design, and monitoring dashboard. All modules are tested to
97% coverage. The dual-domain architecture enforces clean separation of concerns in a
way that mirrors actual regulatory constraints, not just software design preference.

---

## Dashboard

`dashboard/index.html` is a single-file, zero-dependency HTML dashboard for the MLRO
alert queue. Open it directly in any modern browser.

**Features:**

- Live alert queue with priority filter (All / P1 Critical / P2 High / P3 Medium / SAR)
- Sortable table columns (priority, risk score, amount, time open)
- Click-to-expand alert detail panel with SHAP reason codes
- Alert volume donut chart by priority tier
- AML typology distribution bar chart
- Alert fatigue curve (FPR vs sensitivity)
- Daily alert volume trend line
- KPI strip: open alerts, P1 count, SAR queue, mean review time

---

## Database schema

`sql/schema.sql` defines nine InnoDB tables for MySQL 8.0+:

| Table | Purpose |
|---|---|
| `accounts` | Account master with mule flag and risk segment |
| `transactions` | Core transaction ledger with typology flags |
| `model_versions` | Model registry supporting champion/challenger per PRA SS1/23 |
| `fraud_scores` | XGBoost / logistic output with SHAP JSON per transaction |
| `aml_scores` | IsolationForest / GraphScorer output per transaction |
| `unified_scores` | FinCrimeScorer fusion output |
| `aml_alerts` | SAR alert queue with MLRO workflow status |
| `sar_referrals` | NCA SAR filing tracker with consent regime workflow |
| `audit_log` | Immutable FCA SYSC 10A automated decision audit trail |

Three views (`v_active_alerts`, `v_sar_pending`, `v_daily_alert_mi`) provide the primary
MLRO operational interfaces. The velocity query library at `sql/queries/velocity.sql`
supplies parameterised rolling-window aggregates for feature engineering pipelines.

---

## Regulatory alignment

| Module | Instrument | Key obligation |
|---|---|---|
| `fraud/` pipeline | PSR 2017 Reg 98, FCA FCG 3.2 | Fraud detection proportionate to risk |
| `aml/typologies.py` | JMLSG Part I para 5.3, FATF R.20 | Typology coverage and STR filing |
| `aml/sar_scorer.py` | POCA 2002 s.330 | Mandatory disclosure to NCA |
| `aml/evaluation.py` | MLR 2017 Reg 19, FCA SYSC 6.3 | Alert volume vs. review capacity |
| `fraud/explain.py` | FCA SYSC 10A, DP5/22 | Automated decision explainability |
| `sql/schema.sql` | MLR 2017 Reg 40, SYSC 10A | Five-year retention; audit trail |
| Model validation | PRA SS1/23, SR 11-7 | Champion/challenger governance |
| Graph analytics | FATF R.10, FATF R.16 | CDD on unusual patterns; wire monitoring |

Full per-module mapping: [`docs/regulatory.md`](docs/regulatory.md).

---

## Running tests

```bash
# Full suite with coverage report
pytest tests/ -v --cov=fincrime_ml --cov-report=term-missing

# Fraud domain only
pytest tests/test_fraud/ -v

# AML domain only
pytest tests/test_aml/ -v

# Core modules only
pytest tests/test_core/ -v
```

Coverage gate: 80% minimum (currently 97%). Tests use synthetic data only; no external
data files are required.

---

## Project structure

```
fincrime_ml/
├── core/
│   ├── base.py                    # BasePipeline, BaseScorer
│   ├── scorer.py                  # FinCrimeScorer (fraud + AML fusion)
│   └── data/
│       ├── synth_cards.py         # Synthetic card + SWIFT generator
│       ├── synth_aml.py           # PaySim-style AML generator
│       ├── typology_injector.py   # AML typology flag injector
│       └── loaders.py             # IEEE-CIS + PaySim adapters
├── fraud/
│   ├── features.py                # Velocity, deviation, MCC risk features
│   ├── imbalance.py               # SMOTE vs cost-sensitive benchmark
│   ├── explain.py                 # SHAP explainability
│   ├── evaluation.py              # AUC-PR, threshold analysis, cost matrix
│   └── models/
│       ├── xgb_classifier.py      # XGBoost champion
│       └── logistic_baseline.py   # Logistic challenger
└── aml/
    ├── typologies.py              # Structuring, layering, integration, mule
    ├── graph.py                   # NetworkX transaction graph builder
    ├── sar_scorer.py              # SAR trigger scorer (POCA 2002 s.330)
    ├── evaluation.py              # Alert fatigue evaluator
    └── models/
        ├── graph_scorer.py        # Centrality + flow deviation scorer
        └── isolation_forest.py    # Unsupervised AML baseline

tests/                             # 850 tests · 97% coverage
docs/
├── adr/001-dual-domain.md         # Architecture decision record
├── model_validation.md            # PRA SS1/23 validation report
└── regulatory.md                  # FCA/JMLSG/FATF module mapping
sql/
├── schema.sql                     # MySQL 8.0+ transaction monitoring schema
└── queries/
    ├── velocity.sql               # Rolling-window velocity queries
    └── aml_alerts.sql             # MLRO queue and MI queries
dashboard/
└── index.html                     # Standalone MLRO monitoring dashboard
notebooks/
├── fraud_detection.ipynb          # End-to-end fraud walkthrough
└── aml_monitoring.ipynb           # AML typology walkthrough
```

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before
opening a pull request.

**Quick guide:**

```bash
git clone https://github.com/TemidayoA/fincrime-ml.git
cd fincrime-ml
pip install -e ".[dev]"
pre-commit install

# Make changes, then
pytest tests/ -q
python -m black .
python -m ruff check --fix .
```

All pull requests must pass the CI pipeline (black, ruff, pytest at 80% coverage minimum).
Domain separation must be preserved: fraud and AML modules must not import from each other.

---

## Datasets

| Dataset | Domain | Source | Adapter |
|---|---|---|---|
| Synthetic (built-in) | Fraud + AML | Built-in generator | `synth_cards.py`, `synth_aml.py` |
| IEEE-CIS Fraud Detection | Fraud | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) | `loaders.py` |
| PaySim Mobile Money | AML | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) | `loaders.py` |

No raw data files are committed to this repository. All tests use the built-in synthetic
generators and require no external downloads.

---

## Author

**Temidayo Akindahunsi** — Machine Learning Engineer, fintech analytics.
Built on production experience with UK FCA-regulated consumer finance and collections systems.

---

## Licence

MIT — see [LICENSE](LICENSE).
