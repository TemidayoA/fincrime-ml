# FinCrime-ML

[![CI](https://github.com/TemidayoA/fincrime-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/TemidayoA/fincrime-ml/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A dual-domain open-source Python framework for financial crime detection — covering
**payment fraud** and **AML transaction monitoring** — built with ML rigour and
regulatory alignment for the UK banking context.

---

## Why this exists

Most financial crime ML tutorials treat fraud detection as a single binary classification
problem on a Kaggle dataset. In practice, UK-regulated institutions face two distinct
supervisory regimes — PSR 2017 / PSD2 for fraud, and POCA 2002 / MLR 2017 for AML —
with fundamentally different alert workflows, operational latency requirements, and
model performance priorities.

FinCrime-ML models this separation explicitly. The fraud and AML pipelines are independent
modules with a shared core layer. A unified risk scorer optionally combines both signals.

---

## Architecture

```
fincrime_ml/
├── core/
│   ├── base.py          # BasePipeline, BaseScorer — abstract interfaces
│   ├── scorer.py        # Unified fraud + AML risk scorer
│   └── data/
│       ├── synth_cards.py   # Synthetic card + SWIFT transaction generator
│       └── loaders.py       # IEEE-CIS and PaySim dataset adapters
├── fraud/
│   ├── features.py      # Velocity, deviation, MCC risk features
│   ├── imbalance.py     # SMOTE vs cost-sensitive weighting comparison
│   ├── explain.py       # SHAP explainability layer
│   └── models/
│       ├── xgb_classifier.py    # XGBoost with AUC-PR optimisation
│       └── logistic_baseline.py # Interpretable benchmark
├── aml/
│   ├── typologies.py    # Structuring, layering, integration detection
│   ├── graph.py         # NetworkX entity relationship mapping
│   ├── sar_scorer.py    # SAR trigger prioritisation
│   └── models/
│       ├── graph_scorer.py        # Centrality + flow deviation features
│       └── isolation_forest.py   # Unsupervised baseline (no-label scenario)
docs/
├── adr/
│   └── 001-dual-domain.md   # Architecture decision record
└── regulatory.md            # FCA / JMLSG / FATF module mapping
notebooks/
├── fraud_detection.ipynb    # End-to-end fraud walkthrough
└── aml_monitoring.ipynb     # AML typology walkthrough
sql/
├── schema.sql               # Transaction monitoring database schema
└── queries/                 # Common TM SQL query library
dashboard/
└── index.html               # Standalone HTML monitoring dashboard
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/TemidayoA/fincrime-ml.git
cd fincrime-ml
pip install -e ".[dev]"
```

### Generate synthetic transaction data

```python
from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator

gen = SyntheticTransactionGenerator(n_accounts=5_000, seed=42)

# 50,000 card transactions with ~1.5% fraud rate
df = gen.generate(n_transactions=50_000, fraud_rate=0.015)
print(df.shape)          # (50000, 24)
print(df["is_fraud"].value_counts())

# SWIFT wire transfers with AML structuring patterns
wires = gen.generate_wire_transfers(n=5_000)
print(wires[wires["is_structured_amount"]].head())
```

### Train the fraud detection pipeline

```python
from fincrime_ml.fraud.models.xgb_classifier import XGBFraudClassifier

clf = XGBFraudClassifier()
clf.train(df, label_col="is_fraud")

scores = clf.predict(df_holdout)
print(scores[["transaction_id", "risk_score", "risk_tier"]].head(10))

# SHAP reason codes per transaction
explanations = clf.explain(df_holdout.head(100))
print(explanations[["transaction_id", "top_reason_1", "top_reason_2"]].head())
```

### Run AML monitoring

```python
from fincrime_ml.aml.typologies import TypologyEngine
from fincrime_ml.aml.sar_scorer import SARScorer

engine = TypologyEngine()
flagged = engine.detect(wires)   # Returns transactions matching AML typologies

scorer = SARScorer()
prioritised = scorer.score(flagged)
print(prioritised[prioritised["risk_tier"] == "CRITICAL"])
```

---

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary eval metric | AUC-PR (not ROC-AUC) | Imbalanced classes; precision-recall tradeoff is operationally meaningful |
| Fraud/AML separation | Independent modules | Regulatory tipping-off constraint; distinct feature spaces and alert workflows |
| Explainability | SHAP (not LIME) | Additive attribution consistent with FCA SR 11-7 model risk requirements |
| AML baseline | Isolation forest | Labels are sparse/unreliable; unsupervised approach is more realistic |
| Data | Synthetic + IEEE-CIS + PaySim | Reproducible open-source research without proprietary data |

Full rationale in [`docs/adr/001-dual-domain.md`](docs/adr/001-dual-domain.md).

---

## Regulatory alignment

This framework is designed with UK regulatory context in mind:

| Module | Framework |
|---|---|
| Fraud pipeline | PSR 2017, PSD2, FCA PROD |
| AML pipeline | POCA 2002 s.330, MLR 2017, JMLSG Part I Ch.5 |
| Model validation | PRA SS1/23, SR 11-7 |
| SAR scoring | NCA guidance, FATF 40 Recommendations |
| Audit logging | FCA SYSC 10A |

Full mapping: [`docs/regulatory.md`](docs/regulatory.md).

---

## Datasets

| Dataset | Domain | Source | Adapter |
|---|---|---|---|
| Synthetic cards | Fraud | Built-in generator | `synth_cards.py` |
| IEEE-CIS Fraud Detection | Fraud | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) | `loaders.py` |
| PaySim Mobile Money | AML | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) | `loaders.py` |

No raw dataset files are committed. Adapters map each dataset's schema to the
FinCrime-ML unified transaction schema.

---

## Running tests

```bash
pytest tests/ -v
```

Coverage target: 80%. Current badge reflects CI status on `main`.

---

## Contributing

Contributions welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and open
an issue before submitting a pull request for significant changes.

---

## Author

**Temidayo Akindahunsi** — Chief Data Officer, fintech analytics specialist.  
Built on a foundation of production collections and credit risk data systems
in UK FCA-regulated consumer finance.

---

## Licence

MIT — see [LICENSE](LICENSE).
