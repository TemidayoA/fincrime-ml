# Commit schedule — FinCrime-ML (30 days)

Claude Code: run `git log --oneline -5` to find where you are, then execute
the next uncommitted day below. One day = one commit. Do not combine days.

Format: `STATUS | DATE_TARGET | COMMIT_MESSAGE | FILES`

---

## Week 1 — Foundation & synthetic data

| Day | Status | Commit message | Primary files |
|-----|--------|----------------|---------------|
| 1 | ✅ DONE | `infra(core): initial repo scaffold — pyproject, CI, base classes, MIT licence` | `pyproject.toml`, `.gitignore`, `LICENSE`, `CHANGELOG.md`, `.github/workflows/ci.yml`, `fincrime_ml/core/base.py` |
| 2 | ✅ DONE | `docs(adr): dual-domain architecture decision record — fraud vs AML separation` | `docs/adr/001-dual-domain.md` |
| 3 | ✅ DONE | `data(core): synthetic card transaction generator v1 — card, SWIFT, fraud typology injection` | `fincrime_ml/core/data/synth_cards.py`, `tests/test_core/test_synth_cards.py` |
| 4 | ⬜ TODO | `data(core): synthetic generator v2 — digital payments, mule chain seeds, AML structuring patterns` | `fincrime_ml/core/data/synth_cards.py` (extend), `fincrime_ml/core/data/synth_aml.py` |
| 5 | ⬜ TODO | `feat(core): fraud typology injector — CNP, ATO, bust-out, card-present skimming` | `fincrime_ml/core/data/typology_injector.py`, `tests/test_core/test_typology_injector.py` |
| 6 | ⬜ TODO | `data(core): IEEE-CIS fraud dataset loader + schema harmoniser` | `fincrime_ml/core/data/loaders.py`, `tests/test_core/test_loaders.py` |
| 7 | ⬜ TODO | `docs(core): data dictionary v1 — transaction schema, FCA/JMLSG regulatory field mappings` | `docs/data_dictionary.md` |

---

## Week 2 — Fraud detection pipeline

| Day | Status | Commit message | Primary files |
|-----|--------|----------------|---------------|
| 8  | ⬜ TODO | `feat(fraud): feature engineering — velocity windows, amount deviation, MCC risk score` | `fincrime_ml/fraud/features.py`, `tests/test_fraud/test_features.py` |
| 9  | ⬜ TODO | `feat(fraud): class imbalance handler — SMOTE vs cost-sensitive weighting benchmark` | `fincrime_ml/fraud/imbalance.py`, `tests/test_fraud/test_imbalance.py` |
| 10 | ⬜ TODO | `model(fraud): XGBoost classifier v1 — AUC-PR optimisation, cross-validation harness` | `fincrime_ml/fraud/models/xgb_classifier.py`, `tests/test_fraud/test_xgb_classifier.py` |
| 11 | ⬜ TODO | `model(fraud): logistic regression baseline + feature importance comparison` | `fincrime_ml/fraud/models/logistic_baseline.py`, `tests/test_fraud/test_logistic_baseline.py` |
| 12 | ⬜ TODO | `feat(fraud): SHAP explainability layer — per-prediction reason codes, FCA SR11-7 alignment` | `fincrime_ml/fraud/explain.py`, `tests/test_fraud/test_explain.py` |
| 13 | ⬜ TODO | `eval(fraud): evaluation suite — threshold analysis, false positive cost matrix` | `fincrime_ml/fraud/evaluation.py`, `tests/test_fraud/test_evaluation.py` |
| 14 | ⬜ TODO | `docs(fraud): end-to-end fraud detection notebook with domain expert commentary` | `notebooks/fraud_detection.ipynb` |

---

## Week 3 — AML monitoring pipeline

| Day | Status | Commit message | Primary files |
|-----|--------|----------------|---------------|
| 15 | ⬜ TODO | `feat(aml): typology engine — structuring, layering, integration pattern detection (FATF-aligned)` | `fincrime_ml/aml/typologies.py`, `tests/test_aml/test_typologies.py` |
| 16 | ⬜ TODO | `feat(aml): transaction network graph builder — NetworkX entity relationship mapping` | `fincrime_ml/aml/graph.py`, `tests/test_aml/test_graph.py` |
| 17 | ⬜ TODO | `model(aml): graph-based anomaly scorer — centrality and flow deviation features` | `fincrime_ml/aml/models/graph_scorer.py`, `tests/test_aml/test_graph_scorer.py` |
| 18 | ⬜ TODO | `data(aml): PaySim mobile money dataset integration — mule chain annotation layer` | `fincrime_ml/core/data/loaders.py` (extend) |
| 19 | ⬜ TODO | `model(aml): isolation forest unsupervised baseline — no-label scenario` | `fincrime_ml/aml/models/isolation_forest.py`, `tests/test_aml/test_isolation_forest.py` |
| 20 | ⬜ TODO | `feat(aml): SAR trigger scoring — alert prioritisation with MLRO-ready audit output` | `fincrime_ml/aml/sar_scorer.py`, `tests/test_aml/test_sar_scorer.py` |
| 21 | ⬜ TODO | `eval(aml): alert fatigue metric — false positive rate at configurable sensitivity levels` | `fincrime_ml/aml/evaluation.py`, `tests/test_aml/test_evaluation.py` |
| 22 | ⬜ TODO | `docs(aml): AML monitoring notebook — typology walkthrough with JMLSG regulatory commentary` | `notebooks/aml_monitoring.ipynb` |

---

## Week 4 — Integration, dashboard & release

| Day | Status | Commit message | Primary files |
|-----|--------|----------------|---------------|
| 23 | ⬜ TODO | `feat(core): unified FinCrime risk scorer — configurable fraud + AML signal fusion` | `fincrime_ml/core/scorer.py`, `tests/test_core/test_scorer.py` |
| 24 | ⬜ TODO | `feat(dashboard): HTML monitoring dashboard — real-time alert queue with SHAP reason codes` | `dashboard/index.html` |
| 25 | ⬜ TODO | `infra(sql): transaction monitoring schema + query library — MySQL-compatible` | `sql/schema.sql`, `sql/queries/velocity.sql`, `sql/queries/aml_alerts.sql` |
| 26 | ⬜ TODO | `eval(core): model validation report — holdout backtesting, champion/challenger framework` | `docs/model_validation.md` |
| 27 | ⬜ TODO | `infra(ci): pre-commit hooks, coverage badge, ruff + black enforcement` | `.pre-commit-config.yaml`, `README.md` (badge update) |
| 28 | ⬜ TODO | `docs(core): regulatory alignment guide — FCA/JMLSG/FATF mapping per module` | `docs/regulatory.md` |
| 29 | ⬜ TODO | `docs(core): comprehensive README — architecture diagram, quickstart, use case guide` | `README.md` |
| 30 | ⬜ TODO | `infra(ci): v0.1.0 release — CHANGELOG final, PyPI-ready packaging, contributor guide` | `CHANGELOG.md`, `CONTRIBUTING.md`, git tag `v0.1.0` |

---

## Git commands for each day

```bash
# Check where you are
git log --oneline -5
git status

# Stage and commit (replace with today's message from table above)
git add <files>
git commit -m "<commit message from table>"

# Push
git push origin main
```

## Initial push (first time only)

```bash
cd fincrime-ml
git init
git add .
git commit -m "infra(core): initial repo scaffold — pyproject, CI, base classes, MIT licence"
git branch -M main
git remote add origin https://github.com/TemidayoA/fincrime-ml.git
git push -u origin main
```

Then make Day 2 and Day 3 as separate commits immediately after.
