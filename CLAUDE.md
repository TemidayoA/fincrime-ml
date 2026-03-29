# CLAUDE.md — FinCrime-ML project instructions for Claude Code

This file tells Claude Code everything it needs to know to work on this project
autonomously. Read it fully before doing anything else in this repo.

---

## Project overview

**fincrime-ml** is a dual-domain open-source Python framework for financial crime
detection. It covers payment fraud and AML transaction monitoring with ML models,
SHAP explainability, and UK regulatory alignment (FCA, JMLSG, FATF).

**GitHub:** https://github.com/TemidayoA/fincrime-ml  
**Author:** Temidayo Akindahunsi (@TemidayoA)  
**Purpose:** Global Talent Visa (Exceptional Promise) portfolio + genuine open-source tool

---

## Commit discipline — CRITICAL

This project must be built with a **natural, gradual commit history** over 30 days.
Do NOT bulk-commit everything at once. Each session should produce 1–2 commits max,
matching the schedule in `COMMIT_SCHEDULE.md`.

### Commit message format

```
<type>(<scope>): <short description>

[optional body — 2-3 lines max]
```

Types: `feat`, `data`, `model`, `eval`, `docs`, `infra`, `test`, `fix`  
Scopes: `core`, `fraud`, `aml`, `dashboard`, `sql`, `ci`

Examples:
```
feat(fraud): add velocity feature engineering — 1h/24h/7d rolling windows

Computes per-account transaction velocity across three time windows.
High velocity relative to account baseline is the strongest single
predictor of CNP fraud in UK Finance 2023 data.
```

```
model(fraud): XGBoost classifier v1 with AUC-PR optimisation

Uses scale_pos_weight to handle class imbalance.
Primary metric: average_precision (not ROC-AUC — misleading on imbalanced data).
```

---

## Architecture rules

1. **Never import from `fraud/` into `aml/` or vice versa.** Only `core/` is shared.
2. **All pipeline classes must inherit from `BasePipeline`** in `core/base.py`.
3. **All scoring classes must inherit from `BaseScorer`** in `core/base.py`.
4. **Label column convention:** fraud pipelines use `is_fraud`, AML uses `is_suspicious`.
5. **Primary metric is AUC-PR** (average_precision_score), not ROC-AUC. Always report both but optimise on AUC-PR.
6. **SHAP explanations are mandatory** on all supervised models. No model ships without an `explain()` method.
7. **Audit logging must be called** in every `predict()` implementation via `self._log_audit()`.

---

## Code style

- Python 3.11+
- Black formatting, line length 100
- Ruff linting
- Type hints on all public methods
- Docstrings on all public classes and methods (Google style)
- No bare `except:` — always catch specific exceptions

Run before committing:
```bash
black fincrime_ml/ tests/
ruff check fincrime_ml/ tests/
pytest tests/ -v
```

---

## File-by-file build plan

Work through these in order. Each maps to a day in `COMMIT_SCHEDULE.md`.

### Phase 1 — Core / data (Days 1–7) ✅ DONE
- [x] `pyproject.toml` — package config
- [x] `fincrime_ml/core/base.py` — BasePipeline, BaseScorer
- [x] `fincrime_ml/core/data/synth_cards.py` — card + SWIFT generator
- [x] `docs/adr/001-dual-domain.md` — architecture decision record
- [x] `.github/workflows/ci.yml` — GitHub Actions
- [x] `tests/test_core/test_synth_cards.py` — generator tests

### Phase 2 — Fraud pipeline (Days 8–14)
- [ ] `fincrime_ml/core/data/loaders.py` — IEEE-CIS + PaySim adapters (Day 6)
- [ ] `docs/data_dictionary.md` — schema + regulatory field mappings (Day 7)
- [ ] `fincrime_ml/fraud/features.py` — velocity, deviation, MCC risk features (Day 8)
- [ ] `fincrime_ml/fraud/imbalance.py` — SMOTE vs cost-sensitive comparison (Day 9)
- [ ] `fincrime_ml/fraud/models/xgb_classifier.py` — XGBoost + AUC-PR (Day 10)
- [ ] `fincrime_ml/fraud/models/logistic_baseline.py` — interpretable benchmark (Day 11)
- [ ] `fincrime_ml/fraud/explain.py` — SHAP explainability layer (Day 12)
- [ ] `fincrime_ml/fraud/evaluation.py` — threshold analysis, cost matrix (Day 13)
- [ ] `notebooks/fraud_detection.ipynb` — end-to-end walkthrough (Day 14)
- [ ] `tests/test_fraud/` — fraud pipeline tests

### Phase 3 — AML pipeline (Days 15–22)
- [ ] `fincrime_ml/aml/typologies.py` — structuring, layering, integration (Day 15)
- [ ] `fincrime_ml/aml/graph.py` — NetworkX entity relationship mapping (Day 16)
- [ ] `fincrime_ml/aml/models/graph_scorer.py` — centrality + flow features (Day 17)
- [ ] `fincrime_ml/core/data/loaders.py` — PaySim adapter additions (Day 18)
- [ ] `fincrime_ml/aml/models/isolation_forest.py` — unsupervised baseline (Day 19)
- [ ] `fincrime_ml/aml/sar_scorer.py` — SAR trigger prioritisation (Day 20)
- [ ] `fincrime_ml/aml/evaluation.py` — alert fatigue metric (Day 21)
- [ ] `notebooks/aml_monitoring.ipynb` — typology walkthrough (Day 22)
- [ ] `tests/test_aml/` — AML pipeline tests

### Phase 4 — Integration + release (Days 23–30)
- [ ] `fincrime_ml/core/scorer.py` — unified fraud + AML signal fusion (Day 23)
- [ ] `dashboard/index.html` — standalone HTML monitoring dashboard (Day 24)
- [ ] `sql/schema.sql` + `sql/queries/` — TM database layer (Day 25)
- [ ] `docs/model_validation.md` — backtesting, champion/challenger framework (Day 26)
- [ ] CI/CD enhancements — coverage badge, pre-commit hooks (Day 27)
- [ ] `docs/regulatory.md` — FCA/JMLSG/FATF module mapping (Day 28)
- [ ] `README.md` — final version with badges + architecture diagram (Day 29)
- [ ] `v0.1.0` release tag + CHANGELOG final (Day 30)

---

## When building each file

1. Check `COMMIT_SCHEDULE.md` for the day's target and commit message
2. Build only what's scoped to that day — don't jump ahead
3. Run tests before committing
4. Use the exact commit message format from the schedule
5. Never commit `.csv`, `.pkl`, `.parquet`, or any data files
6. Never commit secrets, API keys, or credentials

---

## Testing conventions

- All test files in `tests/test_<domain>/test_<module>.py`
- Use `pytest` fixtures for shared setup
- Every public method should have at least one test
- Test edge cases: empty DataFrames, single-row inputs, zero fraud rate
- Coverage target: 80% minimum

---

## Regulatory context (important for documentation)

When writing docstrings, ADRs, or notebook commentary, reference these frameworks
where relevant — they signal domain expertise:

| Framework | Relevance |
|---|---|
| FCA SYSC 6.3 | Financial crime systems and controls |
| POCA 2002 s.330 | Failure to disclose — AML tipping-off constraint |
| MLR 2017 | Money Laundering Regulations — AML obligations |
| PRA SS1/23 | Model risk management principles |
| SR 11-7 | Model risk management (US Fed — widely referenced in UK too) |
| JMLSG Part I Ch.5 | Guidance on transaction monitoring |
| FATF 40 Recommendations | International AML standards |
| PSR 2017 / PSD2 | Payment fraud obligations |
| FCA SYSC 10A | Record-keeping for automated decision systems |

---

## Session workflow for Claude Code

When starting a new session:

1. Run `cat COMMIT_SCHEDULE.md` to see what's due today
2. Run `git log --oneline -10` to see where we are
3. Build the file(s) for the current day
4. Run `pytest tests/ -v` — fix any failures
5. Run `black fincrime_ml/ && ruff check fincrime_ml/`
6. Commit with the scheduled message
7. Update the checkbox in this file's build plan above

---

## GitHub remote

```bash
git remote add origin https://github.com/TemidayoA/fincrime-ml.git
```

Main branch: `main`  
No force pushes to `main`.
