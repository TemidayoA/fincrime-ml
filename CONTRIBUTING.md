# Contributing to FinCrime-ML

Thank you for your interest in contributing. This document covers the development
setup, code standards, domain separation rules, and pull request process.

---

## Contents

- [Getting started](#getting-started)
- [Domain separation rule](#domain-separation-rule)
- [Code standards](#code-standards)
- [Testing requirements](#testing-requirements)
- [Pull request process](#pull-request-process)
- [Commit message convention](#commit-message-convention)
- [Adding a new model](#adding-a-new-model)
- [Regulatory note](#regulatory-note)

---

## Getting started

```bash
git clone https://github.com/TemidayoA/fincrime-ml.git
cd fincrime-ml
pip install -e ".[dev]"
pre-commit install
```

Verify the setup:

```bash
pytest tests/ -q      # should pass with >= 80% coverage
python -m black --check .
python -m ruff check .
```

---

## Domain separation rule

The fraud and AML modules must not import from each other. This constraint is enforced
at code review and reflects the regulatory tipping-off prohibition under POCA 2002 s.333A:
the fraud investigation workflow and the SAR disclosure workflow must remain operationally
separate.

The permitted import pattern is:

```
fincrime_ml.fraud.*   →  fincrime_ml.core.*   (allowed)
fincrime_ml.aml.*     →  fincrime_ml.core.*   (allowed)
fincrime_ml.core.*    →  fincrime_ml.core.*   (allowed)
fincrime_ml.fraud.*   →  fincrime_ml.aml.*    (PROHIBITED)
fincrime_ml.aml.*     →  fincrime_ml.fraud.*  (PROHIBITED)
```

Pull requests that introduce cross-domain imports will not be merged.

---

## Code standards

- **Python version:** 3.11 or higher.
- **Formatting:** Black with `line-length = 100`. Run `python -m black .` before committing.
- **Linting:** Ruff. Run `python -m ruff check --fix .` before committing.
- **Type hints:** Required on all public method signatures.
- **Docstrings:** Single-line for obvious methods; multi-line only when the reason is
  non-obvious. No boilerplate "This function does X" descriptions.
- **Comments:** Only when the reason is non-obvious (hidden constraint, known bug workaround,
  subtle invariant). No narrative comments describing what the code does.
- **Metrics:** AUC-PR is the primary evaluation metric for all classification tasks in this
  framework. Do not introduce ROC-AUC as a primary metric.
- **No hardcoded credentials, data file paths, or API keys.**

---

## Testing requirements

- All new code must be accompanied by tests in the corresponding `tests/` subdirectory.
- Coverage must not drop below 80% on `fincrime_ml/` (currently 97%).
- Tests must use the built-in synthetic data generators; do not commit real transaction data.
- Use `pytest` fixtures for repeated setup; avoid duplicated test data construction.
- Test file naming: `tests/test_<domain>/test_<module>.py`.

Run the full suite before submitting:

```bash
pytest tests/ -v --cov=fincrime_ml --cov-report=term-missing
```

---

## Pull request process

1. Open an issue first for non-trivial changes to align on the approach.
2. Fork the repository and create a feature branch from `main`:
   `git checkout -b feat/your-feature-name`
3. Make your changes, write tests, and ensure the full test suite passes.
4. Run black and ruff to ensure formatting and linting are clean.
5. Push your branch and open a pull request against `main`.
6. Describe what you changed and why. Reference the relevant regulatory obligation
   if the change affects fraud detection, AML monitoring, or the audit trail.

---

## Commit message convention

This project uses a structured prefix convention:

| Prefix | Use for |
|---|---|
| `feat(domain):` | New feature in `fraud/`, `aml/`, or `core/` |
| `model(domain):` | New or updated ML model |
| `data(domain):` | Data generator, loader, or schema change |
| `eval(domain):` | Evaluation, validation, or metric tooling |
| `docs(domain):` | Documentation — READMEs, guides, notebooks |
| `infra(domain):` | CI, packaging, pre-commit, schema |
| `fix(domain):` | Bug fix |
| `test(domain):` | Test additions or fixes |

Examples:

```
feat(aml): add FATF R.16 cross-border velocity trigger rule
model(fraud): XGBoost v2 — retrain on 2026 Q1 data
fix(core): handle NaN in unified scorer when AML score absent
```

---

## Adding a new model

1. Create `fincrime_ml/<domain>/models/<your_model>.py`.
2. Inherit from `BaseScorer` in `fincrime_ml/core/base.py`.
3. Implement `train()`, `predict()`, and `score()`.
4. Register the model in `model_versions` via the SQL registry if deploying to a
   database-backed environment.
5. Write tests in `tests/test_<domain>/test_<your_model>.py` covering training,
   prediction schema, edge cases, and any threshold logic.
6. Add a champion/challenger entry to `docs/model_validation.md` with AUC-PR and
   the promotion criteria.

---

## Regulatory note

This framework is designed for research, portfolio demonstration, and educational use.
Deployments in FCA-regulated or PRA-regulated institutions require independent model
validation under PRA SS1/23, formal model risk governance documentation, and sign-off
by a qualified MLRO or model risk officer before use in a production transaction
monitoring system.

The regulatory references in this codebase (POCA 2002, MLR 2017, JMLSG, FATF) are
provided for context and educational purposes. They do not constitute legal advice.
