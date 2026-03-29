# Changelog

All notable changes to FinCrime-ML are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### In progress
- Fraud feature engineering module (`fraud/features.py`)
- XGBoost fraud classifier with SHAP explainability
- AML typology engine and transaction network graph
- Unified FinCrime risk scorer
- HTML monitoring dashboard

---

## [0.1.0] — 2024-01-30 (target)

### Added
- Initial package scaffold and `pyproject.toml`
- `BasePipeline` and `BaseScorer` abstract base classes with audit logging
- `SyntheticTransactionGenerator` — card, SWIFT, and digital payment data
- Fraud typology injector (CNP, skimming, ATO, bust-out)
- IEEE-CIS fraud dataset adapter
- Architecture Decision Record: dual-domain design rationale
- Data dictionary with FCA/JMLSG regulatory field mappings
- GitHub Actions CI pipeline (Python 3.11, 3.12)
- MIT licence
