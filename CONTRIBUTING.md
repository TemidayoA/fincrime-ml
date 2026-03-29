# Contributing to FinCrime-ML

Thank you for your interest in contributing. This project welcomes issues,
bug reports, and pull requests from the community.

## Getting started

```bash
git clone https://github.com/TemidayoA/fincrime-ml.git
cd fincrime-ml
pip install -e ".[dev]"
pre-commit install
```

## Pull request process

1. Open an issue first for significant changes
2. Fork the repo and create a feature branch: `git checkout -b feat/your-feature`
3. Write tests for your changes — coverage must not drop below 80%
4. Run `black fincrime_ml/ && ruff check fincrime_ml/ && pytest tests/`
5. Submit a PR against `main` with a clear description

## Code standards

- Python 3.11+, Black formatting, Ruff linting
- Type hints on all public methods
- Google-style docstrings on all public classes and functions
- No hardcoded credentials or data file commits

## Regulatory note

This framework is designed for research and educational use. Deployments in
regulated financial institutions must undergo independent model validation
per PRA SS1/23 and SR 11-7 requirements.
