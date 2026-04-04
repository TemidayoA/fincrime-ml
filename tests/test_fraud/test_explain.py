"""
tests/test_fraud/test_explain.py
==================================
Unit tests for the SHAP explainability layer.
"""

import numpy as np
import pandas as pd
import pytest

from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator
from fincrime_ml.fraud.explain import FraudExplainer, build_explainer, top_reason_codes
from fincrime_ml.fraud.models.logistic_baseline import LogisticFraudBaseline
from fincrime_ml.fraud.models.xgb_classifier import FEATURE_COLS, XGBFraudClassifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    gen = SyntheticTransactionGenerator(n_accounts=200, seed=42)
    return gen.generate(n_transactions=1_000, fraud_rate=0.10)


@pytest.fixture(scope="module")
def fitted_xgb(train_df) -> XGBFraudClassifier:
    clf = XGBFraudClassifier(n_cv_folds=2, seed=42)
    clf.train(train_df)
    return clf


@pytest.fixture(scope="module")
def fitted_lr(train_df) -> LogisticFraudBaseline:
    baseline = LogisticFraudBaseline(n_cv_folds=2, seed=42)
    baseline.train(train_df)
    return baseline


@pytest.fixture(scope="module")
def feature_matrix(train_df, fitted_xgb) -> np.ndarray:
    return fitted_xgb.prepare_features(train_df).to_numpy(dtype=float)


@pytest.fixture(scope="module")
def xgb_explainer(fitted_xgb, feature_matrix) -> FraudExplainer:
    explainer = FraudExplainer(fitted_xgb.model, feature_names=FEATURE_COLS)
    explainer.fit()
    return explainer


@pytest.fixture(scope="module")
def lr_explainer(fitted_lr, feature_matrix) -> FraudExplainer:
    explainer = FraudExplainer(fitted_lr.model, feature_names=FEATURE_COLS)
    explainer.fit(X_background=feature_matrix[:100])
    return explainer


# ---------------------------------------------------------------------------
# Model type detection tests
# ---------------------------------------------------------------------------


def test_xgb_detected_as_tree(fitted_xgb):
    explainer = FraudExplainer(fitted_xgb.model, FEATURE_COLS)
    assert explainer._model_type == "tree"


def test_lr_detected_as_linear(fitted_lr):
    explainer = FraudExplainer(fitted_lr.model, FEATURE_COLS)
    assert explainer._model_type == "linear"


def test_unknown_model_detected_as_kernel():
    class FakeModel:
        pass

    explainer = FraudExplainer(FakeModel(), FEATURE_COLS)
    assert explainer._model_type == "kernel"


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------


def test_fit_sets_shap_explainer(fitted_xgb):
    explainer = FraudExplainer(fitted_xgb.model, FEATURE_COLS)
    assert explainer.shap_explainer is None
    explainer.fit()
    assert explainer.shap_explainer is not None


def test_fit_returns_self(fitted_xgb):
    explainer = FraudExplainer(fitted_xgb.model, FEATURE_COLS)
    result = explainer.fit()
    assert result is explainer


def test_fit_linear_with_background(fitted_lr, feature_matrix):
    explainer = FraudExplainer(fitted_lr.model, FEATURE_COLS)
    explainer.fit(X_background=feature_matrix[:50])
    assert explainer.shap_explainer is not None


def test_fit_kernel_raises_without_background():
    class FakeModel:
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    explainer = FraudExplainer(FakeModel(), ["f1"])
    with pytest.raises(ValueError, match="KernelExplainer requires X_background"):
        explainer.fit()


# ---------------------------------------------------------------------------
# shap_values() tests
# ---------------------------------------------------------------------------


def test_shap_values_shape_xgb(xgb_explainer, feature_matrix):
    values = xgb_explainer.shap_values(feature_matrix[:20])
    assert values.shape == (20, len(FEATURE_COLS))


def test_shap_values_shape_lr(lr_explainer, feature_matrix):
    values = lr_explainer.shap_values(feature_matrix[:20])
    assert values.shape == (20, len(FEATURE_COLS))


def test_shap_values_finite(xgb_explainer, feature_matrix):
    values = xgb_explainer.shap_values(feature_matrix[:20])
    assert np.isfinite(values).all()


def test_shap_values_accepts_dataframe(xgb_explainer, train_df, fitted_xgb):
    X_df = fitted_xgb.prepare_features(train_df.head(10))
    values = xgb_explainer.shap_values(X_df)
    assert values.shape == (10, len(FEATURE_COLS))


def test_shap_values_raises_before_fit(fitted_xgb, feature_matrix):
    explainer = FraudExplainer(fitted_xgb.model, FEATURE_COLS)
    with pytest.raises(RuntimeError, match="not been fitted"):
        explainer.shap_values(feature_matrix[:5])


# ---------------------------------------------------------------------------
# reason_codes() tests
# ---------------------------------------------------------------------------


def test_reason_codes_returns_dataframe(xgb_explainer, feature_matrix):
    result = xgb_explainer.reason_codes(feature_matrix[:10])
    assert isinstance(result, pd.DataFrame)


def test_reason_codes_row_count(xgb_explainer, feature_matrix):
    result = xgb_explainer.reason_codes(feature_matrix[:15])
    assert len(result) == 15


def test_reason_codes_has_transaction_id(xgb_explainer, feature_matrix):
    result = xgb_explainer.reason_codes(feature_matrix[:5])
    assert "transaction_id" in result.columns


def test_reason_codes_top_n_columns_present(xgb_explainer, feature_matrix):
    result = xgb_explainer.reason_codes(feature_matrix[:5], top_n=3)
    for i in range(1, 4):
        assert f"top_reason_{i}" in result.columns
        assert f"shap_{i}" in result.columns
        assert f"direction_{i}" in result.columns


def test_reason_codes_features_are_valid(xgb_explainer, feature_matrix):
    result = xgb_explainer.reason_codes(feature_matrix[:20], top_n=2)
    assert result["top_reason_1"].isin(FEATURE_COLS).all()
    assert result["top_reason_2"].isin(FEATURE_COLS).all()


def test_reason_codes_direction_values(xgb_explainer, feature_matrix):
    result = xgb_explainer.reason_codes(feature_matrix[:10], top_n=2)
    valid = {"increases_risk", "reduces_risk"}
    assert set(result["direction_1"].unique()).issubset(valid)
    assert set(result["direction_2"].unique()).issubset(valid)


def test_reason_codes_custom_transaction_ids(xgb_explainer, feature_matrix):
    ids = [f"TXN-{i:04d}" for i in range(10)]
    result = xgb_explainer.reason_codes(feature_matrix[:10], transaction_ids=ids)
    assert list(result["transaction_id"]) == ids


def test_reason_codes_shap_values_are_floats(xgb_explainer, feature_matrix):
    result = xgb_explainer.reason_codes(feature_matrix[:5], top_n=1)
    assert pd.api.types.is_float_dtype(result["shap_1"])


def test_reason_codes_raises_before_fit(fitted_xgb, feature_matrix):
    explainer = FraudExplainer(fitted_xgb.model, FEATURE_COLS)
    with pytest.raises(RuntimeError, match="not been fitted"):
        explainer.reason_codes(feature_matrix[:5])


# ---------------------------------------------------------------------------
# feature_summary() tests
# ---------------------------------------------------------------------------


def test_feature_summary_returns_dataframe(xgb_explainer, feature_matrix):
    summary = xgb_explainer.feature_summary(feature_matrix[:50])
    assert isinstance(summary, pd.DataFrame)


def test_feature_summary_columns(xgb_explainer, feature_matrix):
    summary = xgb_explainer.feature_summary(feature_matrix[:50])
    for col in ("rank", "feature", "mean_abs_shap"):
        assert col in summary.columns


def test_feature_summary_row_count(xgb_explainer, feature_matrix):
    summary = xgb_explainer.feature_summary(feature_matrix[:50])
    assert len(summary) == len(FEATURE_COLS)


def test_feature_summary_sorted_descending(xgb_explainer, feature_matrix):
    summary = xgb_explainer.feature_summary(feature_matrix[:50])
    assert summary["mean_abs_shap"].is_monotonic_decreasing


def test_feature_summary_rank_starts_at_1(xgb_explainer, feature_matrix):
    summary = xgb_explainer.feature_summary(feature_matrix[:50])
    assert summary["rank"].iloc[0] == 1


def test_feature_summary_non_negative(xgb_explainer, feature_matrix):
    summary = xgb_explainer.feature_summary(feature_matrix[:50])
    assert (summary["mean_abs_shap"] >= 0).all()


def test_feature_summary_raises_before_fit(fitted_xgb, feature_matrix):
    explainer = FraudExplainer(fitted_xgb.model, FEATURE_COLS)
    with pytest.raises(RuntimeError, match="not been fitted"):
        explainer.feature_summary(feature_matrix[:5])


# ---------------------------------------------------------------------------
# explain_single() tests
# ---------------------------------------------------------------------------


def test_explain_single_returns_dict(xgb_explainer, feature_matrix):
    result = xgb_explainer.explain_single(feature_matrix, idx=0)
    assert isinstance(result, dict)


def test_explain_single_keys(xgb_explainer, feature_matrix):
    result = xgb_explainer.explain_single(feature_matrix, idx=0)
    assert "transaction_idx" in result
    assert "top_features" in result


def test_explain_single_top_features_count(xgb_explainer, feature_matrix):
    result = xgb_explainer.explain_single(feature_matrix, idx=0, top_n=3)
    assert len(result["top_features"]) == 3


def test_explain_single_feature_keys(xgb_explainer, feature_matrix):
    result = xgb_explainer.explain_single(feature_matrix, idx=0, top_n=2)
    for feat in result["top_features"]:
        for key in ("feature", "raw_value", "shap_value", "direction"):
            assert key in feat


def test_explain_single_feature_names_valid(xgb_explainer, feature_matrix):
    result = xgb_explainer.explain_single(feature_matrix, idx=0)
    for feat in result["top_features"]:
        assert feat["feature"] in FEATURE_COLS


def test_explain_single_raises_on_out_of_range(xgb_explainer, feature_matrix):
    with pytest.raises(IndexError, match="out of range"):
        xgb_explainer.explain_single(feature_matrix, idx=len(feature_matrix) + 10)


# ---------------------------------------------------------------------------
# build_explainer() convenience function tests
# ---------------------------------------------------------------------------


def test_build_explainer_returns_fitted_explainer(fitted_xgb):
    explainer = build_explainer(fitted_xgb.model, FEATURE_COLS)
    assert explainer.shap_explainer is not None


def test_build_explainer_correct_type(fitted_xgb):
    explainer = build_explainer(fitted_xgb.model, FEATURE_COLS)
    assert isinstance(explainer, FraudExplainer)


# ---------------------------------------------------------------------------
# top_reason_codes() convenience function tests
# ---------------------------------------------------------------------------


def test_top_reason_codes_returns_dataframe(fitted_xgb, feature_matrix):
    result = top_reason_codes(fitted_xgb.model, FEATURE_COLS, feature_matrix[:10])
    assert isinstance(result, pd.DataFrame)


def test_top_reason_codes_row_count(fitted_xgb, feature_matrix):
    result = top_reason_codes(fitted_xgb.model, FEATURE_COLS, feature_matrix[:10])
    assert len(result) == 10


def test_top_reason_codes_with_transaction_ids(fitted_xgb, feature_matrix):
    ids = list(range(10))
    result = top_reason_codes(
        fitted_xgb.model, FEATURE_COLS, feature_matrix[:10], transaction_ids=ids
    )
    assert list(result["transaction_id"]) == ids
