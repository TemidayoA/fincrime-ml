"""
tests/test_fraud/test_imbalance.py
====================================
Unit tests for the class imbalance handler.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fincrime_ml.fraud.imbalance import ImbalanceHandler, ImbalanceResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def imbalanced_data():
    """Synthetic imbalanced dataset: 5% fraud rate, 2 features."""
    rng = np.random.default_rng(42)
    n = 500
    n_fraud = 25  # 5% fraud rate
    # Legitimate: centred at (0, 0)
    X_legit = rng.normal(loc=0.0, scale=1.0, size=(n - n_fraud, 2))
    y_legit = np.zeros(n - n_fraud, dtype=int)
    # Fraud: centred at (3, 3) to be separable
    X_fraud = rng.normal(loc=3.0, scale=0.5, size=(n_fraud, 2))
    y_fraud = np.ones(n_fraud, dtype=int)
    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([y_legit, y_fraud])
    return X, y


@pytest.fixture(scope="module")
def handler():
    return ImbalanceHandler(seed=42, smote_strategy=0.5, cv_folds=3)


# ---------------------------------------------------------------------------
# apply_smote tests
# ---------------------------------------------------------------------------


def test_smote_increases_fraud_count(handler, imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = handler.apply_smote(X, y)
    assert y_res.sum() > y.sum()


def test_smote_preserves_legitimate_count(handler, imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = handler.apply_smote(X, y)
    assert (y_res == 0).sum() == (y == 0).sum()


def test_smote_output_shapes_consistent(handler, imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = handler.apply_smote(X, y)
    assert X_res.shape[0] == len(y_res)
    assert X_res.shape[1] == X.shape[1]


def test_smote_returns_numpy_arrays(handler, imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = handler.apply_smote(X, y)
    assert isinstance(X_res, np.ndarray)
    assert isinstance(y_res, np.ndarray)


def test_smote_fraud_rate_increases(handler, imbalanced_data):
    X, y = imbalanced_data
    original_rate = y.mean()
    X_res, y_res = handler.apply_smote(X, y)
    resampled_rate = y_res.mean()
    assert resampled_rate > original_rate


def test_smote_accepts_pandas_inputs(handler, imbalanced_data):
    import pandas as pd

    X, y = imbalanced_data
    X_df = pd.DataFrame(X, columns=["f1", "f2"])
    y_series = pd.Series(y)
    X_res, y_res = handler.apply_smote(X_df, y_series)
    assert isinstance(X_res, np.ndarray)


def test_smote_raises_on_insufficient_fraud_samples():
    handler = ImbalanceHandler(seed=42)
    X = np.random.default_rng(0).normal(size=(100, 2))
    y = np.zeros(100, dtype=int)
    y[0] = 1  # Only 1 fraud sample; k_neighbors=5 requires at least 6
    with pytest.raises(ValueError, match="k_neighbors"):
        handler.apply_smote(X, y)


def test_smote_raises_on_shape_mismatch(handler):
    X = np.ones((10, 2))
    y = np.ones(5)
    with pytest.raises(ValueError, match="same length"):
        handler.apply_smote(X, y)


def test_smote_reproducible_with_same_seed(imbalanced_data):
    X, y = imbalanced_data
    h1 = ImbalanceHandler(seed=7, smote_strategy=0.5)
    h2 = ImbalanceHandler(seed=7, smote_strategy=0.5)
    X1, y1 = h1.apply_smote(X, y)
    X2, y2 = h2.apply_smote(X, y)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_smote_different_seeds_differ(imbalanced_data):
    X, y = imbalanced_data
    h1 = ImbalanceHandler(seed=1, smote_strategy=0.5)
    h2 = ImbalanceHandler(seed=2, smote_strategy=0.5)
    X1, _ = h1.apply_smote(X, y)
    X2, _ = h2.apply_smote(X, y)
    assert not np.array_equal(X1, X2)


# ---------------------------------------------------------------------------
# compute_sample_weights tests
# ---------------------------------------------------------------------------


def test_sample_weights_shape(handler, imbalanced_data):
    _, y = imbalanced_data
    weights = handler.compute_sample_weights(y)
    assert weights.shape == y.shape


def test_sample_weights_all_positive(handler, imbalanced_data):
    _, y = imbalanced_data
    weights = handler.compute_sample_weights(y)
    assert (weights > 0).all()


def test_sample_weights_fraud_higher_than_legit(handler, imbalanced_data):
    _, y = imbalanced_data
    weights = handler.compute_sample_weights(y)
    fraud_weight = weights[y == 1].mean()
    legit_weight = weights[y == 0].mean()
    assert fraud_weight > legit_weight


def test_sample_weights_returns_numpy_array(handler, imbalanced_data):
    _, y = imbalanced_data
    weights = handler.compute_sample_weights(y)
    assert isinstance(weights, np.ndarray)


def test_sample_weights_uniform_for_equal_classes(handler):
    """Balanced classes should produce equal weights."""
    y = np.array([0] * 50 + [1] * 50)
    weights = handler.compute_sample_weights(y)
    np.testing.assert_allclose(weights[y == 0], weights[y == 1])


def test_sample_weights_raises_on_empty_y(handler):
    with pytest.raises(ValueError, match="must not be empty"):
        handler.compute_sample_weights(np.array([]))


def test_sample_weights_raises_on_single_class(handler):
    with pytest.raises(ValueError, match="at least 2 classes"):
        handler.compute_sample_weights(np.zeros(10, dtype=int))


def test_sample_weights_accepts_pandas_series(handler, imbalanced_data):
    import pandas as pd

    _, y = imbalanced_data
    weights = handler.compute_sample_weights(pd.Series(y))
    assert isinstance(weights, np.ndarray)


# ---------------------------------------------------------------------------
# benchmark / ImbalanceResult tests
# ---------------------------------------------------------------------------


def test_benchmark_returns_two_results(handler, imbalanced_data):
    X, y = imbalanced_data
    results = handler.benchmark(X, y)
    assert len(results) == 2


def test_benchmark_result_strategies(handler, imbalanced_data):
    X, y = imbalanced_data
    results = handler.benchmark(X, y)
    strategies = {r.strategy for r in results}
    assert strategies == {"smote", "cost_sensitive"}


def test_benchmark_auc_pr_in_valid_range(handler, imbalanced_data):
    X, y = imbalanced_data
    results = handler.benchmark(X, y)
    for r in results:
        assert 0.0 <= r.mean_auc_pr <= 1.0, f"{r.strategy} AUC-PR out of range"


def test_benchmark_roc_auc_in_valid_range(handler, imbalanced_data):
    X, y = imbalanced_data
    results = handler.benchmark(X, y)
    for r in results:
        assert 0.0 <= r.mean_roc_auc <= 1.0, f"{r.strategy} ROC-AUC out of range"


def test_benchmark_std_non_negative(handler, imbalanced_data):
    X, y = imbalanced_data
    results = handler.benchmark(X, y)
    for r in results:
        assert r.std_auc_pr >= 0
        assert r.std_roc_auc >= 0


def test_benchmark_fraud_rate_correct(handler, imbalanced_data):
    X, y = imbalanced_data
    results = handler.benchmark(X, y)
    expected_rate = float(y.mean())
    for r in results:
        assert r.fraud_rate == pytest.approx(expected_rate)


def test_benchmark_n_folds_correct(handler, imbalanced_data):
    X, y = imbalanced_data
    results = handler.benchmark(X, y)
    for r in results:
        assert r.n_folds == handler.cv_folds


def test_benchmark_custom_estimator(handler, imbalanced_data):
    X, y = imbalanced_data
    est = LogisticRegression(max_iter=200, random_state=0)
    results = handler.benchmark(X, y, estimator=est)
    assert len(results) == 2


def test_benchmark_separable_data_high_auc_pr(imbalanced_data):
    """Well-separated classes should produce high AUC-PR for both strategies."""
    X, y = imbalanced_data
    handler = ImbalanceHandler(seed=42, smote_strategy=0.5, cv_folds=3)
    results = handler.benchmark(X, y)
    for r in results:
        assert r.mean_auc_pr > 0.5, f"{r.strategy} AUC-PR surprisingly low on separable data"


# ---------------------------------------------------------------------------
# ImbalanceResult tests
# ---------------------------------------------------------------------------


def test_imbalance_result_summary_contains_strategy():
    r = ImbalanceResult(
        strategy="smote",
        mean_auc_pr=0.72,
        std_auc_pr=0.05,
        mean_roc_auc=0.88,
        std_roc_auc=0.03,
        n_folds=5,
        fraud_rate=0.03,
    )
    assert "smote" in r.summary()


def test_imbalance_result_summary_contains_auc_pr():
    r = ImbalanceResult(
        strategy="cost_sensitive",
        mean_auc_pr=0.68,
        std_auc_pr=0.04,
        mean_roc_auc=0.85,
        std_roc_auc=0.02,
        n_folds=5,
        fraud_rate=0.03,
    )
    assert "AUC-PR" in r.summary()


# ---------------------------------------------------------------------------
# best_strategy tests
# ---------------------------------------------------------------------------


def test_best_strategy_returns_valid_name(handler, imbalanced_data):
    X, y = imbalanced_data
    best = handler.best_strategy(X, y)
    assert best in {"smote", "cost_sensitive"}
