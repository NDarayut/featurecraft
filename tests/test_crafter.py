import io

import numpy as np
import pandas as pd

from featurecraft import FeatureCrafter

FAST = dict(population_size=60, generations=8, verbose=0)


def _regression_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.1, 10, n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    cat = rng.choice(list("abcd"), n)
    shift = pd.Series(cat).map({"a": 0, "b": 2, "c": -1, "d": 5}).to_numpy()
    y = np.log1p(x1) + x2 * x3 + 0.5 * shift + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat})
    return X, y


def _classification_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 * x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


def test_recovers_planted_interaction():
    X, y = _regression_data()
    fc = FeatureCrafter(random_state=0, **FAST)
    Xn = fc.fit_transform(X, y)
    assert Xn.shape[0] == len(X) and Xn.shape[1] > X.shape[1]
    formulas = " | ".join(fc.feature_formulas_.values())
    assert "x2" in formulas and "x3" in formulas
    assert fc.holdout_delta_ is not None and fc.holdout_delta_ > 0


def test_same_seed_deterministic():
    X, y = _regression_data()
    a = FeatureCrafter(random_state=3, **FAST).fit(X, y)
    b = FeatureCrafter(random_state=3, **FAST).fit(X, y)
    assert a.feature_formulas_ == b.feature_formulas_
    pd.testing.assert_frame_equal(a.transform(X), b.transform(X))


def test_transform_is_pure_replay():
    X, y = _regression_data()
    fc = FeatureCrafter(random_state=0, **FAST).fit(X, y)
    full = fc.transform(X)
    head = fc.transform(X.head(50))
    pd.testing.assert_frame_equal(full.head(50), head)  # row-independent replay


def test_classification_path():
    X, y = _classification_data()
    fc = FeatureCrafter(random_state=0, **FAST)
    Xn = fc.fit_transform(X, y)
    assert fc.task_ == "classification"
    assert Xn.shape[1] >= X.shape[1]


def test_string_labels_classification():
    X, y = _classification_data()
    labels = np.where(y == 1, "yes", "no")
    fc = FeatureCrafter(random_state=0, **FAST).fit(X, labels)
    assert fc.task_ == "classification"


def test_numpy_input_and_max_new_features():
    X, y = _regression_data()
    fc = FeatureCrafter(random_state=0, max_new_features=2, **FAST)
    Xn = fc.fit_transform(X.to_numpy(), y)
    assert Xn.shape[1] <= X.shape[1] + 2


def test_all_categorical_input():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {"c1": rng.choice(list("abc"), n), "c2": rng.choice(list("xy"), n)}
    )
    y = (X["c1"] == "a").astype(int).to_numpy() + rng.normal(scale=0.1, size=n)
    fc = FeatureCrafter(random_state=0, **FAST)
    Xn = fc.fit_transform(X, y)
    assert Xn.shape[1] >= X.shape[1]  # freq/cross features possible, no crash


def test_constant_column_and_tiny_n():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": np.ones(30), "b": rng.normal(size=30)})
    y = rng.normal(size=30)
    fc = FeatureCrafter(random_state=0, **FAST)
    Xn = fc.fit_transform(X, y)  # must not crash
    assert Xn.shape[0] == 30


def test_time_budget_zero_returns_fast():
    X, y = _regression_data()
    fc = FeatureCrafter(random_state=0, time_budget=0.01, **FAST)
    Xn = fc.fit_transform(X, y)
    assert Xn.shape[0] == len(X)  # valid output even with no budget


def test_operator_subset_respected():
    X, y = _regression_data()
    fc = FeatureCrafter(random_state=0, operators=["mul", "add"], **FAST).fit(X, y)
    for formula in fc.feature_formulas_.values():
        assert "log1p" not in formula and "by (" not in formula


def test_verbose_levels():
    X, y = _regression_data(n=200)
    silent = io.StringIO()
    FeatureCrafter(random_state=0, population_size=40, generations=3,
                   verbose=0, stream=silent).fit(X, y)
    assert silent.getvalue() == ""
    chatty = io.StringIO()
    FeatureCrafter(random_state=0, population_size=40, generations=3,
                   verbose=2, stream=chatty).fit(X, y)
    text = chatty.getvalue()
    assert "gen " in text and "[featurecraft]" in text


def test_json_roundtrip(tmp_path):
    X, y = _regression_data()
    fc = FeatureCrafter(random_state=0, **FAST).fit(X, y)
    path = tmp_path / "model.json"
    fc.to_json(path)
    fc2 = FeatureCrafter.from_json(path)
    pd.testing.assert_frame_equal(fc.transform(X), fc2.transform(X))
    assert fc2.feature_formulas_ == fc.feature_formulas_


def test_rl_policy_off():
    X, y = _regression_data()
    fc = FeatureCrafter(random_state=0, rl_policy=False, **FAST).fit(X, y)
    assert fc.operator_stats_ == {}


def test_datetime_column():
    rng = np.random.default_rng(0)
    n = 300
    ts = pd.date_range("2024-01-01", periods=n, freq="7h")
    x = rng.normal(size=n)
    y = x + (ts.hour > 12).astype(float) + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"ts": ts, "x": x})
    fc = FeatureCrafter(random_state=0, **FAST)
    Xn = fc.fit_transform(X, y)
    assert "ts" in Xn.columns  # original column kept in output
    fc.transform(X.head(10))


def test_errors():
    X, y = _regression_data(n=100)
    fc = FeatureCrafter(**FAST)
    try:
        fc.transform(X)
        raise AssertionError("expected RuntimeError before fit")
    except RuntimeError:
        pass
    try:
        fc.fit(X, y[:50])
        raise AssertionError("expected ValueError on length mismatch")
    except ValueError:
        pass
    fitted = FeatureCrafter(random_state=0, **FAST).fit(X, y)
    try:
        fitted.transform(X.drop(columns=["x1"]))
        raise AssertionError("expected ValueError on missing column")
    except ValueError:
        pass
