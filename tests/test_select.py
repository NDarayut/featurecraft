import numpy as np
import pandas as pd

from featurecraft.feature import FeatureTree
from featurecraft.operators import OPERATORS
from featurecraft.progress import ProgressLog
from featurecraft.select import gatekeeper, select_features
from featurecraft.types import infer_types

OPS = dict(OPERATORS)


def _num(c):
    return FeatureTree(column=c, ctype="num")


def test_redundant_candidate_dropped():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    types = infer_types(X)
    # abs of the identity-like square-root chain: candidate == column a rank-wise
    duplicate = FeatureTree(op="add", children=[_num("a"), _num("a")])  # 2a ~ a
    useful = FeatureTree(op="mul", children=[_num("a"), _num("b")])
    entries = [(0.9, duplicate), (0.8, useful)]
    selected = select_features(
        entries, X, X, types, OPS, max_new_features=10,
        redundancy_threshold=0.98, progress=ProgressLog(verbose=0),
    )
    formulas = [t.formula() for t, _ in selected]
    assert "(a + a)" not in formulas
    assert "(a * b)" in formulas


def test_max_new_features_cap():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({f"c{i}": rng.normal(size=150) for i in range(6)})
    types = infer_types(X)
    entries = []
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            entries.append(
                (0.5, FeatureTree(op="mul", children=[_num(cols[i]), _num(cols[j])]))
            )
    selected = select_features(
        entries, X, X, types, OPS, max_new_features=3,
        redundancy_threshold=0.98, progress=ProgressLog(verbose=0),
    )
    assert len(selected) == 3


def test_gatekeeper_positive_on_real_signal():
    rng = np.random.default_rng(0)
    n = 500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = x1 * x2 + rng.normal(scale=0.05, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    types = infer_types(X)
    tree = FeatureTree(op="mul", children=[_num("x1"), _num("x2")])
    tree.fit_values(X, OPS)
    delta, metric = gatekeeper(X, y, "regression", [tree], types, OPS, seed=0, n_jobs=1)
    assert metric == "r2"
    assert delta is not None and delta > 0.05


def test_gatekeeper_none_when_no_trees_or_tiny():
    X = pd.DataFrame({"a": np.arange(30, dtype=float)})
    types = infer_types(X)
    delta, _ = gatekeeper(
        X, np.arange(30, dtype=float), "regression", [], types, OPS, seed=0, n_jobs=1
    )
    assert delta is None
