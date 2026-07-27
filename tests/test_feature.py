import numpy as np
import pandas as pd

from featurecraft.feature import (
    FeatureTree,
    crossover,
    depth1_candidates,
    mutate,
    random_tree,
)
from featurecraft.operators import OPERATORS
from featurecraft.policy import UniformPolicy
from featurecraft.types import infer_types

OPS = dict(OPERATORS)


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.uniform(0.5, 5, n),
            "b": rng.normal(size=n),
            "c": rng.choice(list("xyz"), n),
            "d": rng.choice(list("mn"), n),
        }
    )


def _num(c):
    return FeatureTree(column=c, ctype="num")


def _cat(c):
    return FeatureTree(column=c, ctype="cat")


def test_fit_then_replay_identical_all_ops():
    X = _data()
    trees = [
        FeatureTree(op="log1p", children=[_num("a")]),
        FeatureTree(op="div", children=[_num("a"), _num("b")]),
        FeatureTree(op="freq", children=[_cat("c")]),
        FeatureTree(op="groupby_mean", children=[_num("a"), _cat("c")]),
        FeatureTree(op="cat_cross", children=[_cat("c"), _cat("d")]),
        FeatureTree(
            op="mul",
            children=[
                FeatureTree(op="groupby_std", children=[_num("b"), _cat("d")]),
                _num("a"),
            ],
        ),
    ]
    for t in trees:
        fitted = t.fit_values(X, OPS)
        replayed = t.values(X, OPS)
        np.testing.assert_array_equal(
            np.asarray(fitted, dtype=float), np.asarray(replayed, dtype=float)
        )


def test_replay_uses_train_state_on_new_data():
    X = _data()
    t = FeatureTree(op="groupby_mean", children=[_num("a"), _cat("c")])
    t.fit_values(X, OPS)
    X2 = pd.DataFrame({"a": [1.0], "b": [0.0], "c": ["unseen"], "d": ["m"]})
    out = t.values(X2, OPS)
    assert out[0] == t.state["fallback"]


def test_unfitted_stateful_raises():
    t = FeatureTree(op="freq", children=[_cat("c")])
    try:
        t.values(_data(), OPS)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_formula_and_roundtrip():
    X = _data()
    t = FeatureTree(
        op="div",
        children=[
            FeatureTree(op="groupby_mean", children=[_num("a"), _cat("c")]),
            FeatureTree(op="log1p", children=[_num("b")]),
        ],
    )
    assert t.formula() == "(mean(a) by (c) / log1p(b))"
    t.fit_values(X, OPS)
    t2 = FeatureTree.from_dict(t.to_dict())
    np.testing.assert_array_equal(t.values(X, OPS), t2.values(X, OPS))
    assert t2.formula() == t.formula()


def test_random_tree_typed_and_depth():
    X = _data()
    types = infer_types(X)
    rng = np.random.default_rng(0)
    policy = UniformPolicy()
    for _ in range(200):
        t = random_tree(rng, types, OPS, policy, max_depth=3)
        assert t is not None
        assert t.depth() <= 3
        assert t.out_type(OPS) == "num"
        t.fit_values(X, OPS)  # must compute without raising


def test_crossover_and_mutation_produce_valid_trees():
    X = _data()
    types = infer_types(X)
    rng = np.random.default_rng(1)
    policy = UniformPolicy()
    trees = [
        t
        for _ in range(30)
        if (t := random_tree(rng, types, OPS, policy, max_depth=3)) is not None
        and not t.is_leaf
    ]
    for i in range(len(trees) - 1):
        child = crossover(trees[i], trees[i + 1], rng, OPS, max_depth=3)
        if child is not None:
            assert child.depth() <= 3
            child.fit_values(X, OPS)
        mutated, _ = mutate(trees[i], rng, types, OPS, policy, max_depth=3)
        if mutated is not None:
            assert mutated.depth() <= 3
            mutated.fit_values(X, OPS)


def test_depth1_candidates_deterministic():
    types = infer_types(_data())
    a = [t.formula() for t in depth1_candidates(types, OPS, limit=100)]
    b = [t.formula() for t in depth1_candidates(types, OPS, limit=100)]
    assert a == b and len(a) > 10
