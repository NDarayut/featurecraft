import numpy as np
import pandas as pd

from featurecraft.evolve import Deadline, EvolveConfig, evolve, fitness_one
from featurecraft.feature import FeatureTree
from featurecraft.operators import OPERATORS
from featurecraft.policy import OperatorBandit
from featurecraft.progress import ProgressLog
from featurecraft.types import infer_types

OPS = dict(OPERATORS)


def _planted(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.1, 10, n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    resid = x2 * x3 + rng.normal(scale=0.05, size=n)  # planted interaction
    return X, resid


def _run(X, resid, seed=0, **kw):
    types = infer_types(X)
    cfg = EvolveConfig(population_size=80, generations=12, **kw)
    policy = OperatorBandit(list(OPS))
    rng = np.random.default_rng(seed)
    return evolve(
        X, resid, types, OPS, policy, cfg, rng,
        ProgressLog(verbose=0), Deadline(None),
    )


def test_fitness_prefers_planted_signal():
    X, resid = _planted()
    good = FeatureTree(
        op="mul",
        children=[
            FeatureTree(column="x2", ctype="num"),
            FeatureTree(column="x3", ctype="num"),
        ],
    )
    noise = FeatureTree(op="log1p", children=[FeatureTree(column="x1", ctype="num")])
    assert fitness_one(good, X, resid, OPS, 0.01) > fitness_one(noise, X, resid, OPS, 0.01) + 0.3


def test_invalid_tree_gets_zero_fitness():
    X, resid = _planted()
    const = FeatureTree(
        op="sub",
        children=[
            FeatureTree(column="x1", ctype="num"),
            FeatureTree(column="x1", ctype="num"),
        ],
    )
    assert fitness_one(const, X, resid, OPS, 0.01) == 0.0


def test_evolve_finds_planted_interaction():
    X, resid = _planted()
    hof, history = _run(X, resid)
    formulas = [t.formula() for _, t in hof.best()[:5]]
    assert any("x2" in f and "x3" in f for f in formulas), formulas
    assert len(history) >= 1
    assert history[-1]["best"] > 0.5


def test_deterministic_same_seed():
    X, resid = _planted()
    hof1, _ = _run(X, resid, seed=7)
    hof2, _ = _run(X, resid, seed=7)
    f1 = [t.formula() for _, t in hof1.best()]
    f2 = [t.formula() for _, t in hof2.best()]
    assert f1 == f2


def test_early_stop_on_noise():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    resid = rng.normal(size=300)
    _, history = _run(X, resid, early_stop=3)
    assert len(history) < 12  # stopped before all generations


def test_deadline_stops_immediately():
    X, resid = _planted()
    types = infer_types(X)
    cfg = EvolveConfig(population_size=50, generations=50)
    rng = np.random.default_rng(0)
    hof, history = evolve(
        X, resid, types, OPS, OperatorBandit(list(OPS)), cfg, rng,
        ProgressLog(verbose=0), Deadline(0.0),
    )
    assert len(history) == 0  # no generation ran, but init/hof still valid
