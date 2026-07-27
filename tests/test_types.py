import numpy as np
import pandas as pd

from featurecraft.types import expand_datetime, infer_task, infer_types


def test_infer_types_basic():
    X = pd.DataFrame(
        {
            "num": np.linspace(0, 1, 100),
            "cat_str": ["a", "b"] * 50,
            "cat_int": [1, 2, 3, 4] * 25,
            "big_int": np.arange(100),
            "flag": [True, False] * 50,
        }
    )
    t = infer_types(X)
    assert "num" in t.numeric
    assert "big_int" in t.numeric  # high-cardinality integer stays numeric
    assert "cat_str" in t.categorical
    assert "cat_int" in t.categorical  # low-cardinality integer -> categorical
    assert "flag" in t.categorical


def test_infer_types_datetime_and_nan():
    X = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=50, freq="D"),
            "num_nan": [np.nan, 1.5] * 25,
        }
    )
    t = infer_types(X)
    assert t.datetime == ("ts",)
    assert "num_nan" in t.numeric


def test_expand_datetime():
    X = pd.DataFrame({"ts": pd.date_range("2024-01-01 06:00", periods=10, freq="h")})
    out = expand_datetime(X, ("ts",))
    assert "ts" not in out.columns
    assert set(out.columns) == {"ts__year", "ts__month", "ts__dow", "ts__hour"}
    assert out["ts__hour"].iloc[0] == 6.0


def test_infer_task():
    assert infer_task(pd.Series(["a", "b", "a"] * 30)) == "classification"
    assert infer_task(pd.Series([0, 1] * 50)) == "classification"
    assert infer_task(pd.Series(np.random.default_rng(0).normal(size=200))) == "regression"
