"""The fixed operator vocabulary.

Every operator is NaN-safe: invalid inputs (log of a negative, division by
~zero) produce NaN, never inf.  Stateful operators (freq, groupby_*,
cat_cross) split into fit_state (train only) and apply (replay anywhere);
their state is a function of X alone, never of y.

Operator kinds and signatures:
    unary_num   (num) -> num
    binary_num  (num, num) -> num
    unary_cat   (cat) -> num          [freq]
    groupby     (num, cat) -> num     [groupby_mean/std/min/max]
    cat_cross   (cat, cat) -> cat
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd

EPS = 1e-12
MIN_GROUP_SIZE = 5
MAX_CROSS_CARDINALITY = 1000
NAN_KEY = "__nan__"


@dataclasses.dataclass(frozen=True)
class Operator:
    name: str
    kind: str            # unary_num | binary_num | unary_cat | groupby | cat_cross
    out_type: str        # "num" | "cat"
    fn: Callable | None = None          # stateless compute
    fit_state: Callable | None = None   # stateful: fit on train inputs -> state
    apply: Callable | None = None       # stateful: (state, inputs) -> values

    @property
    def arity(self) -> int:
        return 1 if self.kind in ("unary_num", "unary_cat") else 2

    @property
    def stateful(self) -> bool:
        return self.fit_state is not None


def _log1p(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    mask = x > -1
    out[mask] = np.log1p(x[mask])
    return out


def _sqrt(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    mask = x >= 0
    out[mask] = np.sqrt(x[mask])
    return out


def _reciprocal(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.abs(x) > EPS
    out[mask] = 1.0 / x[mask]
    return out


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.abs(b) > EPS
    out[mask] = a[mask] / b[mask]
    return out


def _cat_keys(values: np.ndarray) -> np.ndarray:
    """Categorical values as string keys, NaN mapped to a sentinel level."""
    s = pd.Series(values)
    keys = s.astype(object).where(~s.isna(), NAN_KEY)
    return keys.astype(str).to_numpy()


def _fit_freq(cat: np.ndarray) -> dict:
    keys = _cat_keys(cat)
    uniq, counts = np.unique(keys, return_counts=True)
    return {"counts": dict(zip(uniq.tolist(), counts.tolist()))}


def _apply_freq(state: dict, cat: np.ndarray) -> np.ndarray:
    counts = state["counts"]
    keys = _cat_keys(cat)
    return np.array([counts.get(k, 0) for k in keys], dtype=float)


def _fit_groupby(agg: str):
    def fit(num: np.ndarray, cat: np.ndarray) -> dict:
        keys = _cat_keys(cat)
        df = pd.DataFrame({"k": keys, "v": num.astype(float)})
        grouped = df.groupby("k", sort=True)["v"]
        stats = grouped.agg(agg)
        sizes = grouped.size()
        fallback = float(getattr(df["v"], agg)()) if len(df) else float("nan")
        if not np.isfinite(fallback):
            fallback = float("nan")
        mapping = {
            k: float(v)
            for k, v in stats.items()
            if sizes[k] >= MIN_GROUP_SIZE and np.isfinite(v)
        }
        return {"mapping": mapping, "fallback": fallback}

    return fit


def _apply_groupby(state: dict, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    mapping, fallback = state["mapping"], state["fallback"]
    keys = _cat_keys(cat)
    return np.array([mapping.get(k, fallback) for k in keys], dtype=float)


def _fit_cat_cross(a: np.ndarray, b: np.ndarray) -> dict:
    ka, kb = _cat_keys(a), _cat_keys(b)
    pairs = np.char.add(np.char.add(ka, "\x1f"), kb)
    uniq = np.unique(pairs)
    return {"codes": {p: i for i, p in enumerate(uniq.tolist())}}


def _apply_cat_cross(state: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    codes = state["codes"]
    ka, kb = _cat_keys(a), _cat_keys(b)
    pairs = np.char.add(np.char.add(ka, "\x1f"), kb)
    return np.array([codes.get(p, -1) for p in pairs], dtype=float)


OPERATORS: dict[str, Operator] = {
    op.name: op
    for op in [
        Operator("log1p", "unary_num", "num", fn=_log1p),
        Operator("sqrt", "unary_num", "num", fn=_sqrt),
        Operator("square", "unary_num", "num", fn=lambda x: np.asarray(x, dtype=float) ** 2),
        Operator("reciprocal", "unary_num", "num", fn=_reciprocal),
        Operator("abs", "unary_num", "num", fn=lambda x: np.abs(np.asarray(x, dtype=float))),
        Operator("add", "binary_num", "num", fn=lambda a, b: a + b),
        Operator("sub", "binary_num", "num", fn=lambda a, b: a - b),
        Operator("mul", "binary_num", "num", fn=lambda a, b: a * b),
        Operator("div", "binary_num", "num", fn=_div),
        Operator("freq", "unary_cat", "num", fit_state=_fit_freq, apply=_apply_freq),
        Operator("groupby_mean", "groupby", "num", fit_state=_fit_groupby("mean"), apply=_apply_groupby),
        Operator("groupby_std", "groupby", "num", fit_state=_fit_groupby("std"), apply=_apply_groupby),
        Operator("groupby_min", "groupby", "num", fit_state=_fit_groupby("min"), apply=_apply_groupby),
        Operator("groupby_max", "groupby", "num", fit_state=_fit_groupby("max"), apply=_apply_groupby),
        Operator("cat_cross", "cat_cross", "cat", fit_state=_fit_cat_cross, apply=_apply_cat_cross),
    ]
}


def select_operators(names: list[str] | tuple[str, ...] | None) -> dict[str, Operator]:
    """Return the vocabulary, restricted to `names` when given."""
    if names is None:
        return dict(OPERATORS)
    unknown = [n for n in names if n not in OPERATORS]
    if unknown:
        raise ValueError(f"unknown operators: {unknown}; available: {sorted(OPERATORS)}")
    return {n: OPERATORS[n] for n in names}
