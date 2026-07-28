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
    keys = pd.Series(_cat_keys(cat))
    return keys.map(state["counts"]).fillna(0.0).to_numpy(dtype=float)


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
    keys = pd.Series(_cat_keys(cat))
    return (
        keys.map(state["mapping"]).fillna(state["fallback"]).to_numpy(dtype=float)
    )


def _fit_groupby_moments(num: np.ndarray, cat: np.ndarray) -> dict:
    """Per-group mean and std, for the deviation/z-score operators."""
    keys = _cat_keys(cat)
    df = pd.DataFrame({"k": keys, "v": num.astype(float)})
    grouped = df.groupby("k", sort=True)["v"]
    means, stds, sizes = grouped.mean(), grouped.std(), grouped.size()
    ok = {k for k in sizes.index if sizes[k] >= MIN_GROUP_SIZE}
    g_mean = float(df["v"].mean()) if len(df) else float("nan")
    g_std = float(df["v"].std()) if len(df) else float("nan")
    return {
        "mean": {k: float(v) for k, v in means.items() if k in ok and np.isfinite(v)},
        "std": {k: float(v) for k, v in stds.items() if k in ok and np.isfinite(v)},
        "g_mean": g_mean if np.isfinite(g_mean) else 0.0,
        "g_std": g_std if np.isfinite(g_std) and g_std > EPS else 1.0,
    }


def _apply_groupby_dev(state: dict, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """x minus its group mean: how unusual this row is within its group.

    The signal the OpenFE paper's theory section is built around -- a row's
    value matters relative to the group it belongs to, not just absolutely.
    """
    keys = pd.Series(_cat_keys(cat))
    means = keys.map(state["mean"]).fillna(state["g_mean"]).to_numpy(dtype=float)
    return num.astype(float) - means


def _apply_groupby_zscore(state: dict, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    keys = pd.Series(_cat_keys(cat))
    means = keys.map(state["mean"]).fillna(state["g_mean"]).to_numpy(dtype=float)
    stds = keys.map(state["std"]).fillna(state["g_std"]).to_numpy(dtype=float)
    stds = np.where(np.abs(stds) > EPS, stds, np.nan)
    return (num.astype(float) - means) / stds


def _fit_groupby_rank(num: np.ndarray, cat: np.ndarray) -> dict:
    """Sorted training values per group, so replay can rank by searchsorted.

    Stored as plain lists rather than arrays: operator state round-trips
    through JSON in ``FeatureCrafter.to_json``.
    """
    keys = _cat_keys(cat)
    v = num.astype(float)
    finite = np.isfinite(v)
    out: dict[str, list[float]] = {}
    df = pd.DataFrame({"k": keys[finite], "v": v[finite]})
    for k, sub in df.groupby("k", sort=True)["v"]:
        if len(sub) >= MIN_GROUP_SIZE:
            out[k] = np.sort(sub.to_numpy()).tolist()
    return {"sorted": out, "global": np.sort(v[finite]).tolist()}


def _apply_groupby_rank(state: dict, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Within-group quantile of x, in [0, 1]; NaN where the value is NaN."""
    keys = _cat_keys(cat)
    v = num.astype(float)
    out = np.full(v.shape, np.nan, dtype=float)
    tables = state["sorted"]
    g = np.asarray(state["global"], dtype=float)
    for k in pd.unique(keys):
        ref = tables.get(k)
        ref = g if ref is None else np.asarray(ref, dtype=float)
        if ref.size == 0:
            continue
        idx = np.flatnonzero(keys == k)
        vals = v[idx]
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        out[idx[ok]] = np.searchsorted(ref, vals[ok], side="right") / ref.size
    return out


def _fit_nunique(a: np.ndarray, b: np.ndarray) -> dict:
    """Distinct values of b within each level of a (GroupByThenNUnique)."""
    ka, kb = _cat_keys(a), _cat_keys(b)
    df = pd.DataFrame({"a": ka, "b": kb})
    counts = df.groupby("a", sort=True)["b"].nunique()
    return {"counts": {k: float(v) for k, v in counts.items()}, "fallback": 0.0}


def _apply_nunique(state: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    keys = pd.Series(_cat_keys(a))
    return keys.map(state["counts"]).fillna(state["fallback"]).to_numpy(dtype=float)


def _fit_cat_cross(a: np.ndarray, b: np.ndarray) -> dict:
    ka, kb = _cat_keys(a), _cat_keys(b)
    pairs = np.char.add(np.char.add(ka, "\x1f"), kb)
    uniq = np.unique(pairs)
    # The cardinality guard the README documents. It was declared but never
    # enforced: crossing two high-cardinality columns produced a near-unique
    # code per row, which is pure noise to a model and expensive to carry.
    if uniq.size > MAX_CROSS_CARDINALITY:
        return {"codes": {}, "over_cardinality": True}
    return {"codes": {p: i for i, p in enumerate(uniq.tolist())}}


def _apply_cat_cross(state: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if state.get("over_cardinality"):
        return np.full(len(a), np.nan, dtype=float)
    ka, kb = _cat_keys(a), _cat_keys(b)
    pairs = np.char.add(np.char.add(ka, "\x1f"), kb)
    return pd.Series(pairs).map(state["codes"]).fillna(-1.0).to_numpy(dtype=float)


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
        Operator("groupby_median", "groupby", "num", fit_state=_fit_groupby("median"), apply=_apply_groupby),
        # Group-relative operators: these change the partition structure a tree
        # can exploit, which is where tree-model gains come from.  A plain
        # monotone transform of one column cannot help a GBDT at all.
        Operator("groupby_dev", "groupby", "num", fit_state=_fit_groupby_moments, apply=_apply_groupby_dev),
        Operator("groupby_zscore", "groupby", "num", fit_state=_fit_groupby_moments, apply=_apply_groupby_zscore),
        Operator("groupby_rank", "groupby", "num", fit_state=_fit_groupby_rank, apply=_apply_groupby_rank),
        Operator("nunique", "cat_pair", "num", fit_state=_fit_nunique, apply=_apply_nunique),
        Operator("minimum", "binary_num", "num", fn=lambda a, b: np.minimum(a, b)),
        Operator("maximum", "binary_num", "num", fn=lambda a, b: np.maximum(a, b)),
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
