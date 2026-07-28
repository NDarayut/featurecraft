"""Column typing and task inference.

Columns are classified once at fit time into numeric, categorical, and
datetime.  Datetime columns are expanded into numeric component columns
(year, month, day-of-week, hour) so the rest of the engine only ever sees
numeric and categorical inputs.  NaNs are left in place everywhere: numeric
operators propagate them, and categorical state treats NaN as its own level.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

_DT_COMPONENTS = ("year", "month", "dow", "hour")


@dataclasses.dataclass(frozen=True)
class ColumnTypes:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    datetime: tuple[str, ...]


def max_categorical_cardinality(n_rows: int) -> int:
    return max(30, int(np.sqrt(max(n_rows, 1))))


def infer_task(y: pd.Series | np.ndarray) -> str:
    """Return "classification" or "regression" from the target alone."""
    y = pd.Series(np.asarray(y).ravel())
    if y.dtype.kind in "OUSb" or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    nunique = y.nunique(dropna=True)
    if y.dtype.kind in "iu" and nunique <= min(20, max(2, len(y) // 20)):
        return "classification"
    if nunique <= 2:
        return "classification"
    return "regression"


def infer_types(X: pd.DataFrame) -> ColumnTypes:
    """Classify every column of X as numeric, categorical, or datetime."""
    numeric: list[str] = []
    categorical: list[str] = []
    datetime: list[str] = []
    cat_cap = max_categorical_cardinality(len(X))
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime.append(col)
        elif (
            isinstance(s.dtype, pd.CategoricalDtype)
            or s.dtype.kind in "OUSb"
            or pd.api.types.is_bool_dtype(s)
        ):
            categorical.append(col)
        elif s.dtype.kind in "iu" or _is_integer_valued(s):
            nunique = s.nunique(dropna=True)
            if 2 <= nunique <= cat_cap:
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            numeric.append(col)
    return ColumnTypes(tuple(numeric), tuple(categorical), tuple(datetime))


def override_categorical(
    types: ColumnTypes, categorical: list[str] | tuple[str, ...], columns: list[str]
) -> ColumnTypes:
    """Re-type columns from an explicit categorical list.

    Callers that already know which columns are categorical (a benchmark
    harness that ordinal-encoded them, say) can say so.  Inference on a
    pre-encoded frame is unreliable in both directions: a high-cardinality
    categorical encoded to integer codes looks numeric, and a low-cardinality
    genuine count looks categorical.
    """
    unknown = [c for c in categorical if c not in columns]
    if unknown:
        raise ValueError(f"unknown categorical_features: {unknown}")
    cat = set(categorical)
    dt = set(types.datetime)
    return ColumnTypes(
        tuple(c for c in columns if c not in cat and c not in dt),
        tuple(c for c in columns if c in cat),
        types.datetime,
    )


def _is_integer_valued(s: pd.Series) -> bool:
    if s.dtype.kind != "f":
        return False
    vals = s.dropna().to_numpy()
    if vals.size == 0:
        return False
    return bool(np.all(vals == np.round(vals)))


def expand_datetime(X: pd.DataFrame, dt_cols: tuple[str, ...]) -> pd.DataFrame:
    """Replace datetime columns with numeric component columns.

    Pure per-row function of the timestamp, so applying it independently at
    fit and transform time is leak-safe by construction.
    """
    if not dt_cols:
        return X
    X = X.copy()
    for col in dt_cols:
        dt = pd.to_datetime(X[col])
        X[f"{col}__year"] = dt.dt.year.astype(float)
        X[f"{col}__month"] = dt.dt.month.astype(float)
        X[f"{col}__dow"] = dt.dt.dayofweek.astype(float)
        X[f"{col}__hour"] = dt.dt.hour.astype(float)
        X = X.drop(columns=[col])
    return X


def datetime_component_columns(dt_cols: tuple[str, ...]) -> tuple[str, ...]:
    """Names `expand_datetime` will produce for `dt_cols`."""
    return tuple(f"{c}__{p}" for c in dt_cols for p in _DT_COMPONENTS)
