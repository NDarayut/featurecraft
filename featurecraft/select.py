"""Final selection and the gatekeeper.

Selection: hall-of-fame features in fitness order, greedily dropping any
candidate whose |Spearman| with an original numeric column or an
already-accepted feature exceeds the redundancy threshold; survivors get
their replay state fitted on the full training data.

Gatekeeper: an internal 80/20 holdout comparing a LightGBM on the original
features against one on original+engineered.  The literature is clear that
feature generation does not always help (OpenFE: no gain on 19/68 datasets);
the delta is a first-class output and a warning is raised when it is <= 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evolve import fitness_one, rank_array
from .feature import FeatureTree
from .operators import Operator
from .types import ColumnTypes


def encode_for_model(
    X: pd.DataFrame,
    types: ColumnTypes,
    mapping: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Ordinal-encode categorical columns for the internal LightGBM models.

    Codes are learned on the frame passed with mapping=None (train) and
    replayed on later frames; unseen values -> -1, NaN -> -2.
    """
    out = {}
    fitted = mapping if mapping is not None else {}
    for col in X.columns:
        s = X[col]
        if col in types.categorical:
            keys = s.astype(object).where(~s.isna(), "__nan__").astype(str)
            if mapping is None:
                codes = {v: i for i, v in enumerate(sorted(keys.unique()))}
                fitted[col] = codes
            else:
                codes = mapping.get(col, {})
            enc = keys.map(codes).fillna(-1.0).astype(float)
            enc[s.isna().to_numpy()] = -2.0
            out[col] = enc.to_numpy()
        else:
            out[col] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(out, index=X.index), fitted


def make_lgbm(task: str, seed: int, n_jobs: int):
    import lightgbm as lgb

    params = dict(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=n_jobs,
        verbose=-1,
    )
    if task == "classification":
        return lgb.LGBMClassifier(**params)
    return lgb.LGBMRegressor(**params)


def select_features(
    hof_entries: list[tuple[float, FeatureTree]],
    X_train: pd.DataFrame,
    X_sub: pd.DataFrame,
    types: ColumnTypes,
    ops: dict[str, Operator],
    max_new_features: int,
    redundancy_threshold: float,
    progress,
) -> list[tuple[FeatureTree, float]]:
    """Redundancy-prune the hall of fame and fit survivors on full train."""
    accepted: list[tuple[FeatureTree, float]] = []
    accepted_ranks: list[np.ndarray] = []
    base_ranks = []
    for col in types.numeric:
        v = pd.to_numeric(X_sub[col], errors="coerce").to_numpy(dtype=float)
        base_ranks.append(_safe_ranks(v))

    for fit_score, tree in hof_entries:
        if len(accepted) >= max_new_features:
            break
        try:
            vals = np.asarray(tree.fit_values(X_sub, ops), dtype=float)
        except Exception:
            continue
        ranks = _safe_ranks(vals)
        if ranks is None:
            continue
        if any(
            _rank_abs_corr(ranks, other) > redundancy_threshold
            for other in base_ranks + accepted_ranks
            if other is not None
        ):
            progress.note(f"  redundant, dropped: {tree.formula()}", level=2)
            continue
        try:
            tree.fit_values(X_train, ops)  # final replay state, full train
        except Exception:
            continue
        accepted.append((tree, fit_score))
        accepted_ranks.append(ranks)
        progress.note(f"  selected: {tree.formula()}  (fitness {fit_score:.3f})", level=2)
    return accepted


def _safe_ranks(v: np.ndarray) -> np.ndarray | None:
    mask = np.isfinite(v)
    if mask.mean() < 0.2 or mask.sum() < 10:
        return None
    filled = np.where(mask, v, np.nanmedian(v[mask]))
    if np.std(filled) < 1e-12:
        return None
    r = rank_array(filled)
    return r - r.mean()


def _rank_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom < 1e-12:
        return 0.0
    return abs(float((a * b).sum() / denom))


def gatekeeper(
    X: pd.DataFrame,
    y: np.ndarray,
    task: str,
    trees: list[FeatureTree],
    types: ColumnTypes,
    ops: dict[str, Operator],
    seed: int,
    n_jobs: int,
) -> tuple[float | None, str]:
    """Holdout with-vs-without delta.  Positive = engineering helped."""
    n = len(X)
    if n < 25 or not trees:
        return None, ""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(int(n * 0.2), 5)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    if task == "classification" and (
        len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2
    ):
        return None, ""

    # engineered values: state fitted on the 80% only (leak-safe holdout)
    new_tr, new_te = {}, {}
    for i, tree in enumerate(trees):
        t = tree.copy()
        try:
            new_tr[f"fc_{i}"] = np.asarray(t.fit_values(X_tr, ops), dtype=float)
            new_te[f"fc_{i}"] = np.asarray(t.values(X_te, ops), dtype=float)
        except Exception:
            continue

    enc_tr, mapping = encode_for_model(X_tr, types)
    enc_te, _ = encode_for_model(X_te, types, mapping)
    aug_tr = pd.concat([enc_tr, pd.DataFrame(new_tr, index=enc_tr.index)], axis=1)
    aug_te = pd.concat([enc_te, pd.DataFrame(new_te, index=enc_te.index)], axis=1)

    base_score = _fit_score(enc_tr, y_tr, enc_te, y_te, task, seed, n_jobs)
    aug_score = _fit_score(aug_tr, y_tr, aug_te, y_te, task, seed, n_jobs)
    metric = _metric_name(task, y)
    return aug_score - base_score, metric


def _metric_name(task: str, y: np.ndarray) -> str:
    if task == "regression":
        return "r2"
    return "auc" if len(np.unique(y)) == 2 else "accuracy"


def _fit_score(X_tr, y_tr, X_te, y_te, task, seed, n_jobs) -> float:
    model = make_lgbm(task, seed, n_jobs)
    model.fit(X_tr, y_tr)
    if task == "regression":
        from sklearn.metrics import r2_score

        return float(r2_score(y_te, model.predict(X_te)))
    if len(np.unique(y_tr)) == 2:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y_te, model.predict(X_te)))


__all__ = [
    "select_features",
    "gatekeeper",
    "encode_for_model",
    "make_lgbm",
    "fitness_one",
]
