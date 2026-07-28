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
from sklearn.model_selection import KFold, StratifiedKFold

from .evolve import fitness_one, rank_array

_GATEKEEPER_FOLDS = 5
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
    downstream: str = "mixed",
) -> list[tuple[FeatureTree, float]]:
    """Redundancy-prune the hall of fame and fit survivors on full train.

    `downstream` decides what "redundant" means, because it is not the same
    question for every model:

    - "tree": rank correlation.  A gradient-boosted tree is invariant to
      monotone transforms, so `log(x)` really does carry nothing beyond `x`
      and dropping it is right.
    - "linear" / "mixed": Pearson correlation on the raw values.  A monotone
      reshaping of a skewed column is one of the most valuable things you can
      hand a Ridge or kNN model -- it is most of what autofeat does, and in
      the harness's recorded results the linear panel is where AutoFE gains
      actually live (concrete-strength: 0.605 baseline -> 0.852).  Ranking
      would score `log(x)` as a perfect duplicate of `x` and delete it;
      Pearson sees ~0.87 and keeps it, while still dropping a pure rescaling
      like `2x + 1`, which correlates at exactly 1.0.
    """
    use_ranks = downstream == "tree"
    accepted: list[tuple[FeatureTree, float]] = []
    accepted_sigs: list[np.ndarray] = []
    base_sigs = []
    for col in types.numeric:
        v = pd.to_numeric(X_sub[col], errors="coerce").to_numpy(dtype=float)
        base_sigs.append(_signature(v, use_ranks))

    for fit_score, tree in hof_entries:
        if len(accepted) >= max_new_features:
            break
        try:
            vals = np.asarray(tree.fit_values(X_sub, ops), dtype=float)
        except Exception:
            continue
        sig = _signature(vals, use_ranks)
        if sig is None:
            continue
        if any(
            _abs_corr(sig, other) > redundancy_threshold
            for other in base_sigs + accepted_sigs
            if other is not None
        ):
            progress.note(f"  redundant, dropped: {tree.formula()}", level=2)
            continue
        try:
            tree.fit_values(X_train, ops)  # final replay state, full train
        except Exception:
            continue
        accepted.append((tree, fit_score))
        accepted_sigs.append(sig)
        progress.note(f"  selected: {tree.formula()}  (fitness {fit_score:.3f})", level=2)
    return accepted


def _signature(v: np.ndarray, use_ranks: bool) -> np.ndarray | None:
    """Centred vector used for the redundancy comparison."""
    mask = np.isfinite(v)
    if mask.mean() < 0.2 or mask.sum() < 10:
        return None
    filled = np.where(mask, v, np.nanmedian(v[mask]))
    if np.std(filled) < 1e-12:
        return None
    r = rank_array(filled) if use_ranks else filled
    return r - r.mean()


def _safe_ranks(v: np.ndarray) -> np.ndarray | None:
    return _signature(v, use_ranks=True)


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom < 1e-12:
        return 0.0
    return abs(float((a * b).sum() / denom))


_rank_abs_corr = _abs_corr


def attribute_importance(
    trees: list[tuple[FeatureTree, float]],
    X_train: pd.DataFrame,
    y: np.ndarray,
    task: str,
    types: ColumnTypes,
    ops: dict[str, Operator],
    max_new_features: int,
    seed: int,
    n_jobs: int,
    progress,
    downstream: str = "mixed",
) -> list[tuple[FeatureTree, float]]:
    """OpenFE Stage II (Algorithm 4): rank candidates by importance in context.

    Every candidate was scored against one frozen residual vector, so the
    survivors tend to explain the *same* slice of it.  Fitting the base
    features and all candidates together, then ranking by MDI gain, is what
    accounts for interactions between candidates and base features -- and for
    candidates duplicating each other's contribution.  One extra model fit.
    """
    if len(trees) <= max_new_features:
        return trees
    enc, _ = encode_for_model(X_train, types)
    cols: dict[str, np.ndarray] = {}
    keep: list[int] = []
    for i, (tree, _) in enumerate(trees):
        try:
            cols[f"__fc{i}"] = np.asarray(tree.values(X_train, ops), dtype=float)
            keep.append(i)
        except Exception:
            continue
    if not cols:
        return trees[:max_new_features]
    aug = pd.concat([enc, pd.DataFrame(cols, index=enc.index)], axis=1)
    try:
        model = make_lgbm(task, seed, n_jobs)
        model.fit(aug, y)
        gains = model.booster_.feature_importance(importance_type="gain")
    except Exception:
        progress.note("  attribution model failed; keeping fitness order", level=2)
        return trees[:max_new_features]
    gain_by_name = dict(zip(aug.columns, gains))
    if downstream == "tree":
        score_of = {i: float(gain_by_name.get(f"__fc{i}", 0.0)) for i in keep}
    else:
        # Gain alone would delete every monotone transform -- a tree scores
        # them at ~0 by construction -- which is exactly the feature class the
        # linear and kNN panels want.  Blend the tree's opinion with the
        # candidate's own fitness, each as a normalised rank.
        by_gain = sorted(keep, key=lambda i: float(gain_by_name.get(f"__fc{i}", 0.0)))
        by_fit = sorted(keep, key=lambda i: trees[i][1])
        r_gain = {i: r for r, i in enumerate(by_gain)}
        r_fit = {i: r for r, i in enumerate(by_fit)}
        score_of = {i: r_gain[i] + r_fit[i] for i in keep}
    ranked = sorted(keep, key=lambda i: -score_of[i])
    chosen = sorted(ranked[:max_new_features])
    for i in ranked[max_new_features:]:
        progress.note(f"  low attributed gain, dropped: {trees[i][0].formula()}", level=2)
    return [trees[i] for i in chosen]


def gatekeeper(
    X: pd.DataFrame,
    y: np.ndarray,
    task: str,
    trees: list[FeatureTree],
    types: ColumnTypes,
    ops: dict[str, Operator],
    seed: int,
    n_jobs: int,
    downstream: str = "mixed",
    return_details: bool = False,
):
    """Cross-validated with-vs-without delta.  Positive = engineering helped.

    K-fold rather than one 80/20 split: on a few hundred rows the noise in a
    single split is comfortably larger than the effect being measured, and
    this number decides whether the features ship at all.

    The delta is averaged over the model families named by `downstream`.
    Judging on a gradient-boosted tree alone is actively misleading when the
    features will also be fed to a linear or distance-based model: a tree is
    invariant to monotone transforms, so it reports ~0 gain for exactly the
    features that help Ridge and kNN most.  Gating on the tree alone threw
    those features away.
    """
    families = {
        "tree": ("tree",),
        "linear": ("linear",),
        "mixed": ("tree", "linear"),
    }[downstream]
    n = len(X)
    if n < 25 or not trees:
        return (None, "", 0.0) if return_details else (None, "")
    # each fold costs two LightGBM fits; on large data three is plenty to
    # beat the noise of a single split without paying for five
    n_splits = _GATEKEEPER_FOLDS if n <= 20_000 else 3
    if task == "classification":
        counts = np.bincount(np.asarray(y, dtype=int))
        counts = counts[counts > 0]
        if counts.size < 2:
            return (None, "", 0.0) if return_details else (None, "")
        n_splits = int(min(n_splits, counts.min()))
    n_splits = int(min(n_splits, n // 10))
    if n_splits < 2:
        return (None, "", 0.0) if return_details else (None, "")

    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if task == "classification"
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    metric = _metric_name(task, y)
    deltas: list[float] = []
    for train_idx, test_idx in splitter.split(X, y if task == "classification" else None):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if task == "classification" and (
            len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2
        ):
            continue
        # engineered values: state refitted on this fold's train only
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
        for family in families:
            try:
                base = _fit_score(
                    enc_tr, y_tr, enc_te, y_te, task, seed, n_jobs, family)
                aug = _fit_score(
                    aug_tr, y_tr, aug_te, y_te, task, seed, n_jobs, family)
            except Exception:
                continue
            deltas.append(aug - base)
    if not deltas:
        return (None, "", 0.0) if return_details else (None, "")
    mean = float(np.mean(deltas))
    # Fraction of folds that improved. A positive mean carried by one
    # lucky fold is not evidence the features help; requiring agreement
    # across folds is what separates a real gain from split noise.
    win_rate = float(np.mean([d > 0 for d in deltas]))
    return (mean, metric, win_rate) if return_details else (mean, metric)


def _metric_name(task: str, y: np.ndarray) -> str:
    if task == "regression":
        return "r2"
    return "auc" if len(np.unique(y)) == 2 else "accuracy"


def _make_linear(task: str):
    """A scaled linear model, matching how such a panel is normally run."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    est = Ridge() if task == "regression" else LogisticRegression(max_iter=1000)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def _fit_score(X_tr, y_tr, X_te, y_te, task, seed, n_jobs, family="tree") -> float:
    if family == "linear":
        X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
        X_te = X_te.replace([np.inf, -np.inf], np.nan)
        model = _make_linear(task)
    else:
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
    "attribute_importance",
    "gatekeeper",
    "encode_for_model",
    "make_lgbm",
    "fitness_one",
]
