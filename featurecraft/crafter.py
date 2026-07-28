"""FeatureCrafter: the public API.

One class orchestrating the whole algorithm:

1. type the columns, expand datetimes
2. fit one base LightGBM on the original features; compute residuals
3. evolve feature formula trees against the residuals (GA + UCB1 policy)
4. redundancy-prune the hall of fame, fit survivors on full train
5. gatekeeper: holdout with-vs-without delta, warn when engineering
   did not help
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .evolve import Deadline, EvolveConfig, ResidualTarget, evolve
from .feature import FeatureTree
from .operators import select_operators
from .policy import OperatorBandit, UniformPolicy
from .progress import ProgressLog
from .report import build_report
from .select import (
    attribute_importance,
    encode_for_model,
    gatekeeper,
    make_lgbm,
    select_features,
)
from .types import (
    ColumnTypes,
    expand_datetime,
    infer_task,
    infer_types,
    override_categorical,
)

_SUBSAMPLE_ROWS = 2000
_RESIDUAL_FOLDS = 5
# how many candidates to shortlist per final slot, before Stage II ranks them
_SHORTLIST_FACTOR = 3
_DOWNSTREAM = ("mixed", "tree", "linear")
# fraction of gatekeeper folds that must improve before features ship
_GATE_MIN_WIN_RATE = 0.6


def _make_linear_base():
    """Standardised ridge, used only to produce the linear-panel residual."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge()
    )

# LightGBM fatally rejects these in feature names (CheckAllowedJSON), and
# formulas like "cross(a, b)" contain one.  Formulas stay readable in
# feature_formulas_; only the emitted column name is sanitized.
_UNSAFE_NAME_CHARS = str.maketrans({c: ";" for c in '",:[]{}'})


def _safe_name(formula: str) -> str:
    return formula.translate(_UNSAFE_NAME_CHARS)


class FeatureCrafter:
    """Automatic feature engineering via RL-guided genetic search."""

    name = "featurecraft"

    def __init__(
        self,
        task: str | None = None,
        operators: list[str] | tuple[str, ...] | None = None,
        categorical_features: list[str] | tuple[str, ...] | None = None,
        population_size: int = 200,
        generations: int = 25,
        max_depth: int = 3,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,
        tournament_k: int = 3,
        elitism: int = 10,
        parsimony: float = 0.002,
        rl_policy: bool = True,
        ucb_c: float = 1.4,
        max_new_features: int | None = None,
        redundancy_threshold: float = 0.98,
        downstream: str = "mixed",
        gate: bool = True,
        n_jobs: int = 1,
        time_budget: float | None = None,
        random_state: int = 0,
        verbose: int = 1,
        stream=None,
    ):
        self.task = task
        self.operators = tuple(operators) if operators is not None else None
        self.categorical_features = (
            tuple(categorical_features) if categorical_features is not None else None
        )
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.parsimony = parsimony
        self.rl_policy = rl_policy
        self.ucb_c = ucb_c
        self.max_new_features = max_new_features
        self.redundancy_threshold = redundancy_threshold
        if downstream not in _DOWNSTREAM:
            raise ValueError(
                f"downstream must be one of {_DOWNSTREAM}, got {downstream!r}")
        self.downstream = downstream
        self.gate = gate
        self.n_jobs = n_jobs
        self.time_budget = time_budget
        self.random_state = random_state
        self.verbose = verbose
        self.stream = stream
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, task: str | None = None) -> "FeatureCrafter":
        t_start = time.monotonic()
        X = self._as_frame(X)
        y = np.asarray(pd.Series(y).to_numpy()).ravel()
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if pd.isna(pd.Series(y)).any():
            raise ValueError("y contains missing values")

        self.task_ = task or self.task or infer_task(y)
        if self.task_ not in ("classification", "regression"):
            raise ValueError(f"unknown task: {self.task_!r}")
        self.columns_in_ = list(X.columns)

        raw_types = infer_types(X)
        self._dt_cols = raw_types.datetime
        X_work = expand_datetime(X, self._dt_cols)
        self.types_ = infer_types(X_work)
        if self.categorical_features is not None:
            self.types_ = override_categorical(
                self.types_, self.categorical_features, list(X_work.columns)
            )
        ops = select_operators(list(self.operators) if self.operators else None)
        self._ops = ops

        progress = ProgressLog(self.verbose, self.stream)
        progress.start(len(X), X.shape[1], self.task_)
        rng = np.random.default_rng(self.random_state)
        deadline = Deadline(
            self.time_budget * 0.8 if self.time_budget is not None else None
        )
        max_new = self.max_new_features
        if max_new is None:
            max_new = min(2 * max(X_work.shape[1], 1), 50)

        self.selected_ = []
        self.feature_names_ = []
        self.feature_formulas_ = {}
        self.operator_stats_ = {}
        self.holdout_delta_ = None
        self.holdout_metric_ = ""
        history: list[dict] = []

        usable = bool(self.types_.numeric) or bool(self.types_.categorical)
        if usable and len(X_work) >= 10:
            resid, y_codes = self._residuals(X_work, y, progress)
            target = ResidualTarget(resid, seed=self.random_state)
            sub_idx = self._subsample_index(rng, len(X_work))
            X_sub = X_work.iloc[sub_idx].reset_index(drop=True)
            resid_sub = target.subset(sub_idx)

            policy = (
                OperatorBandit(list(ops), self.ucb_c)
                if self.rl_policy
                else UniformPolicy()
            )
            cfg = EvolveConfig(
                population_size=self.population_size,
                generations=self.generations,
                max_depth=self.max_depth,
                crossover_rate=self.crossover_rate,
                mutation_rate=self.mutation_rate,
                tournament_k=self.tournament_k,
                elitism=self.elitism,
                parsimony=self.parsimony,
                hof_capacity=max(3 * max_new, 30),
                n_jobs=self.n_jobs,
            )
            hof, history = evolve(
                X_sub, resid_sub, self.types_, ops, policy, cfg, rng, progress,
                deadline, X_full=X_work, target_full=target,
            )
            self.operator_stats_ = policy.stats()

            hof_entries = hof.best()
            # Over-select, then let Stage II cut to the budget: importance in
            # the presence of the base features is a better ranking than the
            # static fitness the candidates were generated against.
            shortlist = select_features(
                hof_entries,
                X_work,
                X_sub,
                self.types_,
                ops,
                max_new * _SHORTLIST_FACTOR,
                self.redundancy_threshold,
                progress,
                downstream=self.downstream,
            )
            y_model = y_codes if self.task_ == "classification" else y.astype(float)
            self.selected_ = attribute_importance(
                shortlist,
                X_work,
                y_model,
                self.task_,
                self.types_,
                ops,
                max_new,
                self.random_state,
                self.n_jobs,
                progress,
                self.downstream,
            )
            self._near_misses = [
                (f, t.formula())
                for f, t in hof_entries
                if t.formula() not in {tr.formula() for tr, _ in self.selected_}
            ][:10]

            self.feature_names_ = self._make_names(
                [t for t, _ in self.selected_], set(self.columns_in_)
            )
            self.feature_formulas_ = {
                name: tree.formula()
                for name, (tree, _) in zip(self.feature_names_, self.selected_)
            }

            self.holdout_delta_, self.holdout_metric_, win_rate = gatekeeper(
                X_work,
                y_model,
                self.task_,
                [t for t, _ in self.selected_],
                self.types_,
                ops,
                self.random_state,
                self.n_jobs,
                self.downstream,
                return_details=True,
            )
            # Act on the verdict rather than only reporting it.  Feature
            # generation genuinely does not always help (OpenFE gained
            # nothing on 19 of 68 datasets); shipping features that measurably
            # hurt is strictly worse than shipping none, and the benchmark
            # mean is dominated by the datasets where a method degrades.
            # A positive mean carried by one lucky fold is not evidence, so
            # the folds must also agree.  Measured on the benchmark panel, a
            # mean-only gate let through the two datasets where featurecraft
            # scored below the raw baseline.
            passed = (
                self.holdout_delta_ is not None
                and self.holdout_delta_ > 0
                and win_rate >= _GATE_MIN_WIN_RATE
            )
            if self.gate and self.holdout_delta_ is not None and not passed:
                progress.note(
                    f"gatekeeper: delta {self.holdout_delta_:+.4f} "
                    f"{self.holdout_metric_} over {win_rate:.0%} of folds; "
                    "emitting no features",
                    level=1,
                )
                self.selected_ = []
                self.feature_names_ = []
                self.feature_formulas_ = {}
        else:
            self._near_misses = []
            progress.note("no usable columns or too few rows; nothing generated", level=1)

        self._history = history
        self._elapsed = time.monotonic() - t_start
        self._fitted = True
        self.report_ = build_report(self)
        progress.finish(len(self.selected_), self.holdout_delta_, self.holdout_metric_)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureCrafter is not fitted; call fit first")
        X = self._as_frame(X)
        missing = [c for c in self.columns_in_ if c not in X.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        X = X[self.columns_in_]
        # Base the output on the datetime-expanded frame: re-emitting raw
        # datetime columns would hand a datetime64 dtype to the downstream
        # model, which most estimators cannot consume.
        X_work = expand_datetime(X, self._dt_cols)
        out = X_work.copy()
        for name, (tree, _) in zip(self.feature_names_, self.selected_):
            out[name] = np.asarray(tree.values(X_work, self._ops), dtype=float)
        return out

    def fit_transform(self, X, y, task: str | None = None) -> pd.DataFrame:
        return self.fit(X, y, task).transform(X)

    def get_feature_names(self) -> list[str]:
        return list(self.feature_names_)

    def get_feature_formulas(self) -> dict[str, str]:
        return dict(self.feature_formulas_)

    # -------------------------------------------------------------- reports
    def save_report(self, path) -> None:
        if not self._fitted:
            raise RuntimeError("FeatureCrafter is not fitted; call fit first")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.report_)

    # ---------------------------------------------------------- persistence
    def to_json(self, path=None) -> str:
        if not self._fitted:
            raise RuntimeError("FeatureCrafter is not fitted; call fit first")
        payload = {
            "version": 1,
            "task": self.task_,
            "columns_in": self.columns_in_,
            "dt_cols": list(self._dt_cols),
            "types": {
                "numeric": list(self.types_.numeric),
                "categorical": list(self.types_.categorical),
                "datetime": list(self.types_.datetime),
            },
            "operators": list(self._ops),
            "features": [
                {"name": name, "fitness": fit, "tree": tree.to_dict()}
                for name, (tree, fit) in zip(self.feature_names_, self.selected_)
            ],
            "operator_stats": self.operator_stats_,
            "holdout_delta": self.holdout_delta_,
            "holdout_metric": self.holdout_metric_,
        }
        text = json.dumps(payload)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_json(cls, source) -> "FeatureCrafter":
        if isinstance(source, str) and source.lstrip().startswith("{"):
            payload = json.loads(source)
        else:
            with open(source, encoding="utf-8") as fh:
                payload = json.load(fh)
        fc = cls(task=payload["task"])
        fc.task_ = payload["task"]
        fc.columns_in_ = payload["columns_in"]
        fc._dt_cols = tuple(payload["dt_cols"])
        fc.types_ = ColumnTypes(
            tuple(payload["types"]["numeric"]),
            tuple(payload["types"]["categorical"]),
            tuple(payload["types"]["datetime"]),
        )
        fc._ops = select_operators(payload["operators"])
        fc.selected_ = [
            (FeatureTree.from_dict(f["tree"]), f["fitness"])
            for f in payload["features"]
        ]
        fc.feature_names_ = [f["name"] for f in payload["features"]]
        fc.feature_formulas_ = {
            name: tree.formula()
            for name, (tree, _) in zip(fc.feature_names_, fc.selected_)
        }
        fc.operator_stats_ = payload.get("operator_stats", {})
        fc.holdout_delta_ = payload.get("holdout_delta")
        fc.holdout_metric_ = payload.get("holdout_metric", "")
        fc._history = []
        fc._near_misses = []
        fc._elapsed = 0.0
        fc._fitted = True
        fc.report_ = build_report(fc)
        return fc

    # ------------------------------------------------------------- internal
    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    @staticmethod
    def _subsample_index(rng: np.random.Generator, n: int) -> np.ndarray:
        if n <= _SUBSAMPLE_ROWS:
            return np.arange(n)
        return np.sort(rng.choice(n, size=_SUBSAMPLE_ROWS, replace=False))

    def _residuals(
        self, X_work: pd.DataFrame, y: np.ndarray, progress
    ) -> tuple[np.ndarray, np.ndarray]:
        """Out-of-fold residuals: what the base model cannot yet explain.

        Fitting and predicting on the same rows would leave a residual that is
        mostly memorised noise -- a 100-tree LightGBM reproduces its training
        target closely -- and the whole search would then be chasing that
        noise.  OpenFE uses out-of-fold predictions here for the same reason.

        Returns a 2-D residual of shape (n, k): k=1 for regression and binary
        classification, k=n_classes for multiclass, where column c is the
        one-vs-rest gradient for class c.  The previous label-code expectation
        (`y_codes - proba @ arange`) treated class labels as ordinal, which is
        only meaningful for a binary target.
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        enc, _ = encode_for_model(X_work, self.types_)
        n = len(enc)

        if self.task_ == "classification":
            y_codes, _ = pd.factorize(pd.Series(y), sort=True)
            n_classes = int(y_codes.max()) + 1
            counts = np.bincount(y_codes, minlength=n_classes)
            n_splits = int(min(_RESIDUAL_FOLDS, counts[counts > 0].min()))
            onehot = np.eye(n_classes)[y_codes]
            if n_splits < 2:
                proba = self._insample_proba(enc, y_codes, n_classes)
                progress.note(
                    "too few members in the rarest class for out-of-fold "
                    "residuals; falling back to in-sample",
                    level=1,
                )
            else:
                splitter = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                )
                proba = np.zeros((n, n_classes), dtype=float)
                for tr, te in splitter.split(enc, y_codes):
                    model = make_lgbm(self.task_, self.random_state, self.n_jobs)
                    model.fit(enc.iloc[tr], y_codes[tr])
                    # a fold can miss a class entirely; map back by classes_
                    p = model.predict_proba(enc.iloc[te])
                    proba[np.ix_(te, np.asarray(model.classes_, dtype=int))] = p
                progress.note(
                    f"base model fitted ({n_splits}-fold out-of-fold); "
                    "evolving against class residuals"
                )
            resid = onehot - proba
            if n_classes == 2:
                # the two columns are exact negatives; one carries the signal
                resid = resid[:, 1:2]
                # a linear probability model is a fair stand-in for the linear
                # panel here; for multiclass the OvR columns already dominate
                resid = self._with_linear_residual(
                    enc, resid, y_codes.astype(float))
            return resid, y_codes

        y = y.astype(float)
        n_splits = int(min(_RESIDUAL_FOLDS, n // 2))
        if n_splits < 2:
            model = make_lgbm(self.task_, self.random_state, self.n_jobs)
            model.fit(enc, y)
            pred = model.predict(enc)
            progress.note(
                "too few rows for out-of-fold residuals; falling back to in-sample",
                level=1,
            )
        else:
            splitter = KFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            pred = np.zeros(n, dtype=float)
            for tr, te in splitter.split(enc):
                model = make_lgbm(self.task_, self.random_state, self.n_jobs)
                model.fit(enc.iloc[tr], y[tr])
                pred[te] = model.predict(enc.iloc[te])
            progress.note(
                f"base model fitted ({n_splits}-fold out-of-fold); "
                "evolving against regression residuals"
            )
        return self._with_linear_residual(enc, (y - pred).reshape(-1, 1), y), y

    def _with_linear_residual(
        self, enc: pd.DataFrame, resid: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Append what a *linear* base model cannot explain.

        A gradient-boosted tree's residual is the wrong target when the
        features will also feed Ridge or kNN.  Once the tree has explained
        everything it can, that residual is noise and nothing scores above
        zero -- so no features get generated at all, even on datasets where a
        nonlinear reshaping would lift the linear model substantially.  What
        helps there is not extra signal but a change of *form*, and the only
        way to see it is to score against a linear model's residual as well.

        Scored as an extra residual column: `fitness_one` already takes the
        best over columns, so a candidate need only explain one of them.
        """
        if self.downstream == "tree":
            return resid
        try:
            from sklearn.model_selection import KFold

            X = enc.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            n = len(X)
            n_splits = int(min(_RESIDUAL_FOLDS, n // 2))
            if n_splits < 2:
                return resid
            pred = np.zeros(n, dtype=float)
            target = y.astype(float)
            for tr, te in KFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            ).split(X):
                model = _make_linear_base()
                model.fit(X[tr], target[tr])
                pred[te] = model.predict(X[te])
            lin = (target - pred).reshape(-1, 1)
            if not np.all(np.isfinite(lin)):
                return resid
            return np.hstack([resid, lin])
        except Exception:
            return resid

    def _insample_proba(self, enc, y_codes, n_classes: int) -> np.ndarray:
        model = make_lgbm(self.task_, self.random_state, self.n_jobs)
        model.fit(enc, y_codes)
        proba = np.zeros((len(enc), n_classes), dtype=float)
        proba[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(enc)
        return proba

    def _make_names(self, trees: list[FeatureTree], taken: set[str]) -> list[str]:
        names = []
        for tree in trees:
            base = _safe_name(tree.formula())
            name, k = base, 2
            while name in taken:
                name = f"{base}__{k}"
                k += 1
            taken.add(name)
            names.append(name)
        return names

    def __repr__(self) -> str:
        if not self._fitted:
            return "FeatureCrafter(unfitted)"
        return (
            f"FeatureCrafter(fitted, {len(self.selected_)} features, "
            f"task={self.task_!r})"
        )
