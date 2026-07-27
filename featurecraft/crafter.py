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

from .evolve import Deadline, EvolveConfig, evolve
from .feature import FeatureTree
from .operators import select_operators
from .policy import OperatorBandit, UniformPolicy
from .progress import ProgressLog
from .report import build_report
from .select import encode_for_model, gatekeeper, make_lgbm, select_features
from .types import ColumnTypes, expand_datetime, infer_task, infer_types

_SUBSAMPLE_ROWS = 2000


class FeatureCrafter:
    """Automatic feature engineering via RL-guided genetic search."""

    name = "featurecraft"

    def __init__(
        self,
        task: str | None = None,
        operators: list[str] | tuple[str, ...] | None = None,
        population_size: int = 200,
        generations: int = 25,
        max_depth: int = 3,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,
        tournament_k: int = 3,
        elitism: int = 10,
        parsimony: float = 0.01,
        rl_policy: bool = True,
        ucb_c: float = 1.4,
        max_new_features: int | None = None,
        redundancy_threshold: float = 0.98,
        n_jobs: int = 1,
        time_budget: float | None = None,
        random_state: int = 0,
        verbose: int = 1,
        stream=None,
    ):
        self.task = task
        self.operators = tuple(operators) if operators is not None else None
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
            sub_idx = self._subsample_index(rng, len(X_work))
            X_sub = X_work.iloc[sub_idx].reset_index(drop=True)
            resid_sub = resid[sub_idx]

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
                X_sub, resid_sub, self.types_, ops, policy, cfg, rng, progress, deadline
            )
            self.operator_stats_ = policy.stats()

            hof_entries = hof.best()
            self.selected_ = select_features(
                hof_entries,
                X_work,
                X_sub,
                self.types_,
                ops,
                max_new,
                self.redundancy_threshold,
                progress,
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

            self.holdout_delta_, self.holdout_metric_ = gatekeeper(
                X_work,
                y_codes if self.task_ == "classification" else y.astype(float),
                self.task_,
                [t for t, _ in self.selected_],
                self.types_,
                ops,
                self.random_state,
                self.n_jobs,
            )
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
        X_work = expand_datetime(X, self._dt_cols)
        out = X.copy()
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
        """Fit the base model once; residual = what it cannot yet explain."""
        enc, _ = encode_for_model(X_work, self.types_)
        model = make_lgbm(self.task_, self.random_state, self.n_jobs)
        if self.task_ == "classification":
            y_codes, _ = pd.factorize(pd.Series(y), sort=True)
            model.fit(enc, y_codes)
            proba = model.predict_proba(enc)
            expected = proba @ np.arange(proba.shape[1])
            resid = y_codes.astype(float) - expected
            progress.note("base model fitted; evolving against class residuals")
            return resid, y_codes
        y = y.astype(float)
        model.fit(enc, y)
        resid = y - model.predict(enc)
        progress.note("base model fitted; evolving against regression residuals")
        return resid, y

    def _make_names(self, trees: list[FeatureTree], taken: set[str]) -> list[str]:
        names = []
        for tree in trees:
            base = tree.formula()
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
