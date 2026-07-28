"""The evolutionary loop.

Fitness is a FeatureBoost score (OpenFE, ICML 2023, Algorithm 2): the share of
the base model's out-of-fold residual that a candidate can explain on its own,
estimated out-of-fold over quantile bins.  No model is trained per candidate --
scoring is pure numpy, which is what keeps the search fast (evaluation, not
generation, is the field's bottleneck).

This replaced |Spearman rank correlation|, which had two fatal blind spots for
a feature generator:

- Rank correlation is invariant to monotone transforms, so `x`, `log1p(x)` and
  `sqrt(x)` scored *identically* (verified: equal to 10 decimal places).
- It only sees monotone dependence.  Against a planted `z**2` residual,
  |Spearman| scores 0.015 -- indistinguishable from noise -- where the binned
  score sees 0.990.  Quadratic and V-shaped structure was simply invisible.

The OpenFE paper's own ablation (Table 6, "OpenFE-MI") reports that swapping
FeatureBoost for a univariate criterion is worse on every dataset tested.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .feature import FeatureTree, crossover, depth1_candidates, mutate, random_tree
from .operators import Operator

_CHUNK = 64


@dataclasses.dataclass
class EvolveConfig:
    population_size: int = 200
    generations: int = 25
    max_depth: int = 3
    crossover_rate: float = 0.6
    mutation_rate: float = 0.3
    tournament_k: int = 3
    elitism: int = 10
    # A tie-break, not a real term in the objective.  At the old 0.01 a 5-node
    # tree -- i.e. a legitimate order-2 interaction -- was docked 0.05, which
    # is most of the margin between a good and a mediocre candidate, and the
    # search collapsed toward depth 1.  Now that fitness is an R^2-scale
    # quantity rather than a correlation, weak-but-real features score lower
    # still, so the old penalty would bite even harder.
    parsimony: float = 0.002
    hof_capacity: int = 150
    early_stop: int = 8
    n_jobs: int = 1
    # Order-1 candidates to enumerate before the GA runs.  OpenFE enumerates
    # the first-order space exhaustively -- O(d*m^2), a few thousand for a
    # typical table -- and that is a stronger guarantee than any amount of
    # random sampling.  The GA's job is order-2 and above, built on whatever
    # survives here.
    order1_budget: int = 20000


class Deadline:
    def __init__(self, budget: float | None):
        self.t_end = None if budget is None else time.monotonic() + budget

    def expired(self) -> bool:
        return self.t_end is not None and time.monotonic() >= self.t_end


def rank_array(v: np.ndarray) -> np.ndarray:
    return np.array(pd.Series(v).rank(method="average"), dtype=float)


N_BINS = 32


class ResidualTarget:
    """The residual being explained, plus everything constant across candidates.

    Holding the fold assignment and per-column total sum of squares here means
    they are computed once per run rather than once per candidate.
    """

    __slots__ = ("resid", "fold", "sst", "n_cols")

    def __init__(self, resid: np.ndarray, seed: int = 0):
        resid = np.asarray(resid, dtype=float)
        if resid.ndim == 1:
            resid = resid.reshape(-1, 1)
        self.resid = resid
        self.n_cols = resid.shape[1]
        # A fixed random fold assignment, not an alternating one: rows may
        # arrive sorted by the target, which would make parity folds
        # systematically unbalanced.
        rng = np.random.default_rng(seed)
        self.fold = rng.integers(0, 2, size=resid.shape[0])
        centred = resid - np.nanmean(resid, axis=0, keepdims=True)
        self.sst = np.nansum(centred**2, axis=0)

    def subset(self, idx: np.ndarray) -> "ResidualTarget":
        out = object.__new__(ResidualTarget)
        out.resid = self.resid[idx]
        out.fold = self.fold[idx]
        out.n_cols = self.n_cols
        centred = out.resid - np.nanmean(out.resid, axis=0, keepdims=True)
        out.sst = np.nansum(centred**2, axis=0)
        return out

    def __len__(self) -> int:
        return self.resid.shape[0]


def _bin_codes(v: np.ndarray, n_bins: int = N_BINS) -> tuple[np.ndarray, int] | None:
    """Quantile-bin a candidate's values into at most `n_bins` buckets."""
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    if edges.size == 0:
        return None
    codes = np.searchsorted(edges, v, side="right").astype(np.intp)
    return codes, int(edges.size) + 1


def _oof_bin_r2(codes: np.ndarray, n_codes: int, r: np.ndarray, fold: np.ndarray,
                sst: float) -> float:
    """Share of `r` explained by a piecewise-constant fit on `codes`.

    This is FeatureBoost with squared loss: the candidate alone predicts the
    residual, and the score is the resulting loss reduction.  Bin means are
    always taken from the *other* fold, so a high-cardinality feature cannot
    score well simply by isolating single rows.
    """
    if sst < 1e-12:
        return 0.0
    pred = np.empty(codes.size, dtype=float)
    for f in (0, 1):
        te = fold == f
        tr = ~te
        if not te.any():
            continue
        if not tr.any():
            pred[te] = 0.0
            continue
        cnt = np.bincount(codes[tr], minlength=n_codes)
        tot = np.bincount(codes[tr], weights=r[tr], minlength=n_codes)
        gmean = float(r[tr].mean())
        means = np.where(cnt > 0, tot / np.maximum(cnt, 1), gmean)
        pred[te] = means[codes[te]]
    sse = float(((r - pred) ** 2).sum())
    return max(1.0 - sse / sst, 0.0)


def fitness_one(
    tree: FeatureTree,
    X_sub: pd.DataFrame,
    target: ResidualTarget,
    ops: dict[str, Operator],
    parsimony: float,
) -> float:
    """Fitness of one tree on the given rows (state fitted here is throwaway)."""
    if not isinstance(target, ResidualTarget):
        target = ResidualTarget(target)
    try:
        vals = np.asarray(tree.fit_values(X_sub, ops), dtype=float)
    except Exception:
        return 0.0
    mask = np.isfinite(vals)
    if mask.mean() < 0.2 or mask.sum() < 10:
        return 0.0
    v = vals[mask]
    if np.std(v) < 1e-12:
        return 0.0
    binned = _bin_codes(v)
    if binned is None:
        return 0.0
    codes, n_codes = binned
    fold = target.fold[mask]
    all_rows = bool(mask.all())
    best = 0.0
    for c in range(target.n_cols):
        # multiclass: columns are one-vs-rest gradients, and a feature that
        # explains any single class is worth keeping
        r = target.resid[mask, c]
        finite = np.isfinite(r)
        if finite.all():
            rr, cc, ff = r, codes, fold
            # the precomputed total only applies when no rows were dropped
            sst = float(target.sst[c]) if all_rows else float(
                ((r - r.mean()) ** 2).sum())
        else:
            if finite.sum() < 10:
                continue
            rr, cc, ff = r[finite], codes[finite], fold[finite]
            sst = float(((rr - rr.mean()) ** 2).sum())
        best = max(best, _oof_bin_r2(cc, n_codes, rr, ff, sst))
    return max(best - parsimony * tree.n_nodes(), 0.0)


def _fitness_chunk(trees, X_sub, target, ops, parsimony):
    return [fitness_one(t, X_sub, target, ops, parsimony) for t in trees]


def successive_halving(
    trees: list[FeatureTree],
    X: pd.DataFrame,
    target: ResidualTarget,
    ops: dict[str, Operator],
    parsimony: float,
    keep: int,
    min_rows: int = 512,
    max_rows: int = 50_000,
    n_jobs: int = 1,
    deadline: "Deadline | None" = None,
) -> list[tuple[float, FeatureTree]]:
    """Score a large candidate pool on geometrically increasing data.

    OpenFE's SuccessivePruning (Algorithm 3).  Start on a small block of rows,
    keep the better half, double the data, repeat -- so the cheap rounds
    eliminate most of the field and only the finalists are scored on the full
    dataset.  Roughly the cost of scoring everything once on the small block,
    but the surviving estimates are made on far more rows.

    This replaced scoring every candidate once on a fixed 2,000-row subsample.
    The OpenFE ablation (Table 6, "w.o. Successive") reports that subsampling
    instead of halving costs accuracy on essentially every dataset: a fixed
    small sample makes every estimate equally noisy, including the ones that
    decide the final answer.
    """
    if not trees:
        return []
    # Past a certain sample the ranking is already stable, and the last
    # doublings are the expensive ones; cap rather than always going to n.
    n = min(len(X), max_rows)
    rows = min(max(min_rows, 64), n)
    pool = list(trees)
    scores: list[float] = []
    while True:
        sub_X = X.iloc[:rows]
        sub_t = target.subset(np.arange(rows))
        scores = _score_pool(pool, sub_X, sub_t, ops, parsimony, n_jobs)
        if rows >= n or len(pool) <= keep or (deadline is not None and deadline.expired()):
            break
        order = np.argsort(scores)[::-1]
        nxt = max(keep, len(pool) // 2)
        pool = [pool[int(i)] for i in order[:nxt]]
        rows = min(rows * 2, n)
    ranked = sorted(zip(scores, range(len(pool))), key=lambda p: -p[0])
    return [(s, pool[i]) for s, i in ranked]


def _score_pool(trees, X, target, ops, parsimony, n_jobs):
    if n_jobs != 1 and len(trees) > _CHUNK:
        chunks = [trees[i : i + _CHUNK] for i in range(0, len(trees), _CHUNK)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fitness_chunk)(c, X, target, ops, parsimony) for c in chunks
        )
        return [s for chunk in results for s in chunk]
    return _fitness_chunk(trees, X, target, ops, parsimony)


class HallOfFame:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries: dict[str, tuple[float, FeatureTree]] = {}

    def offer(self, tree: FeatureTree, fitness: float) -> None:
        if fitness <= 0:
            return
        key = tree.formula()
        cur = self.entries.get(key)
        if cur is None or fitness > cur[0]:
            self.entries[key] = (fitness, tree)
        if len(self.entries) > self.capacity * 2:
            self._trim()

    def _trim(self) -> None:
        keep = sorted(self.entries.items(), key=lambda kv: -kv[1][0])[: self.capacity]
        self.entries = dict(keep)

    def best(self) -> list[tuple[float, FeatureTree]]:
        self._trim()
        return sorted(self.entries.values(), key=lambda e: -e[0])

    def best_fitness(self) -> float:
        return max((f for f, _ in self.entries.values()), default=0.0)

    def __len__(self) -> int:
        return len(self.entries)


def evolve(
    X_sub: pd.DataFrame,
    resid_sub: "np.ndarray | ResidualTarget",
    types,
    ops: dict[str, Operator],
    policy,
    cfg: EvolveConfig,
    rng: np.random.Generator,
    progress,
    deadline: Deadline,
    X_full: pd.DataFrame | None = None,
    target_full: "ResidualTarget | None" = None,
) -> tuple[HallOfFame, list[dict]]:
    """Run the GA; return the hall of fame and the per-generation history.

    `X_full`/`target_full` are the unsubsampled data, used for the order-1
    successive-halving stage; the GA itself runs on `X_sub`.
    """
    if not isinstance(resid_sub, ResidualTarget):
        resid_sub = ResidualTarget(resid_sub)
    if X_full is None or target_full is None:
        X_full, target_full = X_sub, resid_sub
    memo: dict[str, float] = {}
    hof = HallOfFame(cfg.hof_capacity)
    history: list[dict] = []

    def evaluate(trees: list[FeatureTree]) -> list[float]:
        todo = [(i, t) for i, t in enumerate(trees) if t.formula() not in memo]
        if todo:
            todo_trees = [t for _, t in todo]
            if cfg.n_jobs != 1 and len(todo_trees) > _CHUNK:
                chunks = [
                    todo_trees[i : i + _CHUNK] for i in range(0, len(todo_trees), _CHUNK)
                ]
                results = Parallel(n_jobs=cfg.n_jobs)(
                    delayed(_fitness_chunk)(c, X_sub, resid_sub, ops, cfg.parsimony)
                    for c in chunks
                )
                scores = [s for chunk in results for s in chunk]
            else:
                scores = _fitness_chunk(todo_trees, X_sub, resid_sub, ops, cfg.parsimony)
            for (_, t), s in zip(todo, scores):
                memo[t.formula()] = s
        out = [memo[t.formula()] for t in trees]
        for t, s in zip(trees, out):
            hof.offer(t, s)
        return out

    # ---- init: half seeded depth-1 candidates, half random trees ----------
    population: list[FeatureTree] = []
    seen: set[str] = set()

    seeds = depth1_candidates(
        types, ops, limit=cfg.population_size, total=cfg.order1_budget
    )
    if seeds:
        # Order 1: enumerate broadly, then let successive halving pay for it.
        # Survivors are scored on the full data, so these are better estimates
        # than anything the GA produces on the subsample -- offer them to the
        # hall of fame directly.
        progress.note(
            f"order-1: {len(seeds)} candidates -> successive halving", level=2
        )
        ranked = successive_halving(
            seeds, X_full, target_full, ops, cfg.parsimony,
            keep=cfg.population_size, n_jobs=cfg.n_jobs, deadline=deadline,
        )
        for score, tree in ranked:
            hof.offer(tree, score)
        for _, tree in ranked[: cfg.population_size // 2]:
            f = tree.formula()
            if f not in seen:
                seen.add(f)
                population.append(tree)

    tries = 0
    while len(population) < cfg.population_size and tries < cfg.population_size * 20:
        tries += 1
        t = random_tree(rng, types, ops, policy, max_depth=min(cfg.max_depth, 2))
        if t is None or t.is_leaf:
            continue
        f = t.formula()
        if f in seen:
            continue
        seen.add(f)
        population.append(t)
    if not population:
        return hof, history

    fitnesses = evaluate(population)
    # reward the policy for the ops used in fresh random trees
    for t, f in zip(population, fitnesses):
        if not t.is_leaf:
            policy.update(t.op, f)

    best_seen = hof.best_fitness()
    stale = 0

    for gen in range(1, cfg.generations + 1):
        if deadline.expired():
            progress.note(f"time budget reached at generation {gen}", level=1)
            break

        order = np.argsort(fitnesses)[::-1]
        elite_idx = order[: cfg.elitism]
        next_pop: list[FeatureTree] = [population[int(i)] for i in elite_idx]
        next_seen = {t.formula() for t in next_pop}
        # (child_index, op_used, parent_fitness) for policy rewards
        records: list[tuple[int, str, float]] = []

        def tournament() -> int:
            idx = rng.integers(len(population), size=cfg.tournament_k)
            return int(max(idx, key=lambda i: fitnesses[int(i)]))

        attempts = 0
        while len(next_pop) < cfg.population_size and attempts < cfg.population_size * 10:
            attempts += 1
            i = tournament()
            parent = population[i]
            parent_fit = fitnesses[i]
            child: FeatureTree | None = None
            op_used: str | None = None
            if rng.random() < cfg.crossover_rate:
                j = tournament()
                child = crossover(parent, population[j], rng, ops, cfg.max_depth)
            if child is None:
                child = parent.copy()
            if rng.random() < cfg.mutation_rate or child.formula() == parent.formula():
                mutated, op_used = mutate(child, rng, types, ops, policy, cfg.max_depth)
                if mutated is not None:
                    child = mutated
            if child.is_leaf:
                continue
            f = child.formula()
            if f in next_seen:
                continue
            next_seen.add(f)
            if op_used is None and not child.is_leaf:
                # A crossover-only child previously recorded nothing, so
                # crossover contributed no learning signal to the bandit at
                # all -- only mutations ever taught it anything.  Credit the
                # op at the root of the child that crossover produced.
                op_used = child.op
            if op_used is not None:
                records.append((len(next_pop), op_used, parent_fit))
            next_pop.append(child)

        population = next_pop
        fitnesses = evaluate(population)
        for idx, op_used, parent_fit in records:
            policy.update(op_used, fitnesses[idx] - parent_fit)

        best = hof.best_fitness()
        mean = float(np.mean(fitnesses)) if fitnesses else 0.0
        top = hof.best()
        history.append(
            {"generation": gen, "best": best, "mean": mean, "hof_size": len(hof)}
        )
        progress.generation(
            gen, cfg.generations, best, mean, len(hof),
            top[0][1].formula() if top else "-",
        )

        if best > best_seen + 1e-9:
            best_seen = best
            stale = 0
        else:
            stale += 1
            if stale >= cfg.early_stop:
                progress.note(
                    f"early stop: no improvement for {cfg.early_stop} generations",
                    level=1,
                )
                break

    return hof, history
