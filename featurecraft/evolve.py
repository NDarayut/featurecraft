"""The evolutionary loop.

Fitness is |Spearman rank correlation| between a candidate's values and the
base model's residuals, computed on a fixed row subsample, minus a parsimony
penalty per tree node.  No model is trained per candidate — fitness is pure
numpy, which is what makes the search fast (evaluation, not generation, is
the field's bottleneck).
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
    parsimony: float = 0.01
    hof_capacity: int = 150
    early_stop: int = 8
    n_jobs: int = 1


class Deadline:
    def __init__(self, budget: float | None):
        self.t_end = None if budget is None else time.monotonic() + budget

    def expired(self) -> bool:
        return self.t_end is not None and time.monotonic() >= self.t_end


def rank_array(v: np.ndarray) -> np.ndarray:
    return np.array(pd.Series(v).rank(method="average"), dtype=float)


def _rank_corr(v: np.ndarray, r: np.ndarray) -> float:
    rv = rank_array(v)
    rr = rank_array(r)
    rv -= rv.mean()
    rr -= rr.mean()
    denom = np.sqrt((rv**2).sum() * (rr**2).sum())
    if denom < 1e-12:
        return 0.0
    return abs(float((rv * rr).sum() / denom))


def fitness_one(
    tree: FeatureTree,
    X_sub: pd.DataFrame,
    resid: np.ndarray,
    ops: dict[str, Operator],
    parsimony: float,
) -> float:
    """Fitness of one tree on the subsample (state fitted here is throwaway)."""
    try:
        vals = np.asarray(tree.fit_values(X_sub, ops), dtype=float)
    except Exception:
        return 0.0
    mask = np.isfinite(vals) & np.isfinite(resid)
    if mask.mean() < 0.2 or mask.sum() < 10:
        return 0.0
    v = vals[mask]
    if np.std(v) < 1e-12:
        return 0.0
    corr = _rank_corr(v, resid[mask])
    return max(corr - parsimony * tree.n_nodes(), 0.0)


def _fitness_chunk(trees, X_sub, resid, ops, parsimony):
    return [fitness_one(t, X_sub, resid, ops, parsimony) for t in trees]


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
    resid_sub: np.ndarray,
    types,
    ops: dict[str, Operator],
    policy,
    cfg: EvolveConfig,
    rng: np.random.Generator,
    progress,
    deadline: Deadline,
) -> tuple[HallOfFame, list[dict]]:
    """Run the GA; return the hall of fame and the per-generation history."""
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

    seeds = depth1_candidates(types, ops, limit=cfg.population_size)
    if seeds:
        seed_scores = evaluate(seeds)
        ranked = sorted(zip(seed_scores, range(len(seeds))), key=lambda p: -p[0])
        for _, i in ranked[: cfg.population_size // 2]:
            f = seeds[i].formula()
            if f not in seen:
                seen.add(f)
                population.append(seeds[i])

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
