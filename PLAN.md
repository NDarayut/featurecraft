# Plan: `featurecraft` — our own RL-guided genetic feature engineering tool, from scratch

## Context

The `auto-feature-engineering` repo stays a pure benchmark harness — **nothing is ever committed to it**. We build a new, self-contained repository at `/run/media/pc/disk1/featurecraft` (own git history, own pyproject, zero imports from `afe`, no wrapped third-party AutoFE methods). One algorithm, ours, implemented from scratch; LightGBM appears only as the internal scoring model, the way every paper uses some downstream learner.

**Why GA + simple RL.** The user asked for something simpler than a multi-phase cascade but still good, incorporating both evolutionary search and reinforcement learning in an easy-to-use form. The design: a **genetic algorithm explores** the space of feature formulas, and a **reinforcement-learning policy guides** it — a UCB bandit (the simplest, most robust form of RL; the exact framing FUSE built on) that learns *during the run* which operators and column types produce fitter offspring, and biases mutation and tree generation toward them. This captures the operator-as-action idea from the RL papers (CAFEM, E-AFE) without their heavy machinery — no neural networks, no replay buffers, no reward shaping, no training instability; the policy is a per-operator statistics table updated online. Evolutionary search itself is backed by `feat.pdf` (GP-based feature construction) and LLM-FE (whose backbone is evolutionary search with a population memory). The GA gives natural feature composition (trees grow via crossover/mutation), anytime behavior (stop at any generation), obvious progress tracking, and interpretable output (every individual *is* a readable formula). We keep the literature's two hardest-won lessons in the fitness design: score against the **residual** of a base model (OpenFE — reward only what the base features can't already explain) and score **cheaply on a subsample** (E-AFE/FUSE — evaluation is ~90% of runtime, so fitness must be numpy-fast, not model-fits).

**The FeatureCraft algorithm (one evolutionary loop):**
1. Fit one base LightGBM on the original features; compute residuals (regression: y − pred; classification: true-class code − expected code). This happens **once**.
2. Initialize a population of random small formula trees over typed columns (unary/binary arithmetic, freq, groupby aggregations, category crosses — the operator vocabulary the whole literature converged on), seeded partly with depth-1 candidates ranked by a quick association screen.
3. Evolve for G generations: fitness = |rank correlation| between the candidate's values and the residuals, computed on a fixed row subsample with throwaway state — pure numpy, parallelizable, no model training per candidate. Tournament selection, subtree crossover, point mutation (swap operator / swap input / grow / prune), elitism, formula-dedup, and a fitness penalty on tree depth (parsimony pressure → simple, interpretable features win ties).
4. **RL operator policy**: every new tree node and every mutation picks its operator by asking a UCB1 bandit; when an offspring's fitness is computed, the bandit is rewarded with the fitness improvement over its parent (clipped to [0, 1]). Early generations explore operators uniformly; later generations exploit whichever operators actually work *on this dataset* — the search adapts online instead of sampling blindly.
5. A hall of fame collects the best unique individuals ever seen. After the last generation: greedy redundancy prune (|Spearman| > 0.98 vs originals and already-accepted features), fit surviving features' replay state on the full training data, keep top `max_new_features`.
6. Gatekeeper: internal 80/20 holdout, LightGBM score with vs without the new features; report the delta and warn when ≤ 0 (the OpenFE-19/68-datasets lesson, built in).

Deterministic given a seed, leak-safe (all fitted state from X-train only; no target encoding), fully offline.

## Repository layout (small on purpose)

```
/run/media/pc/disk1/featurecraft/
    pyproject.toml          # deps: numpy, pandas, scikit-learn, lightgbm, joblib; python >=3.10
    README.md               # quickstart, the algorithm in one page, paper lineage
    .gitignore
    featurecraft/
        __init__.py         # exports FeatureCrafter, __version__
        crafter.py          # FeatureCrafter: public API + orchestration (base model, GA, select, gatekeep)
        types.py            # column typing (numeric/categorical/datetime), NaN policy
        operators.py        # fixed operator vocabulary (subset-selectable via config)
        feature.py          # FeatureTree: formula tree + fitted state + leak-safe replay + random gen/crossover/mutation
        evolve.py           # the GA loop: fitness, tournament, elitism, hall of fame, deadline checks
        policy.py           # RL: UCB1 operator bandit (~50 lines) guiding mutation/generation
        select.py           # redundancy prune + final cap + holdout gatekeeper
        progress.py         # verbosity levels, per-generation lines, ETA, stderr
        report.py           # markdown run report
    tests/
        test_types.py  test_operators.py  test_feature.py  test_evolve.py
        test_select.py  test_crafter.py  test_report.py
```

`git init` there; first commit is the scaffold.

## Public API — one class

```python
from featurecraft import FeatureCrafter

fc = FeatureCrafter(
    task=None,                  # "classification" | "regression" | None -> inferred from y
    operators=None,             # subset of built-in vocabulary names; None = all
    population_size=200, generations=25, max_depth=3,
    crossover_rate=0.6, mutation_rate=0.3, tournament_k=3, elitism=10,
    parsimony=0.01,             # fitness penalty per tree node
    rl_policy=True, ucb_c=1.4,  # RL operator guidance; False -> uniform random operators
    max_new_features=None,      # default min(2*n_cols, 50)
    redundancy_threshold=0.98,
    n_jobs=1, time_budget=None, random_state=0, verbose=1,
)
fc.fit(X, y)                        # raw pandas: cats/NaNs/datetimes handled natively
X_new = fc.transform(X_test)        # original cols + engineered cols
X_new = fc.fit_transform(X, y)      # also accepts task as 3rd positional
fc.feature_names_, fc.feature_formulas_        # e.g. "log1p(income) / mean(income) by (region)"
fc.report_ / fc.save_report("run_report.md")
fc.to_json(path) / FeatureCrafter.from_json(path)
```

The class defines `name = "featurecraft"`, so it structurally satisfies the benchmark harness's `AutoFEMethod` protocol — benchmarking later is just `pip install -e ../featurecraft` in the benchmark venv and passing the class to `compare()`. No reference to the benchmark repo exists in featurecraft, or vice versa.

## Input handling (`types.py`)

On fit: numeric (float/high-cardinality int), categorical (object/category/bool, or low-cardinality integer: 2 ≤ nunique ≤ max(30, √n)), datetime (expanded to year/month/dow/hour numeric components). NaNs stay — operators propagate NaN; categorical state treats NaN as its own level. No target encoding (leakage safety).

## Feature trees (`feature.py`) — representation, genetics, leak-safety

`FeatureTree(op, inputs, state)` — inputs are column names (leaves) or nested trees, depth ≤ `max_depth`.
- **Replay**: `fit_values(X_train)` computes values and captures state (`freq` → count map; `groupby_mean/std/min/max` → per-category aggregate + global fallback, min group size 5; `cat_cross` → pair→code map, unseen → −1). State comes from X only. `values(X)` is pure replay — used by `transform`, never refits. During evolution, fitness uses throwaway state fitted on the subsample; final state is fitted once on full train in `select`.
- **Genetics**: `random_tree(rng, depth)` (typed — ops only get compatible inputs); `crossover(a, b, rng)` = swap random subtrees (type-compatible); `mutate(t, rng)` = one of {swap operator, swap input column, grow leaf into subtree, prune subtree to leaf}. Depth clamped; invalid offspring (constant output, >80% NaN on the subsample) get fitness 0.
- `formula()` renders readable strings (dedup key); `to_dict/from_dict` for JSON persistence.

## Operator vocabulary (`operators.py`)

All NaN-safe; guards emit NaN, never inf. Unary numeric: `log1p`, `sqrt`, `square`, `reciprocal`, `abs`. Binary numeric: `add`, `sub`, `mul`, `div`. Categorical: `freq`, `cat_cross` (cardinality product ≤ 1000). Groupby num-by-cat: `mean`, `std`, `min`, `max`. Fixed vocabulary — part of the algorithm; `operators=` selects a subset by name.

## The GA loop (`evolve.py`)

Seeded `np.random.Generator` throughout. Fitness subsample = min(n, 2000) rows, fixed at start.

- **Init**: half the population random trees (depth 1–2), half depth-1 candidates ranked by |rank correlation with residuals| (a quick screen so generation 0 isn't all noise). Tree generation draws operators from the RL policy.
- **Fitness**: rank-transform candidate values on the subsample once, then |Pearson on ranks| against rank-transformed residuals, minus `parsimony * n_nodes`. Invalid → 0. Evaluated joblib-parallel in fixed-size chunks keyed by index (deterministic under any n_jobs). Per-individual memoization by formula string (populations re-visit formulas often).
- **Generation step**: elitism (top `elitism` copied), then fill by tournament-selected parents → crossover (p=0.6) or clone → mutation (p=0.3); dedup by formula (mutate duplicates again, up to 3 tries).
- **Hall of fame**: top 3×`max_new_features` unique individuals ever seen, by fitness.
- **Stopping**: `generations` reached, or `time_budget` deadline hit (checked each generation — anytime behavior for free), or no hall-of-fame improvement for 8 generations (early stop).

## The RL operator policy (`policy.py`)

`OperatorBandit` — plain UCB1 over the operator vocabulary, one arm per operator (optionally per (operator, input-type) pair; start with per-operator only):
- `choose(rng, valid_ops) -> op`: pick `argmax(mean_reward + ucb_c * sqrt(ln(total_pulls) / pulls))` among currently-valid operators; unpulled arms first (guaranteed initial exploration).
- `update(op, reward)`: reward = `clip(child_fitness − parent_fitness, 0, 1)` for mutations, `clip(fitness, 0, 1)` for fresh trees; incremental mean update. Called once per evaluated offspring, from the main process (no cross-process state).
- Deterministic given the seeded rng; `rl_policy=False` falls back to uniform choice (also our ablation baseline in tests).
- The learned table (`pulls`, `mean_reward` per operator) is exposed as `fc.operator_stats_` and printed in the report — a genuinely useful output in its own right ("on this dataset, ratios and groupby-means paid off; log transforms did nothing").

This is the whole RL component: exploration/exploitation with an online reward signal, ~50 lines, no training loop to babysit — simple and easy to use, per the design constraint.

## Selection & gatekeeper (`select.py`)

Hall of fame in fitness order → greedy redundancy prune (|Spearman| on the subsample > `redundancy_threshold` vs original columns and already-accepted features) → fit replay state on full train → cap at `max_new_features`. Then the gatekeeper: internal 80/20 holdout split, one LightGBM with original features vs one with original+new; delta in the report, explicit warning when ≤ 0.

## Progress & verbosity (`progress.py`)

Stderr, ASCII, flushed; `stream=` injectable. `verbose=1`: one line per generation — `"[featurecraft] gen 12/25: best 0.412  mean 0.198  hof 31  ~14s left"` — plus start/finish summaries (`"selected 18 features in 38.2s, holdout delta +0.011"`). `verbose=2`: adds the current best formula each generation and selection/gatekeeper detail. `verbose=0`: silent. ETA = elapsed/generations-done × remaining.

## Report (`report.py`)

Markdown: config + data summary; evolution curve table (generation | best | mean | hall-of-fame size); selected-features table (name | formula | fitness | depth); learned operator-policy table (operator | pulls | mean reward); near-misses; holdout delta + warning if non-positive.

## Tests (pytest, synthetic, offline)

- types: column classification incl. low-cardinality ints, datetimes, NaN columns.
- operators: values + NaN domains (log of negative, div by ~0 → NaN not inf).
- feature: fit_values ≡ values replay for every op; unseen category → fallback; random_tree respects typing and depth; crossover/mutation produce valid typed trees; formula round-trip via to_dict/from_dict preserves transform output.
- policy: UCB1 explores every arm first; converges to a rigged high-reward arm; deterministic under fixed seed; `rl_policy=False` path.
- evolve: fitness memoization; deterministic population trajectory for a fixed seed regardless of n_jobs; planted signal (`y = log(x1) + x2*x3 + noise`) → hall of fame contains matching formulas within 25 generations; on that dataset the bandit's top arms include the planted operators (mul/log1p); early stop fires on a no-signal dataset.
- select: duplicated column pruned; caps respected; gatekeeper warning on pure-noise data.
- crafter end-to-end: recovers planted arithmetic and groupby signals; same-seed determinism of formulas and transform output; leak-safety (transform uses only fitted state); classification and regression; all-categorical / all-numeric / constant / tiny-n (30 rows) inputs; `time_budget≈0` returns valid (possibly feature-less) output fast; verbose capture at 0/1/2.
- report: evolution table + every formula present; warning when delta ≤ 0; save_report writes.

## Build order & verification

1. Scaffold repo (`git init`, pyproject, README stub); `pip install -e .` in a venv.
2. `types.py` → `operators.py` → `feature.py` (+ tests) — pure numpy/pandas, includes the genetic operators.
3. `policy.py` + `evolve.py` + `progress.py` (+ tests) — bandit, base-model residuals, GA loop.
4. `select.py` (+ tests) — pruning + gatekeeper.
5. `crafter.py` orchestration + persistence (+ end-to-end tests); `report.py` (+ tests).
6. Full `pytest`; README with quickstart, one-page algorithm description, paper-lineage table.
7. Smoke: run on a synthetic dataset and one real CSV; check formulas, report, verbose output at levels 1/2; confirm a planted signal is recovered.
8. (Optional, zero commits to the benchmark repo): `pip install -e ../featurecraft` there, then `compare(methods=[BaselineMethod, FeatureCrafter], datasets=["german-credit"])`.

## Future directions (algorithm evolution, not plugins)

Refit residuals mid-run (every ~10 generations) so evolution chases what's still unexplained; contextual bandit (operator × input-type arms, or a small tabular Q over transformation sequences à la CAFEM-lite); proper classification residuals (one-vs-rest); island populations for diversity (LLM-FE uses these); multi-table/relational operators.
