# featurecraft

Automatic feature engineering for tabular data, driven by an RL-guided
genetic search. One class, sensible defaults, readable output: every
engineered feature is a human-readable formula, and every run produces a
markdown report that says what was built, why, and whether it actually
helped.

```python
from featurecraft import FeatureCrafter

fc = FeatureCrafter()
X_new = fc.fit_transform(X, y)      # original columns + engineered columns
X_test_new = fc.transform(X_test)   # leak-safe replay on new data

fc.feature_formulas_    # {"(x2 * x3)": "(x2 * x3)", "mean(income) by (region)": ...}
fc.holdout_delta_       # did it actually help? (internal holdout, e.g. +0.014 r2)
fc.save_report("run_report.md")
```

Raw pandas frames go in as-is: categoricals, missing values, and datetime
columns are handled natively. Classification and regression are supported
(the task is inferred from `y` when not given).

## The algorithm

One evolutionary loop, guided by a small reinforcement-learning policy:

1. **Base model residuals.** A single LightGBM is fitted on the original
   features once. Its residuals define what is still unexplained — new
   features are only rewarded for capturing signal the original features
   don't already carry.
2. **Population of formula trees.** Candidate features are small typed
   formula trees over the columns (`log1p`, `sqrt`, `square`, `reciprocal`,
   `abs`, `+ - * /`, frequency encoding, groupby mean/std/min/max,
   category crosses).
3. **Cheap fitness.** Fitness is the |Spearman rank correlation| between a
   candidate's values and the residuals, computed on a fixed row subsample,
   minus a parsimony penalty per node (simple formulas win ties). Pure
   numpy — no model is trained per candidate.
4. **Genetic search.** Tournament selection, subtree crossover, point
   mutation, elitism, duplicate elimination, and a hall of fame of the best
   unique features ever seen. Early stopping and an optional wall-clock
   `time_budget` (the loop is anytime: stop it whenever, keep the best so far).
5. **RL operator policy.** Every operator choice (fresh trees, mutations)
   is made by a UCB1 bandit rewarded with the offspring's fitness
   improvement. Early generations explore all operators; later generations
   exploit the ones that keep working *on this dataset*. The learned table
   is exposed as `fc.operator_stats_` and printed in the report.
6. **Selection + gatekeeper.** The hall of fame is greedily pruned for
   redundancy (|Spearman| > 0.98 against original columns or already-accepted
   features), survivors get their replay state fitted on the full training
   data, and an internal 80/20 holdout measures the with-vs-without delta —
   with an explicit warning when feature engineering did not help.

Deterministic given `random_state`. Leak-safe by construction: all fitted
state (frequency maps, groupby aggregates, cross codes) is a function of
the training X only — never of y — and `transform` is pure replay.

## Options

```python
FeatureCrafter(
    task=None,                # "classification" | "regression" | inferred
    operators=None,           # e.g. ["mul", "div", "groupby_mean"] to restrict
    population_size=200, generations=25, max_depth=3,
    crossover_rate=0.6, mutation_rate=0.3, tournament_k=3, elitism=10,
    parsimony=0.01,           # complexity penalty; higher -> simpler formulas
    rl_policy=True, ucb_c=1.4,
    max_new_features=None,    # default min(2 * n_cols, 50)
    redundancy_threshold=0.98,
    n_jobs=1,                 # parallel fitness evaluation (joblib)
    time_budget=None,         # seconds; anytime stop
    random_state=0,
    verbose=1,                # 0 silent, 1 per-generation lines, 2 detail
)
```

Persistence: `fc.to_json(path)` / `FeatureCrafter.from_json(path)` round-trip
a fitted instance, including all replay state.

## Research lineage

The design distills a set of automated-feature-engineering papers rather
than wrapping any of them:

| Idea in featurecraft | Source |
|---|---|
| Score candidates against a base model's residuals, never retrain per candidate | OpenFE (ICML 2023, "FeatureBoost") |
| Evaluate on subsamples; evaluation, not generation, is the bottleneck | E-AFE (2022), FUSE (ICML 2010) |
| Small symbolic operator vocabulary, interpretable by construction | LFE (IJCAI 2017), autofeat (2020), ExploreKit (ICDM 2016) |
| Evolutionary search over feature programs | Zhou & Hu (2023, GP feature construction), LLM-FE (2025, evolutionary backbone) |
| Operators as actions of a learning agent | CAFEM (PAKDD 2020), E-AFE — reduced here to a UCB1 bandit |
| Bandit/UCT framing of feature search | FUSE (ICML 2010) |
| Redundancy-aware final selection | GELFE (KBS 2025), AutoLearn (ICDM 2017, stability selection) |
| "Generation doesn't always help" gatekeeper | OpenFE §6 (no gain on 19/68 datasets), GELFE §6 |

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Tests are fully offline and synthetic (49 tests, ~10 s).
