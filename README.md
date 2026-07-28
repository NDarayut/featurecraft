# featurecraft

Automatic feature engineering for tabular data, driven by an RL-guided genetic
search. One class, sensible defaults, readable output: every engineered feature
is a human-readable formula, and every run produces a markdown report that says
what was built and whether it actually helped.

```bash
pip install -e .
```

## Usage

```python
from featurecraft import FeatureCrafter

fc = FeatureCrafter()
X_new = fc.fit_transform(X, y)      # original columns + engineered columns
X_test_new = fc.transform(X_test)   # leak-safe replay on new data

fc.feature_formulas_    # {"(x2 * x3)": "(x2 * x3)", "mean(income) by (region)": ...}
fc.holdout_delta_       # did it actually help? (cross-validated, e.g. +0.014 r2)
fc.save_report("run_report.md")
```

Raw pandas frames go in as-is: categoricals, missing values and datetime
columns are handled natively. Classification and regression are both supported,
and the task is inferred from `y` when not given.

Tell it what the features are for — the answer differs by model:

```python
FeatureCrafter(downstream="tree")    # gradient boosting
FeatureCrafter(downstream="linear")  # Ridge, logistic regression, kNN
FeatureCrafter(downstream="mixed")   # default: both
```

If the input is already encoded, say which columns were categorical, since
inference cannot recover that from integer codes:

```python
FeatureCrafter(categorical_features=["region", "device_id"])
```

A fitted instance round-trips to JSON, including all replay state:

```python
fc.to_json("model.json")
fc = FeatureCrafter.from_json("model.json")
```

## How it works

1. **Out-of-fold base residuals.** LightGBM is fitted K-fold on the original
   features, and its out-of-fold predictions define what is still unexplained —
   so new features are only rewarded for signal the originals don't already
   carry. Out-of-fold matters: in-sample residuals from a 100-tree model are
   largely memorised noise, and the search would chase that noise.
   Classification uses the one-vs-rest gradient, one column per class. Unless
   `downstream="tree"`, a linear model's residual is added as another column,
   because a tree's residual cannot reveal where a *reshaping* of a feature
   would help Ridge or kNN.

2. **Candidate features are typed formula trees.** Leaves are columns; nodes
   are operators — `log1p`, `sqrt`, `square`, `reciprocal`, `abs`, `+ - * /`,
   `min`, `max`, frequency encoding, groupby mean/std/min/max/median,
   group-relative deviation/z-score/rank, `nunique`, and category crosses. The
   first-order space is enumerated broadly; the genetic search composes
   order-2 and above on top of whatever survives.

3. **FeatureBoost fitness.** A candidate scores by the share of the residual it
   explains on its own, fitted over quantile bins and estimated out-of-fold,
   minus a small parsimony tie-break. Pure numpy — no model is trained per
   candidate, which is what keeps the search fast. Binning rather than
   correlation is deliberate: a rank correlation is invariant to monotone
   transforms and blind to non-monotone structure, scoring a planted `z²`
   relationship at 0.015 where this scores 0.99.

4. **Successive halving.** Candidates start on a small block of rows; the
   better half survives each round as the data doubles, so only the finalists
   are scored on the full sample. Cheaper than scoring everything once, and far
   less noisy than a fixed subsample.

5. **Genetic search with an RL operator policy.** Tournament selection, subtree
   crossover, point mutation, elitism, duplicate elimination and a hall of fame
   of the best unique features ever seen. Every operator choice is made by a
   UCB1 bandit rewarded with the offspring's improvement over its parent: early
   generations explore all operators, later ones exploit whatever works *on
   this dataset*. The learned table is exposed as `fc.operator_stats_`. Early
   stopping and an optional wall-clock `time_budget` make the loop anytime —
   stop it whenever and keep the best so far.

6. **Selection, attribution, gatekeeper.** Redundant candidates are pruned —
   against ranks for `downstream="tree"`, against Pearson otherwise, because a
   tree is invariant to monotone transforms but Ridge and kNN are not. The
   shortlist is then re-ranked by LightGBM gain importance *in the presence of
   the base features*, which accounts for interactions the per-candidate score
   cannot see. Finally a K-fold gatekeeper measures the with-vs-without delta,
   and unless it is positive across a majority of folds, **no features are
   emitted at all** — feature generation genuinely does not always help, and
   shipping features that measurably hurt is worse than shipping none.

Deterministic given `random_state`. Leak-safe by construction: all fitted state
(frequency maps, groupby aggregates, cross codes) is a function of the training
X only — never of y — and `transform` is pure replay.

## Options

```python
FeatureCrafter(
    task=None,                 # "classification" | "regression" | inferred
    operators=None,            # e.g. ["mul", "div", "groupby_mean"] to restrict
    categorical_features=None, # override type inference on pre-encoded input
    population_size=200, generations=25, max_depth=3,
    crossover_rate=0.6, mutation_rate=0.3, tournament_k=3, elitism=10,
    parsimony=0.002,           # complexity tie-break; higher -> simpler formulas
    rl_policy=True, ucb_c=1.4,
    max_new_features=None,     # default min(2 * n_cols, 50)
    redundancy_threshold=0.98,
    downstream="mixed",        # "tree" | "linear" | "mixed"
    gate=True,                 # emit nothing when the holdout delta isn't positive
    n_jobs=1,                  # parallel fitness evaluation (joblib)
    time_budget=None,          # seconds; anytime stop
    random_state=0,
    verbose=1,                 # 0 silent, 1 per-generation lines, 2 detail
)
```

## Requirements

Python ≥ 3.10, with numpy, pandas, scikit-learn, lightgbm and joblib.

## License

MIT
