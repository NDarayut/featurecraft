import numpy as np
import pandas as pd

from featurecraft import FeatureCrafter

FAST = dict(population_size=60, generations=6, verbose=0)


def test_report_contents(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = x1 * x2 + rng.normal(scale=0.05, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    fc = FeatureCrafter(random_state=0, **FAST).fit(X, y)

    report = fc.report_
    assert report.startswith("# featurecraft run report")
    assert "## Selected features" in report
    assert "## Evolution" in report
    assert "## Operator policy" in report
    for name in fc.feature_names_:
        assert name in report

    out = tmp_path / "report.md"
    fc.save_report(out)
    assert out.read_text().startswith("# featurecraft run report")


def test_report_warns_on_noise():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.normal(size=n)  # pure noise: engineering cannot help
    fc = FeatureCrafter(random_state=0, **FAST).fit(X, y)
    if fc.holdout_delta_ is not None and fc.holdout_delta_ <= 0:
        assert "Warning" in fc.report_
