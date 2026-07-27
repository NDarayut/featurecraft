"""featurecraft: RL-guided genetic automatic feature engineering.

Usage:
    from featurecraft import FeatureCrafter

    fc = FeatureCrafter()
    X_new = fc.fit_transform(X, y)
    fc.feature_formulas_          # what was built, as readable formulas
    fc.save_report("report.md")   # full run report
"""

from .crafter import FeatureCrafter

__version__ = "0.1.0"
__all__ = ["FeatureCrafter", "__version__"]
