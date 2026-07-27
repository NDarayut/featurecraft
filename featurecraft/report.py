"""Markdown run report."""

from __future__ import annotations


def build_report(fc) -> str:
    lines: list[str] = []
    lines.append("# featurecraft run report")
    lines.append("")
    lines.append(f"- task: {fc.task_}")
    lines.append(f"- input columns: {len(fc.columns_in_)}")
    lines.append(
        f"- column types: {len(fc.types_.numeric)} numeric, "
        f"{len(fc.types_.categorical)} categorical, "
        f"{len(fc.types_.datetime)} datetime"
    )
    lines.append(f"- selected features: {len(fc.selected_)}")
    if getattr(fc, "_elapsed", 0.0):
        lines.append(f"- fit time: {fc._elapsed:.1f}s")
    lines.append("")

    if fc.holdout_delta_ is not None:
        sign = "+" if fc.holdout_delta_ >= 0 else ""
        lines.append("## Holdout check")
        lines.append("")
        lines.append(
            f"Internal 80/20 holdout, LightGBM with vs without engineered "
            f"features: **{sign}{fc.holdout_delta_:.4f} {fc.holdout_metric_}**."
        )
        if fc.holdout_delta_ <= 0:
            lines.append("")
            lines.append(
                "> **Warning:** the engineered features did not improve the "
                "holdout score on this dataset. Feature generation does not "
                "always help; consider keeping the original features."
            )
        lines.append("")

    if fc.selected_:
        lines.append("## Selected features")
        lines.append("")
        lines.append("| # | feature | fitness | depth |")
        lines.append("|---|---------|---------|-------|")
        for i, (name, (tree, fit)) in enumerate(
            zip(fc.feature_names_, fc.selected_), start=1
        ):
            lines.append(f"| {i} | `{name}` | {fit:.3f} | {tree.depth()} |")
        lines.append("")

    if getattr(fc, "_history", None):
        lines.append("## Evolution")
        lines.append("")
        lines.append("| generation | best | mean | hall of fame |")
        lines.append("|-----------:|-----:|-----:|-------------:|")
        for h in fc._history:
            lines.append(
                f"| {h['generation']} | {h['best']:.3f} | {h['mean']:.3f} "
                f"| {h['hof_size']} |"
            )
        lines.append("")

    if fc.operator_stats_:
        lines.append("## Operator policy (learned during the run)")
        lines.append("")
        lines.append("| operator | pulls | mean reward |")
        lines.append("|----------|------:|------------:|")
        for op, s in fc.operator_stats_.items():
            lines.append(f"| {op} | {s['pulls']} | {s['mean_reward']:.4f} |")
        lines.append("")

    misses = getattr(fc, "_near_misses", [])
    if misses:
        lines.append("## Near misses (high fitness, not selected)")
        lines.append("")
        for fit, formula in misses:
            lines.append(f"- `{formula}` (fitness {fit:.3f})")
        lines.append("")

    return "\n".join(lines)
