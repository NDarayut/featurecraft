"""Progress reporting: ASCII lines to stderr, three verbosity levels.

verbose=0  silent
verbose=1  start line, one line per generation (best/mean fitness, ETA),
           selection + finish summary
verbose=2  additionally the current best formula per generation and
           selection/gatekeeper detail
"""

from __future__ import annotations

import sys
import time


def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class ProgressLog:
    def __init__(self, verbose: int = 1, stream=None):
        self.verbose = verbose
        self.stream = stream if stream is not None else sys.stderr
        self.t0 = time.monotonic()

    def _emit(self, level: int, msg: str) -> None:
        if self.verbose >= level:
            print(f"[featurecraft] {msg}", file=self.stream, flush=True)

    def start(self, n_rows: int, n_cols: int, task: str) -> None:
        self.t0 = time.monotonic()
        self._emit(1, f"fitting on {n_rows} rows x {n_cols} cols ({task})")

    def generation(
        self,
        gen: int,
        total: int,
        best: float,
        mean: float,
        hof_size: int,
        best_formula: str,
    ) -> None:
        elapsed = time.monotonic() - self.t0
        eta = (elapsed / max(gen, 1)) * (total - gen)
        self._emit(
            1,
            f"gen {gen}/{total}: best {best:.3f}  mean {mean:.3f}  "
            f"hof {hof_size}  ~{fmt_secs(eta)} left",
        )
        self._emit(2, f"  best formula: {best_formula}")

    def note(self, msg: str, level: int = 2) -> None:
        self._emit(level, msg)

    def finish(self, n_selected: int, delta: float | None, metric: str) -> None:
        elapsed = time.monotonic() - self.t0
        if delta is None:
            self._emit(1, f"selected {n_selected} features in {fmt_secs(elapsed)}")
            return
        sign = "+" if delta >= 0 else ""
        self._emit(
            1,
            f"selected {n_selected} features in {fmt_secs(elapsed)}, "
            f"holdout {metric} delta {sign}{delta:.4f}",
        )
        if delta <= 0:
            self._emit(
                1,
                "warning: engineered features did not improve the holdout "
                "score; consider using the original features",
            )
