from __future__ import annotations

import csv
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineProfiler:
    """Per-call timing collector for every named pipeline component.

    Times are stored in milliseconds. Use start()/stop() around code blocks,
    or record() to log a pre-computed value. All samples are kept in memory
    so you can export stats and individual measurements after a session.
    """

    def __init__(self):
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._active: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def start(self, name: str) -> None:
        self._active[name] = time.perf_counter()

    def stop(self, name: str) -> float | None:
        if name not in self._active:
            return None
        dt_ms = (time.perf_counter() - self._active.pop(name)) * 1000.0
        self._samples[name].append(dt_ms)
        return dt_ms

    def record(self, name: str, ms: float) -> None:
        self._samples[name].append(float(ms))

    def get_samples(self, name: str) -> list[float]:
        return list(self._samples.get(name, []))

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, vals in self._samples.items():
            if not vals:
                continue
            s = sorted(vals)
            n = len(s)
            out[name] = {
                "count": n,
                "mean_ms": statistics.mean(s),
                "min_ms": s[0],
                "max_ms": s[-1],
                "p50_ms": s[n // 2],
                "p95_ms": s[min(int(n * 0.95), n - 1)],
            }
        return out

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def save_samples_csv(self, path: Path) -> None:
        """Wide CSV: one column per component, one row per measurement.

        Different components have different call counts; empty cells where
        one component has fewer samples than the longest column.
        """
        if not self._samples:
            logger.warning("Nincs időzítési adat a CSV exporthoz.")
            return
        keys = sorted(self._samples.keys())
        max_rows = max(len(v) for v in self._samples.values())
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"{k}_ms" for k in keys])
            for i in range(max_rows):
                writer.writerow([
                    f"{self._samples[k][i]:.4f}" if i < len(self._samples[k]) else ""
                    for k in keys
                ])
        logger.info("Komponens minták mentve: %s (%d sor)", path, max_rows)

    def save_summary_csv(self, path: Path) -> None:
        """One row per component with mean/min/max/p50/p95."""
        stats = self.get_stats()
        if not stats:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["component", "count", "mean_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, s in sorted(stats.items()):
                writer.writerow({
                    "component": name,
                    "count": s["count"],
                    "mean_ms": f"{s['mean_ms']:.4f}",
                    "min_ms": f"{s['min_ms']:.4f}",
                    "max_ms": f"{s['max_ms']:.4f}",
                    "p50_ms": f"{s['p50_ms']:.4f}",
                    "p95_ms": f"{s['p95_ms']:.4f}",
                })
        logger.info("Összesítő mentve: %s", path)

    def report(self) -> None:
        stats = self.get_stats()
        if not stats:
            logger.info("Nincs időzítési adat.")
            return
        w = 82
        logger.info("=" * w)
        logger.info("  PIPELINE IDŐZÍTÉSI ÖSSZEFOGLALÓ  (ms)")
        logger.info("=" * w)
        logger.info(
            "  %-26s %5s %9s %8s %8s %8s %8s",
            "Komponens", "db", "átlag", "min", "max", "p50", "p95",
        )
        logger.info("-" * w)
        for name, s in sorted(stats.items()):
            logger.info(
                "  %-26s %5d %9.2f %8.2f %8.2f %8.2f %8.2f",
                name, s["count"],
                s["mean_ms"], s["min_ms"], s["max_ms"], s["p50_ms"], s["p95_ms"],
            )
        logger.info("=" * w)
