# file: utils/logging.py

"""Central logging utilities."""

from __future__ import annotations

import logging
from collections import defaultdict


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class StatsTracker:
    def __init__(self) -> None:
        self.episode_counts = defaultdict(int)
        self.success_counts = defaultdict(int)

    def record(self, task: str, reward: float, done: bool) -> None:
        if done:
            self.episode_counts[task] += 1
            if reward > 0.5:
                self.success_counts[task] += 1

    def report(self) -> None:
        for task, n in self.episode_counts.items():
            s = self.success_counts[task]
            rate = s / n if n else 0.0
            logging.info("[STATS] %s: %d/%d = %.3f", task, s, n, rate)

