# file: utils/metrics.py

"""Lightweight metric helpers for RL logging."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict


class AverageMeter:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    def update(self, key: str, value: float, weight: int = 1) -> None:
        self.totals[key] += value * weight
        self.counts[key] += weight

    def get(self, key: str) -> float:
        if self.counts[key] == 0:
            return 0.0
        return self.totals[key] / self.counts[key]

    def summary(self) -> Dict[str, float]:
        return {k: self.get(k) for k in self.totals}

