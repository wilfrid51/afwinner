# file: tasks/affine_sat.py

"""Wrapper around the afferent SAT generator that exposes the BaseTask API."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from env.sat import SATTask as AffineSATGenerator
from tasks import BaseTask
from utils.sampling import sample_model
from utils.tokenization import build_prompt, strip_prompt_from_completion
from utils.concurrency import run_sync

logger = logging.getLogger(__name__)


class AffineSATTask(BaseTask):
    """Single-turn SAT task compatible with the unified RL interface."""

    def __init__(self, name: str = "sat") -> None:
        self.name = name
        self._generator = AffineSATGenerator()
        self._challenge = None
        self._last_prompt = ""

    # ------------------------------------------------------------------
    def reset(self) -> str:
        """Sample a fresh SAT challenge and return its textual prompt."""

        challenge = run_sync(self._generator.generate())
        self._challenge = challenge
        self._last_prompt = challenge.prompt
        return challenge.prompt

    # ------------------------------------------------------------------
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self._challenge is None:
            raise RuntimeError("Call reset() before step().")

        score = float(run_sync(self._generator.evaluate(action, self._challenge)))
        reward = 1.0 if score >= 0.999 else 0.0
        info = {
            "score": score,
            "solution": self._challenge.extra.get("solution"),
            "clauses": self._challenge.extra.get("clauses"),
        }
        # Single turn interaction; episode ends immediately.
        self._challenge = None
        return "", reward, True, info

    # ------------------------------------------------------------------
    def evaluate(self, model, tokenizer, num_episodes: int = 100) -> float:
        """Run single-turn evaluations and report success rate."""

        successes = 0
        for _ in range(num_episodes):
            prompt = self.reset()
            completion = sample_model(
                model,
                tokenizer,
                build_prompt(self.name, prompt),
                max_new_tokens=256,
                temperature=0.2,
            )
            action = strip_prompt_from_completion(build_prompt(self.name, prompt), completion)
            _, reward, _, _ = self.step(action)
            successes += 1 if reward > 0.5 else 0
        return successes / max(1, num_episodes)

