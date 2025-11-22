# file: tasks/affine_abd.py

"""ABD wrapper translating Affine's async API into BaseTask."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from env.abd import ABDTask as AffineABDGenerator
from tasks import BaseTask
from tasks.local_affine_dataset import SyntheticABDData
from utils.concurrency import run_sync
from utils.sampling import sample_model
from utils.tokenization import build_prompt, strip_prompt_from_completion

logger = logging.getLogger(__name__)


class AffineABDTask(BaseTask):
    """Reverse-input task solved in a single turn."""

    def __init__(self, name: str = "abd", dataset=None) -> None:
        self.name = name
        self._dataset = dataset or SyntheticABDData()
        self._generator = AffineABDGenerator(dataset=self._dataset)
        self._challenge = None

    def reset(self) -> str:
        challenge = run_sync(self._generator.generate())
        self._challenge = challenge
        return challenge.prompt

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self._challenge is None:
            raise RuntimeError("Call reset() first")

        reward = float(run_sync(self._generator.evaluate(action, self._challenge)))
        info = {
            "program": self._challenge.extra.get("program"),
            "expected_output": self._challenge.extra.get("expected_output"),
        }
        self._challenge = None
        return "", reward, True, info

    def evaluate(self, model, tokenizer, num_episodes: int = 100) -> float:
        successes = 0
        for _ in range(num_episodes):
            prompt = self.reset()
            completion = sample_model(
                model,
                tokenizer,
                build_prompt(self.name, prompt),
                temperature=0.3,
            )
            action = strip_prompt_from_completion(build_prompt(self.name, prompt), completion)
            _, reward, _, _ = self.step(action)
            successes += 1 if reward > 0.5 else 0
        return successes / max(1, num_episodes)

