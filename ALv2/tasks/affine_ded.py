# file: tasks/affine_ded.py

"""DED (program synthesis) wrapper."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from env.ded import DEDTask as AffineDEDGenerator
from tasks import BaseTask
from tasks.local_affine_dataset import SyntheticDEDData
from utils.concurrency import run_sync
from utils.sampling import sample_model
from utils.tokenization import build_prompt, strip_prompt_from_completion

logger = logging.getLogger(__name__)


class AffineDEDTask(BaseTask):
    def __init__(self, name: str = "ded", dataset=None) -> None:
        self.name = name
        self._dataset = dataset or SyntheticDEDData()
        self._generator = AffineDEDGenerator(dataset=self._dataset)
        self._challenge = None

    def reset(self) -> str:
        challenge = run_sync(self._generator.generate())
        self._challenge = challenge
        return challenge.prompt

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self._challenge is None:
            raise RuntimeError("Call reset() before step().")

        reward = float(run_sync(self._generator.evaluate(action, self._challenge)))
        info = {
            "sample": self._challenge.extra.get("sample"),
        }
        self._challenge = None
        return "", reward, True, info

    def evaluate(self, model, tokenizer, num_episodes: int = 100) -> float:
        rewards = []
        for _ in range(num_episodes):
            prompt = self.reset()
            completion = sample_model(
                model,
                tokenizer,
                build_prompt(self.name, prompt),
                max_new_tokens=512,
                temperature=0.4,
            )
            action = strip_prompt_from_completion(build_prompt(self.name, prompt), completion)
            _, reward, _, _ = self.step(action)
            rewards.append(reward)
        return sum(rewards) / max(1, len(rewards))

