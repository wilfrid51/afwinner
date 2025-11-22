# file: tasks/__init__.py

"""Task protocol and shared utilities for multi-task RL."""

from __future__ import annotations

from typing import Protocol, Tuple, Dict, Any


class BaseTask(Protocol):
    """Minimal interface that every task implementation must follow."""

    name: str

    def reset(self) -> str:
        """Start a new episode and return the initial observation/prompt."""

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Apply an action coming from the policy and return ``(obs, reward, done, info)``.
        """

    def evaluate(self, model, tokenizer, num_episodes: int = 100) -> float:
        """Run offline evaluation using the provided model without updating weights."""

