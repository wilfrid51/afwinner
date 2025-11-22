# file: tasks/agentevol_online.py

"""HTTP client + BaseTask wrapper for live AgentGym environments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import requests

from tasks import BaseTask
from utils.sampling import sample_model
from utils.tokenization import build_prompt, strip_prompt_from_completion

logger = logging.getLogger(__name__)


@dataclass
class AgentGymEnvClient:
    base_url: str
    env_name: str
    timeout: float = 10.0

    def reset(self) -> str:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"env_name": self.env_name},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("observation", "")

    def act(self, action: str) -> Tuple[str, float, bool, Dict]:
        resp = requests.post(
            f"{self.base_url}/step",
            json={"env_name": self.env_name, "action": action},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return (
            payload.get("observation", ""),
            float(payload.get("reward", 0.0)),
            bool(payload.get("done", False)),
            payload.get("info", {}) or {},
        )


class AgentEvolOnlineTask(BaseTask):
    offline = False

    def __init__(self, env_name: str, base_url: str, name_suffix: str = "online") -> None:
        self.env_name = env_name
        self.name = f"{env_name}_{name_suffix}"
        self._client = AgentGymEnvClient(base_url=base_url, env_name=env_name)

    def reset(self) -> str:
        return self._client.reset()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        obs, reward, done, info = self._client.act(action)
        info.setdefault("env_name", self.env_name)
        info.setdefault("offline", False)
        return obs, reward, done, info

    def evaluate(self, model, tokenizer, num_episodes: int = 10, max_steps: int = 32) -> float:  # noqa: D401,E501
        """Roll out online episodes and measure average reward."""

        total_reward = 0.0
        for _ in range(num_episodes):
            obs = self.reset()
            for _ in range(max_steps):
                prompt = build_prompt(self.name, obs)
                completion = sample_model(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=128,
                    temperature=0.7,
                )
                action = strip_prompt_from_completion(prompt, completion)
                obs, reward, done, _ = self.step(action)
                total_reward += reward
                if done:
                    break
        return total_reward / max(1, num_episodes)

