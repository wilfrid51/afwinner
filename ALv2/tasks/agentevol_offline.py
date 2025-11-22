# file: tasks/agentevol_offline.py

"""Offline AgentGym wrapper built around JSONL trajectory dumps."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from tasks import BaseTask
from utils.sampling import sample_model
from utils.tokenization import build_prompt, strip_prompt_from_completion


@dataclass
class OfflineStep:
    obs: str
    action: str
    reward: float
    done: bool


class AgentEvolOfflineTask(BaseTask):
    offline: bool = True

    def __init__(
        self,
        env_name: str,
        data_path: Path | str,
        seed: int = 0,
        name_suffix: str = "offline",
    ) -> None:
        self.env_name = env_name
        self.name = f"{env_name}_{name_suffix}"
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing offline data at {self.data_path}")
        self._rng = random.Random(seed)
        self._trajectories: List[List[OfflineStep]] = self._load_file()
        self._traj: List[OfflineStep] | None = None
        self._idx = 0

    # ------------------------------------------------------------------
    def _load_file(self) -> List[List[OfflineStep]]:
        data: List[List[OfflineStep]] = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    steps = row.get("trajectory") or row.get("steps") or []
                elif isinstance(row, list):
                    steps = row
                else:
                    continue
                traj: List[OfflineStep] = []
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    traj.append(
                        OfflineStep(
                            obs=step.get("obs", ""),
                            action=step.get("action", ""),
                            reward=float(step.get("reward", 0.0)),
                            done=bool(step.get("done", False)),
                        )
                    )
                if traj:
                    data.append(traj)
        if not data:
            raise RuntimeError(f"No trajectories decoded from {self.data_path}")
        return data

    # ------------------------------------------------------------------
    def reset(self) -> str:
        self._traj = self._rng.choice(self._trajectories)
        self._idx = 0
        return self._traj[0].obs

    # ------------------------------------------------------------------
    def step(self, action: str | None) -> Tuple[str, float, bool, Dict]:  # noqa: ARG002
        if self._traj is None:
            raise RuntimeError("Call reset() first")
        if self._idx >= len(self._traj):
            return "", 0.0, True, {"offline": True, "env_name": self.env_name}

        step = self._traj[self._idx]
        self._idx += 1
        next_obs = self._traj[self._idx].obs if self._idx < len(self._traj) else ""
        info = {
            "expert_action": step.action,
            "offline": True,
            "env_name": self.env_name,
        }
        return next_obs, step.reward, step.done or self._idx >= len(self._traj), info

    # ------------------------------------------------------------------
    def evaluate(self, model, tokenizer, num_episodes: int = 100) -> float:
        """Compute imitation accuracy over held-out trajectories."""

        total_pairs = 0
        correct = 0
        for _ in range(num_episodes):
            traj = self._rng.choice(self._trajectories)
            for step in traj:
                prompt = build_prompt(self.name, step.obs)
                completion = sample_model(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=128,
                    temperature=0.8,
                )
                action = strip_prompt_from_completion(prompt, completion)
                total_pairs += 1
                if action.strip() == step.action.strip():
                    correct += 1
        return correct / max(1, total_pairs)

