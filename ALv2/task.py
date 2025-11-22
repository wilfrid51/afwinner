from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


logger = logging.getLogger(__name__)


class AgentEvolTask:
    """
    Offline task wrapper that replays pre-recorded expert trajectories
    (e.g. AgentGym datasets such as WebShop, ALFWorld, BabyAI, SciWorld, TextCraft)
    inside a standard RL training loop.
    """

    def __init__(
        self,
        env_name: str,
        data_path: Optional[str | Path] = None,
        *,
        trajectories: Optional[Sequence[Sequence[Dict[str, Any]]]] = None,
        eval_split: float = 0.1,
        seed: Optional[int] = None,
        warn_on_action_mismatch: bool = False,
    ) -> None:
        """
        Args:
            env_name: Name of the AgentGym task or underlying environment.
            data_path: Path to JSON/JSONL file(s) or directory containing trajectories.
            trajectories: Optional in-memory trajectories (used instead of reading from disk).
            eval_split: Fraction of trajectories held out for evaluation.
            seed: Random seed used for sampling trajectories.
            warn_on_action_mismatch: Whether to log discrepancies between agent and expert actions.
        """
        self.env_name = env_name
        self.warn_on_action_mismatch = warn_on_action_mismatch
        self.rng = random.Random(seed)

        loaded = (
            [list(traj) for traj in trajectories]
            if trajectories is not None
            else self._load_from_path(data_path)
        )
        if not loaded:
            raise ValueError("AgentEvolTask requires at least one trajectory.")

        split_idx = int(len(loaded) * (1 - eval_split))
        if split_idx <= 0:
            split_idx = len(loaded)

        self._train_trajs = loaded[:split_idx]
        self._eval_trajs = loaded[split_idx:] or loaded[: min(len(loaded), 16)]

        self._current_traj: Optional[List[Dict[str, Any]]] = None
        self._current_index: int = 0
        self._episode_done: bool = True
        self._last_observation: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # RL-style API                                                       #
    # ------------------------------------------------------------------ #
    def reset(self) -> Any:
        """Start a new offline episode by sampling a stored trajectory."""
        if not self._train_trajs:
            raise RuntimeError("No training trajectories loaded.")

        self._current_traj = self.rng.choice(self._train_trajs)
        self._current_index = 0
        self._episode_done = False

        first_step = self._current_traj[0]
        self._last_observation = first_step.get("obs")
        return self._last_observation

    def step(self, action: Optional[str] = None):
        """
        Replay the next expert transition, ignoring the agent's action for
        state progression (offline behavioral cloning style).
        """
        if self._current_traj is None:
            raise RuntimeError("Call reset() before step().")

        if self._episode_done:
            return self._last_observation, 0.0, True, {
                "env": self.env_name,
                "info": "Episode already finished; call reset().",
            }

        if self._current_index >= len(self._current_traj):
            self._episode_done = True
            return self._last_observation, 0.0, True, {
                "env": self.env_name,
                "info": "Trajectory exhausted; call reset().",
            }

        step_data = self._current_traj[self._current_index]
        expert_action = step_data.get("action")
        reward = float(step_data.get("reward", 0.0))
        done = bool(step_data.get("done", False))

        if (
            self.warn_on_action_mismatch
            and action is not None
            and expert_action is not None
            and action != expert_action
        ):
            logger.warning(
                "[AgentEvolTask] Action mismatch at step %d: agent=%s vs expert=%s",
                self._current_index,
                action,
                expert_action,
            )

        next_obs = step_data.get("next_obs")
        if next_obs is None and self._current_index + 1 < len(self._current_traj):
            next_obs = self._current_traj[self._current_index + 1].get("obs")
        if next_obs is None:
            next_obs = step_data.get("obs")

        info = {
            "env": self.env_name,
            "expert_action": expert_action,
            "agent_action": action,
            "step_index": self._current_index,
            "trajectory_length": len(self._current_traj),
            "matched_action": (action == expert_action) if action is not None else None,
        }

        self._current_index += 1
        self._last_observation = next_obs
        if done or self._current_index >= len(self._current_traj):
            self._episode_done = True

        return next_obs, reward, done, info

    # ------------------------------------------------------------------ #
    # Evaluation helper                                                  #
    # ------------------------------------------------------------------ #
    def evaluate(self, model: Any, num_episodes: int = 10) -> float:
        """
        Simple offline evaluation: measure how often the provided `model`
        reproduces the expert actions on held-out trajectories.
        """
        if not self._eval_trajs:
            logger.warning("No dedicated eval split; falling back to training set.")
            self._eval_trajs = self._train_trajs

        episodes = min(num_episodes, len(self._eval_trajs))
        if episodes == 0:
            return 0.0

        successes = 0
        for traj in self._rng_sample(self._eval_trajs, episodes):
            if self._match_trajectory(model, traj):
                successes += 1

        return successes / episodes

    # ------------------------------------------------------------------ #
    # Internal utilities                                                 #
    # ------------------------------------------------------------------ #
    def _load_from_path(self, data_path: Optional[str | Path]) -> List[List[Dict[str, Any]]]:
        if data_path is None:
            return []

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory path not found: {path}")

        trajectories: List[List[Dict[str, Any]]] = []

        def _maybe_add(traj: Any) -> None:
            if not traj:
                return
            if isinstance(traj, dict):
                traj = traj.get("steps") or traj.get("trajectory")
            if isinstance(traj, list):
                cleaned = [dict(step) for step in traj if isinstance(step, dict)]
                if cleaned:
                    trajectories.append(cleaned)

        if path.is_file():
            trajectories.extend(self._read_file(path))
        else:
            for file in sorted(path.glob("**/*.json")):
                trajectories.extend(self._read_file(file))
            for file in sorted(path.glob("**/*.jsonl")):
                trajectories.extend(self._read_file(file))

        return trajectories

    def _read_file(self, file_path: Path) -> List[List[Dict[str, Any]]]:
        data: List[List[Dict[str, Any]]] = []
        if file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        if isinstance(payload, list):
                            data.append([dict(step) for step in payload])
                        elif isinstance(payload, dict) and "steps" in payload:
                            data.append([dict(step) for step in payload["steps"]])
                    except json.JSONDecodeError as exc:
                        logger.warning("Failed to parse %s: %s", file_path, exc)
        else:
            with file_path.open("r", encoding="utf-8") as fp:
                try:
                    payload = json.load(fp)
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse %s: %s", file_path, exc)
                    return data
            if isinstance(payload, list):
                data.extend(
                    [dict(step) for step in traj]
                    for traj in payload
                    if isinstance(traj, list)
                )
            elif isinstance(payload, dict):
                steps = payload.get("steps") or payload.get("trajectory")
                if isinstance(steps, list):
                    data.append([dict(step) for step in steps])
        return data

    def _match_trajectory(self, model: Any, trajectory: Sequence[Dict[str, Any]]) -> bool:
        """Return True if `model` reproduces expert actions for the trajectory."""
        observation = trajectory[0].get("obs")
        for step in trajectory:
            expert_action = step.get("action")
            predicted = self._call_model(model, observation)
            if predicted != expert_action:
                return False
            observation = step.get("next_obs") or observation
            if step.get("done"):
                break
        return True

    def _call_model(self, model: Any, observation: Any) -> Optional[str]:
        if hasattr(model, "act") and callable(model.act):  # type: ignore[attr-defined]
            return model.act(observation, env=self.env_name)  # type: ignore[call-arg]
        if callable(model):
            return model(observation)
        if hasattr(model, "predict") and callable(model.predict):
            return model.predict(observation)
        return None

    def _rng_sample(self, data: Sequence[Any], k: int) -> Iterable[Any]:
        """Sample without replacement (falls back to random choices if needed)."""
        if k >= len(data):
            yield from data
            return
        indices = list(range(len(data)))
        self.rng.shuffle(indices)
        for idx in indices[:k]:
            yield data[idx]
