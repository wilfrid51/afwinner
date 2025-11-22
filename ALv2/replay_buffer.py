# file: replay_buffer.py

"""Task-aware replay buffer plus disk-backed helpers."""

from __future__ import annotations

import json
import random
import time
import uuid
from collections import deque, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional


@dataclass
class Transition:
    obs: str
    action: str
    reward: float
    next_obs: str
    done: bool
    task: str
    offline: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, seed: int = 0) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.random = random.Random(seed)
        self.task_stats: Dict[str, List[float]] = defaultdict(list)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)
        self.task_stats[transition.task].append(transition.reward)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for t in transitions:
            self.add(t)

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        return self.random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayDiskWriter:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_episode(self, transitions: List[Transition]) -> Path:
        timestamp = int(time.time() * 1000)
        file_path = self.directory / f"episode-{timestamp}-{uuid.uuid4().hex}.jsonl"
        with file_path.open("w", encoding="utf-8") as f:
            for t in transitions:
                f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        return file_path


class ReplayDirectoryReader:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._consumed: set[Path] = set()

    def fetch_new(self, max_files: Optional[int] = None) -> List[Transition]:
        transitions: List[Transition] = []
        files = sorted(self.directory.glob("episode-*.jsonl"))
        new_files = [f for f in files if f not in self._consumed]
        if max_files is not None:
            new_files = new_files[-max_files:]
        for path in new_files:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    transitions.append(Transition(**payload))
            self._consumed.add(path)
        return transitions

