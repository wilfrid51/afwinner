# replay_buffer.py
import os
import json
import time
import uuid
from typing import Dict, Any, List, Optional


class LocalReplayBuffer:
    """
    Simple local filesystem-based replay buffer.

    Each item is written as a JSON file in `replay_dir`.
    Actors call `push(item)`.
    Learner calls `sample_batch_groups(batch_groups)` to get up to N items,
    then deletes the files after reading.

    This is Option 1: no Redis, just local files.
    """

    def __init__(self, replay_dir: str = "replay"):
        self.replay_dir = replay_dir
        os.makedirs(self.replay_dir, exist_ok=True)

    def _make_paths(self) -> (str, str):
        """
        Create a unique filename stem and return (temp_path, final_path).
        We write to temp first, then rename to final for atomicity.
        """
        stem = f"{int(time.time() * 1e9)}_{os.getpid()}_{uuid.uuid4().hex}"
        temp_path = os.path.join(self.replay_dir, stem + ".tmp")
        final_path = os.path.join(self.replay_dir, stem + ".json")
        return temp_path, final_path

    def push(self, item: Dict[str, Any]) -> None:
        """
        Append one trajectory group to the buffer.
        """
        temp_path, final_path = self._make_paths()
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(item, f)
            os.replace(temp_path, final_path)  # atomic rename
        except Exception:
            # Best-effort cleanup
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            # We silently drop this item on error

    def _list_json_files(self) -> List[str]:
        files = []
        try:
            for fname in os.listdir(self.replay_dir):
                if fname.endswith(".json"):
                    files.append(fname)
        except FileNotFoundError:
            os.makedirs(self.replay_dir, exist_ok=True)
        files.sort()  # approximate FIFO by filename (time-based prefix)
        return files

    def sample_batch_groups(self, batch_groups: int, timeout: int = 1) -> List[Dict[str, Any]]:
        """
        Pop up to `batch_groups` trajectory groups from the buffer.

        Blocks up to `timeout` seconds waiting for at least one item.
        After first item is read, it tries to read up to `batch_groups-1`
        more items without blocking.

        Returns:
            List[dict] with length in [0, batch_groups].
        """
        deadline = time.time() + timeout
        batch: List[Dict[str, Any]] = []

        # Wait until we see at least one file or timeout
        while True:
            files = self._list_json_files()
            if files or time.time() >= deadline:
                break
            time.sleep(0.05)

        if not files:
            return batch  # no data

        # Take up to batch_groups files
        to_take = files[:batch_groups]

        for fname in to_take:
            path = os.path.join(self.replay_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    item = json.load(f)
                batch.append(item)
            except Exception:
                # If file is corrupt / mid-write, skip it
                pass
            finally:
                # Best-effort removal
                try:
                    os.remove(path)
                except Exception:
                    pass

        return batch


# Backwards-compatible name if old code imports RedisReplayBuffer
RedisReplayBuffer = LocalReplayBuffer
