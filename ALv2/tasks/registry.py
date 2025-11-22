# file: tasks/registry.py

"""Task factory that instantiates all supported environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from tasks import BaseTask
from tasks.affine_abd import AffineABDTask
from tasks.affine_ded import AffineDEDTask
from tasks.affine_sat import AffineSATTask
from tasks.agentevol_offline import AgentEvolOfflineTask
from tasks.agentevol_online import AgentEvolOnlineTask

DEFAULT_TASKS_CONFIG = {
    "affine": {
        "sat": {},
        "abd": {},
        "ded": {},
    },
    "agentevol_offline": {
        "data_root": "data/agentevol",
        "envs": ["webshop", "alfworld", "babyai", "sciworld", "textcraft"],
    },
    "agentevol_online": {
        "base_url": "http://localhost:8000",
        "envs": [],
    },
}


def _load_yaml(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return DEFAULT_TASKS_CONFIG
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or DEFAULT_TASKS_CONFIG


def build_tasks(config: Dict[str, Any] | str | Path | None = None) -> Dict[str, BaseTask]:
    """Instantiate all tasks defined in the YAML/dict config."""

    raw_cfg = _load_yaml(config) if isinstance(config, (str, Path, type(None))) else config
    if raw_cfg is None:
        raw_cfg = DEFAULT_TASKS_CONFIG

    tasks: Dict[str, BaseTask] = {}

    # Affine tasks ------------------------------------------------------
    affine_cfg = raw_cfg.get("affine", {})
    if affine_cfg.get("sat", {}) is not None:
        tasks["sat"] = AffineSATTask()
    if affine_cfg.get("abd", {}) is not None:
        tasks["abd"] = AffineABDTask()
    if affine_cfg.get("ded", {}) is not None:
        tasks["ded"] = AffineDEDTask()

    # Offline AgentGym --------------------------------------------------
    offline_cfg = raw_cfg.get("agentevol_offline", {})
    data_root = Path(offline_cfg.get("data_root", "data/agentevol"))
    for env_name in offline_cfg.get("envs", []):
        path = data_root / f"{env_name}.jsonl"
        tasks[f"{env_name}_offline"] = AgentEvolOfflineTask(env_name, path)

    # Online AgentGym ---------------------------------------------------
    online_cfg = raw_cfg.get("agentevol_online", {})
    base_url = online_cfg.get("base_url", "http://localhost:8000")
    for env_name in online_cfg.get("envs", []):
        tasks[f"{env_name}_online"] = AgentEvolOnlineTask(env_name, base_url)

    if not tasks:
        raise RuntimeError("No tasks instantiated; check configuration")

    return tasks

