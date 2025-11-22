import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data/agentevol")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We only care about these envs
ENV_TARGETS = {"webshop", "alfworld", "babyai", "sciworld", "textcraft"}

def detect_env(item_id: str) -> str | None:
    """Heuristic mapping from item_id to env name."""
    s = item_id.lower()
    if s.startswith("webshop"):
        return "webshop"
    if s.startswith("look_at_obj") or s.startswith("alfworld"):
        return "alfworld"
    if s.startswith("babyai"):
        return "babyai"
    if s.startswith("sciworld") or s.startswith("sci"):
        return "sciworld"
    if s.startswith("textcraft"):
        return "textcraft"
    return None

print("[INFO] Loading AgentGym/AgentTraj-L (train split)...")
ds = load_dataset("AgentGym/AgentTraj-L", split="train")

buffers: dict[str, list[list[dict]]] = {e: [] for e in ENV_TARGETS}

for idx, row in enumerate(ds):
    item_id = row["item_id"]
    env = detect_env(item_id)
    if env not in ENV_TARGETS:
        continue

    conv = row["conversations"]
    if not conv:
        continue

    traj: list[dict] = []
    last_obs: str | None = None

    for msg in conv:
        who = msg["from"]
        text = msg["value"]

        if who == "human":
            # Treat every human message as new observation text
            last_obs = text

        elif who in ("gpt", "assistant"):
            # Treat this as an action taken under the last observation
            if last_obs is None:
                continue
            action = text
            traj.append({
                "obs": last_obs,
                "action": action,
                "reward": 0.0,   # filled later
                "done": False,
            })

    if traj:
        # Give +1 reward on the last step only (success)
        traj[-1]["reward"] = 1.0
        traj[-1]["done"] = True
        buffers[env].append(traj)

    if (idx + 1) % 1000 == 0:
        print(f"[INFO] processed {idx + 1} rows")

print("[INFO] Writing JSONL files...")
for env, trajs in buffers.items():
    out_path = OUT_DIR / f"{env}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  {env}: {len(trajs)} trajectories -> {out_path}")

print("[DONE]")
