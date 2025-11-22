# file: scripts/run_actor.sh
#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="your-hf-user/strong-base-model"
CONFIG_PATH="configs/tasks.yaml"
REPLAY_DIR="replay"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6,7"} \
python actor.py \
  --model_name "$MODEL_NAME" \
  --tasks_config "$CONFIG_PATH" \
  --replay_dir "$REPLAY_DIR" \
  --max_steps_per_episode 32 \
  --schedule round_robin
