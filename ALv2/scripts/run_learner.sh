# file: scripts/run_learner.sh
#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="your-hf-user/strong-base-model"
REPLAY_DIR="replay"
OUTPUT_DIR="output"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python learner.py \
  --model_name "$MODEL_NAME" \
  --replay_dir "$REPLAY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --total_steps 10000 \
  --batch_size 64
