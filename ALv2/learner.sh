CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python3 learner.py \
  --model_name caphe/Affine_top1 \
  --replay_dir replay \
  --output_dir output \
  --batch_size 2 \
  --total_steps 10000 \
  --lambda_bc 0.1 \
  --min_buffer 512 \
  --max_seq_len 1536 \
  --gradient_checkpointing