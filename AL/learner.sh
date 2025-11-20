CUDA_VISIBLE_DEVICES=0 python learner.py \
  --model_name caphe/Affine_top1 \
  --output_dir output \
  --replay_dir replay \
  --batch_groups 2 \
  --group_size 4 \
  --bf16 \
  --max_seq_len 1600
