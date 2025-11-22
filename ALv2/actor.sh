export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
torchrun --nproc_per_node=7 actor.py \
  --model_name caphe/Affine_top1 \
  --tasks_config configs/tasks.yaml \
  --replay_dir replay \
  --max_steps_per_episode 32 \
  --schedule round_robin