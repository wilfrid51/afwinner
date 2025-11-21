# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
# torchrun --nproc_per_node=7 actor.py \
#   --model_name caphe/Affine_top1 \
#   --replay_dir replay \
#   --tasks ABD \
#   --weights_dir output \
#   --batch_size 4 \
#   --group_size 4

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
torchrun --nproc_per_node=7 actor.py \
  --model_name caphe/Affine_top1 \
  --replay_dir replay \
  --tasks ABD,DED,SAT \
  --weights_dir output \
  --batch_size 4 \
  --group_size 4