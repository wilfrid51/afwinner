# train_rl_multitask.py
#
# Multi-task RL (GRPO-style) over your Affine tasks:
#   - SATTask (sat.py)
#   - ABDTask (abd.py)
#   - DEDTask (ded.py)
#
# One shared model, LoRA adapters, DDP over multiple GPUs.

import os
# Disable tokenizer parallelism to avoid deadlocks when using DDP/multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Optimize CUDA memory allocation to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import random
import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Your task/env code
from sat import SATTask
from abd import ABDTask
from ded import DEDTask
from dataset import R2Dataset


# ----------------- config ----------------- #

@dataclass
class RLConfig:
    # ---- model / generation ----
    model_name_or_path: str = "5Fafur/Affine-grab"  # <<< SET THIS
    # model_name_or_path: str = "trongg/Affine_robertoCalories"  # <<< SET THIS
    max_prompt_len: int = 512
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0

    # ---- optimization ----
    lr: float = 5e-5
    weight_decay: float = 0.01
    total_steps: int = 5000
    global_batch_size: int = 8  # total across all GPUs

    # ---- RL / GRPO ----
    kl_coef: float = 0.05
    adv_norm_eps: float = 1e-8
    weight_clip: float = 5.0

    # ---- LoRA ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ---- data ----
    dataset_name: str = "satpalsr/rl-python"  # same as env.py

    # ---- logging / ckpt ----
    log_every: int = 10
    save_every: int = 1000
    output_dir: str = "./rl_lora_ckpts"

    # ---- misc ----
    seed: int = 42


# ----------------- DDP utils ----------------- #

def setup_ddp():
    """Init torch.distributed from torchrun env vars."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Run with torchrun, e.g.: "
            "torchrun --nproc_per_node=8 train_rl_multitask.py --model-name YOUR_MODEL"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int):
    s = seed + rank
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------- LoRA setup ----------------- #

def add_lora(model: AutoModelForCausalLM, cfg: RLConfig) -> AutoModelForCausalLM:
    """
    Attach LoRA adapters to attention/MLP blocks.
    You might need to adjust target_modules depending on your HF model.
    For llama/qwen-style models these names are typical.
    """
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)

    # Freeze base weights, train only LoRA params
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    return model


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


# ----------------- logprob helpers ----------------- #

def compute_logprobs_for_responses(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute average logprob over *response* tokens for each sample.
    Memory-efficient: processes logits in chunks to avoid large log_softmax.

    input_ids:      [B, T]
    attention_mask: [B, T]
    prompt_lengths: [B] (token length of the prompt *before* generation)
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]
        del out  # Free memory immediately

    # Process in chunks to avoid large log_softmax tensor
    B, T, V = logits.shape
    target_ids = input_ids[:, 1:]  # [B, T-1]
    chunk_size = 32  # Process 32 tokens at a time
    
    token_logprobs_list = []
    for i in range(0, T - 1, chunk_size):
        end_idx = min(i + chunk_size, T - 1)
        logits_chunk = logits[:, i:end_idx+1, :]  # [B, chunk, V]
        target_chunk = target_ids[:, i:end_idx]  # [B, chunk]
        
        # Compute log_softmax only for this chunk
        log_probs_chunk = torch.log_softmax(logits_chunk[:, :-1, :], dim=-1)  # [B, chunk-1, V]
        token_logprobs_chunk = log_probs_chunk.gather(
            -1, target_chunk.unsqueeze(-1)
        ).squeeze(-1)  # [B, chunk-1]
        token_logprobs_list.append(token_logprobs_chunk)
        
        del logits_chunk, log_probs_chunk, token_logprobs_chunk
    
    del logits  # Free large tensor
    
    token_logprobs = torch.cat(token_logprobs_list, dim=1)  # [B, T-1]

    B, Tm1 = token_logprobs.shape
    device = token_logprobs.device

    idxs = torch.arange(Tm1, device=device).unsqueeze(0).expand(B, -1)
    start = (prompt_lengths - 1).clamp(min=0).unsqueeze(1)      # [B,1]

    # response tokens: positions >= prompt length (after shift)
    resp_mask = (idxs >= start).float() * attention_mask[:, 1:].float()

    resp_logprob_sum = (token_logprobs * resp_mask).sum(dim=1)  # [B]
    resp_len = resp_mask.sum(dim=1).clamp(min=1.0)              # [B]

    avg_logprob = resp_logprob_sum / resp_len
    return avg_logprob


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    cfg: RLConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Generate responses for a batch of prompts with the current policy.
    Returns:
      - all_input_ids:      [B, T]
      - all_attention_mask: [B, T]
      - prompt_lengths:     [B]
      - responses_text:     list[str]
    """
    model.eval()

    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_prompt_len,
        return_attention_mask=True,
        return_length=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_lengths = enc["length"].to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )

    all_input_ids = gen_ids
    all_attention_mask = (gen_ids != tokenizer.pad_token_id).long()

    responses = []
    for i, ids in enumerate(gen_ids):
        start = int(prompt_lengths[i].item())
        resp_ids = ids[start:]
        if tokenizer.eos_token_id is not None:
            eos_pos = (resp_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if len(eos_pos) > 0:
                resp_ids = resp_ids[: eos_pos[0, 0]]
        text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        responses.append(text)

    return all_input_ids, all_attention_mask, prompt_lengths, responses


# ----------------- GRPO weights ----------------- #

def compute_grpo_weights(
    rewards: torch.Tensor,
    logprob_old: torch.Tensor,
    logprob_ref: torch.Tensor,
    cfg: RLConfig,
) -> torch.Tensor:
    """
    GRPO-style weighting:

      adv = reward - kl_coef * KL_approx
      KL_approx ~ (logprob_old - logprob_ref)

      weights = normalized, clipped advantage
    """
    kl = (logprob_old - logprob_ref)         # [B]
    adv = rewards - cfg.kl_coef * kl        # [B]

    mean = adv.mean()
    std = adv.std(unbiased=False)
    norm_adv = (adv - mean) / (std + cfg.adv_norm_eps)

    weights = torch.clamp(norm_adv, -cfg.weight_clip, cfg.weight_clip)
    return weights.detach()


# ----------------- checkpoint save/load ----------------- #

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint directory based on step number."""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("step-"):
            try:
                step = int(item.split("-")[1])
                ckpt_path = os.path.join(output_dir, item)
                if os.path.exists(os.path.join(ckpt_path, "training_state.pt")):
                    checkpoints.append((step, ckpt_path))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
    
    # Sort by step number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    ckpt_dir: str,
    rank: int,
):
    """Save full training checkpoint including model, optimizer, scheduler, and step."""
    if is_main(rank):
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save model (PEFT/LoRA will save adapter weights)
        model.save_pretrained(ckpt_dir)
        
        # Save optimizer and scheduler state
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
            },
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        
        # Save config/metadata
        metadata = {
            "step": step,
            "checkpoint_dir": ckpt_dir,
        }
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[rank {rank}] Saved checkpoint at step {step} to {ckpt_dir}")


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    ckpt_dir: str,
    device: torch.device,
    rank: int,
) -> int:
    """
    Load checkpoint and return the step number.
    Returns step number if successful, 0 if checkpoint doesn't exist.
    """
    training_state_path = os.path.join(ckpt_dir, "training_state.pt")
    metadata_path = os.path.join(ckpt_dir, "metadata.json")
    
    if not os.path.exists(training_state_path):
        if is_main(rank):
            print(f"[rank {rank}] No checkpoint found at {ckpt_dir}, starting from scratch")
        return 0
    
    if is_main(rank):
        print(f"[rank {rank}] Loading checkpoint from {ckpt_dir}")
    
    # Load model weights (PEFT/LoRA adapter weights)
    try:
        # Get the actual model (unwrap DDP if needed)
        actual_model = model.module if hasattr(model, 'module') else model
        
        # Check if this is a PEFT checkpoint
        if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
            # Load PEFT adapter weights
            if is_main(rank):
                print(f"[rank {rank}] Loading PEFT adapter weights...")
            
            # Use PEFT's load_adapter method if available
            if hasattr(actual_model, 'load_adapter'):
                actual_model.load_adapter(ckpt_dir)
            else:
                # Fallback: load adapter weights manually
                from peft import PeftModel
                # Get base model
                if hasattr(actual_model, 'get_base_model'):
                    base_model = actual_model.get_base_model()
                else:
                    base_model = actual_model
                # Load adapter
                loaded_model = PeftModel.from_pretrained(base_model, ckpt_dir)
                # Replace the model
                if hasattr(model, 'module'):
                    model.module = loaded_model
                else:
                    model = loaded_model
        else:
            # Regular PyTorch checkpoint (full model)
            if is_main(rank):
                print(f"[rank {rank}] Loading full model weights...")
            checkpoint_path = os.path.join(ckpt_dir, "pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(ckpt_dir, "model.safetensors")
            if os.path.exists(checkpoint_path):
                # Try loading as state dict
                try:
                    state_dict = torch.load(checkpoint_path, map_location=device)
                    actual_model.load_state_dict(state_dict, strict=False)
                except:
                    # If that fails, try using from_pretrained
                    if hasattr(model, 'module'):
                        model.module = AutoModelForCausalLM.from_pretrained(ckpt_dir)
                    else:
                        model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    except Exception as e:
        if is_main(rank):
            print(f"[rank {rank}] Warning: Could not load model weights: {e}")
            print(f"[rank {rank}] Continuing with existing model weights...")
    
    # Load optimizer and scheduler state
    try:
        checkpoint = torch.load(training_state_path, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = checkpoint.get("step", 0)
        
        if is_main(rank):
            print(f"[rank {rank}] Loaded checkpoint: step={step}")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    print(f"[rank {rank}] Checkpoint metadata: {metadata}")
        
        return step
    except Exception as e:
        if is_main(rank):
            print(f"[rank {rank}] Error loading training state: {e}")
        return 0


# ----------------- async helpers for your tasks ----------------- #

async def async_generate_challenges(tasks: List[object], task_ids: List[int]):
    """Run task.generate() concurrently for a batch."""
    coros = [tasks[t_idx].generate() for t_idx in task_ids]
    return await asyncio.gather(*coros)


async def async_evaluate_challenges(
    tasks: List[object],
    task_ids: List[int],
    responses: List[str],
    challenges: List[object],
):
    """Run task.evaluate() concurrently for a batch."""
    coros = [
        tasks[t_idx].evaluate(resp, ch)
        for t_idx, resp, ch in zip(task_ids, responses, challenges)
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)
    scores: List[float] = []
    for r in results:
        if isinstance(r, Exception):
            # If task crashes, just give 0 reward
            scores.append(0.0)
        else:
            scores.append(float(r))
    return scores


# ----------------- main training loop (async) ----------------- #

async def train(rank: int, world_size: int, local_rank: int, cfg: RLConfig, args):
    device = torch.device(f"cuda:{local_rank}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Set left padding for decoder-only models
    tokenizer.padding_side = 'left'

    # ---- dtype ----
    # Force bfloat16 for memory reduction (saves ~50% memory vs float32)
    dtype = torch.bfloat16
    if not torch.cuda.is_bf16_supported():
        if is_main(rank):
            print(f"[rank {rank}] Warning: bfloat16 not supported, falling back to float16")
        dtype = torch.float16

    # ---- base policy + reference (frozen) ----
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        dtype=dtype,
        device_map=None,
        # torch_dtype=dtype,  # Explicitly set torch_dtype
    )
    # Load ref_policy on GPU with 8-bit quantization to save memory
    # This is much faster than CPU transfers and saves ~50% memory
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        ref_policy = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            quantization_config=quantization_config,
            device_map={"": device},  # Load directly to GPU
        )
    except Exception:
        # Fallback to regular loading if quantization fails
        ref_policy = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            dtype=dtype,
            device_map=None,
        )
        ref_policy.to(device)

    # Explicitly convert to bfloat16 to ensure all weights are in bfloat16
    policy = policy.to(dtype=dtype).to(device)
    
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    # ---- attach LoRA to policy only ----
    policy = add_lora(policy, cfg)
    # Ensure LoRA adapters are also in bfloat16
    policy = policy.to(dtype=dtype).to(device)
    
    # Enable cuDNN benchmarking for faster convolutions/RNNs (if any)
    torch.backends.cudnn.benchmark = True
    
    # Compile model for faster inference (PyTorch 2.0+)
    # This can provide 20-30% speedup
    # Note: Compile before DDP wrapping, and skip if quantization is used
    try:
        if hasattr(torch, 'compile'):
            # Only compile if not quantized (quantized models have different structure)
            if not hasattr(ref_policy, 'hf_quantizer'):  # Check if quantized
                ref_policy = torch.compile(ref_policy, mode="reduce-overhead")
            # Policy compilation will happen after DDP (DDP doesn't support compiled models well)
    except Exception:
        pass  # Fallback if compilation fails

    ddp_policy = DDP(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    optim_params = get_trainable_params(ddp_policy)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.total_steps,
    )
    
    # GradScaler for mixed precision training (bfloat16 forward, float32 backward)
    # Note: bfloat16 doesn't need scaling, but GradScaler helps with stability
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))  # Only scale for float16

    # ---- your tasks + shared R2Dataset ----
    shared_dataset = R2Dataset(dataset_name=cfg.dataset_name)
    tasks: List[object] = [
        SATTask(),                      # no dataset needed
        ABDTask(dataset=shared_dataset),
        DEDTask(dataset=shared_dataset),
    ]
    task_names = ["sat", "abd", "ded"]
    num_tasks = len(tasks)

    per_device_batch = max(1, cfg.global_batch_size // world_size)

    if is_main(rank):
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"[rank {rank}] world_size={world_size}, per_device_batch={per_device_batch}")
        print(f"[rank {rank}] trainable params (LoRA only): "
              f"{sum(p.numel() for p in optim_params)}")

    # ----- load checkpoint if provided -----
    start_step = 0
    if hasattr(args, 'resume_from') and args.resume_from is not None:
        # Load checkpoint (model, optimizer, scheduler)
        start_step = load_checkpoint(
            ddp_policy,  # Pass DDP model, function will handle .module
            optimizer,
            scheduler,
            args.resume_from,
            device,
            rank,
        )
        # Sync step across all ranks for DDP
        if world_size > 1:
            step_tensor = torch.tensor([start_step], device=device)
            dist.broadcast(step_tensor, src=0)
            start_step = step_tensor.item()
    
    step = start_step
    ddp_policy.train()

    if is_main(rank) and start_step > 0:
        print(f"[rank {rank}] Resuming training from step {start_step}")

    while step < cfg.total_steps:
        # ----- 1) sample tasks + prompts (multi-task) -----
        chosen_task_ids = [random.randrange(num_tasks) for _ in range(per_device_batch)]

        # async generate challenges from your envs
        challenges = await async_generate_challenges(tasks, chosen_task_ids)
        prompts = [ch.prompt for ch in challenges]

        # ----- 2) rollout: generate responses from current policy -----
        # Model is already in bfloat16, so generation will use bfloat16 automatically
        input_ids, attn_mask, prompt_lens, responses = generate_responses(
            ddp_policy.module,
            tokenizer,
            prompts,
            cfg,
            device,
        )
        # Cache clearing removed - let PyTorch manage memory automatically

        # ----- 3) compute rewards via your task evaluators -----
        scores = await async_evaluate_challenges(
            tasks,
            chosen_task_ids,
            responses,
            challenges,
        )
        rewards = torch.tensor(scores, dtype=torch.float32, device=device)  # [B]

        # ----- 4) logprob_old and logprob_ref (no grad) -----
        # Use autocast for inference to save memory
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
            # Compute both in parallel for better GPU utilization
            logprob_old = compute_logprobs_for_responses(
                ddp_policy.module, input_ids, attn_mask, prompt_lens
            ).to(device)

            # Reference policy is now on GPU, no CPU transfer needed
            logprob_ref = compute_logprobs_for_responses(
                ref_policy, input_ids, attn_mask, prompt_lens
            ).to(device)

        # ----- 5) GRPO weights -----
        weights = compute_grpo_weights(rewards, logprob_old, logprob_ref, cfg)  # [B]

        # ----- 6) training step (recompute logprobs with grad) -----
        ddp_policy.train()
        optimizer.zero_grad(set_to_none=True)

        # Use autocast for mixed precision training (bfloat16 forward, float32 backward)
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
            # Do ONE forward pass (unavoidable for gradients), then process logits in tiny chunks
            out = ddp_policy(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits  # [B, T, V] - this is the memory bottleneck
        
        B, T, V = logits.shape
        target_ids = input_ids[:, 1:]  # [B, T-1]
        
        # Pre-compute response mask
        Tm1 = T - 1
        idxs = torch.arange(Tm1, device=device).unsqueeze(0).expand(B, -1)
        start = (prompt_lens - 1).clamp(min=0).unsqueeze(1)
        resp_mask = (idxs >= start).float() * attn_mask[:, 1:].float()  # [B, T-1]
        resp_len = resp_mask.sum(dim=1).clamp(min=1.0)  # [B]
        
        # Process logits in chunks to balance memory and speed
        # Increased from 4 to 16 for better throughput (can adjust based on GPU memory)
        chunk_size = 16
        resp_logprob_sum = None
        
        for i in range(0, T - 1, chunk_size):
            end_idx = min(i + chunk_size, T - 1)
            chunk_len = end_idx - i
            
            # Extract tiny slice of logits (preserves gradients)
            logits_chunk = logits[:, i:end_idx+1, :]  # [B, chunk_len+1, V]
            target_chunk = target_ids[:, i:end_idx]  # [B, chunk_len]
            mask_chunk = resp_mask[:, i:end_idx]  # [B, chunk_len]
            
            # Use cross_entropy (more memory efficient than log_softmax)
            logits_flat = logits_chunk[:, :-1, :].reshape(-1, V)  # [B*chunk_len, V]
            target_flat = target_chunk.reshape(-1)  # [B*chunk_len]
            
            # NLL per token (negative log likelihood) - preserves gradients
            nll_chunk = F.cross_entropy(logits_flat, target_flat, reduction='none')  # [B*chunk_len]
            nll_chunk = nll_chunk.reshape(B, chunk_len)  # [B, chunk_len]
            
            # Apply mask and accumulate (keep gradients)
            chunk_sum = (nll_chunk * mask_chunk).sum(dim=1)  # [B]
            
            if resp_logprob_sum is None:
                resp_logprob_sum = chunk_sum
            else:
                resp_logprob_sum = resp_logprob_sum + chunk_sum
            
            # Clean up intermediate tensors (but keep resp_logprob_sum)
            del logits_chunk, target_chunk, mask_chunk, logits_flat, target_flat, nll_chunk, chunk_sum
        
        # Convert NLL to logprob (negate)
        avg_logprob_new = -resp_logprob_sum / resp_len  # [B]
        loss = -(weights * avg_logprob_new).mean()
        
        # Use scaler for mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Cleanup (only clear cache once per step, not multiple times)
        del out, logits, resp_logprob_sum, resp_mask, resp_len

        step += 1

        # ----- logging (global averages) -----
        if step % cfg.log_every == 0:
            with torch.no_grad():
                local_reward_mean = rewards.mean()
                local_kl_mean = (logprob_old - logprob_ref).mean()
                metrics = torch.stack([local_reward_mean, local_kl_mean]).to(device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics /= world_size
                avg_reward, avg_kl = metrics.tolist()

            if is_main(rank):
                print(
                    f"step {step} | loss {loss.item():.4f} | "
                    f"reward {avg_reward:.4f} | kl {avg_kl:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.3e}"
                )

        # ----- checkpoint -----
        if is_main(rank) and step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"step-{step}")
            save_checkpoint(
                ddp_policy.module,
                optimizer,
                scheduler,
                step,
                ckpt_dir,
                rank,
            )


# ----------------- entrypoint ----------------- #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HF model id (overrides RLConfig.model_name_or_path)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from. "
              "If 'auto', will automatically find the latest checkpoint in output_dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = RLConfig()

    # override from CLI if provided
    if args.model_name is not None:
        cfg.model_name_or_path = args.model_name
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.global_batch_size is not None:
        cfg.global_batch_size = args.global_batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    
    # Handle auto-resume
    if args.resume_from == "auto":
        latest_ckpt = find_latest_checkpoint(cfg.output_dir)
        if latest_ckpt:
            args.resume_from = latest_ckpt
            print(f"Auto-resuming from latest checkpoint: {latest_ckpt}")
        else:
            args.resume_from = None
            print("No checkpoint found for auto-resume, starting from scratch")

    rank, world_size, local_rank = setup_ddp()
    set_seed(cfg.seed, rank)

    try:
        asyncio.run(train(rank, world_size, local_rank, cfg, args))
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
