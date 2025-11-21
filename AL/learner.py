# learner.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
from typing import List, Dict, Any

import re
import shutil

import torch
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel

from replay_buffer import LocalReplayBuffer


logger = logging.getLogger("LEARNER")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

def log_with_color(color: str, message: str):
    """
    Log a message with a color.
    """
    print(f"\033[{color}m{message}\033[0m")


def set_torch_defaults():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def compute_logprobs_only(
    model: nn.Module,
    sequences: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_token_id: int,
    chunk_size: int = 4,
) -> torch.Tensor:
    """
    Compute sum of log-probs of completion tokens (after prompt) for each sequence.

    Uses micro-batching along batch dimension to reduce peak memory,
    which is important on single GPU with long sequences.
    """
    device = sequences.device
    N, T = sequences.shape
    all_sums = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        seq_chunk = sequences[start:end]          # [C, T]
        pl_chunk = prompt_lengths[start:end]      # [C]

        attention_mask = (seq_chunk != pad_token_id).long()

        outputs = model(seq_chunk, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = seq_chunk[:, 1:]

        del outputs

        log_probs = torch.log_softmax(logits, dim=-1)
        del logits

        token_logprobs = log_probs.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)

        del log_probs

        C, Tm1 = token_logprobs.shape
        positions = torch.arange(Tm1, device=device).unsqueeze(0).expand(C, -1)
        prompt_lens_minus1 = (pl_chunk - 1).unsqueeze(1).expand_as(positions)

        completion_mask = positions >= prompt_lens_minus1
        completion_mask = completion_mask & (targets != pad_token_id)

        token_logprobs = token_logprobs * completion_mask
        logprob_sums = token_logprobs.sum(dim=-1)  # [C]

        all_sums.append(logprob_sums)

        del (
            token_logprobs,
            targets,
            attention_mask,
            positions,
            prompt_lens_minus1,
            completion_mask,
        )
        torch.cuda.empty_cache()

    return torch.cat(all_sums, dim=0)


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    GRPO-style: subtract group mean reward within each group of size group_size.
    """
    assert rewards.numel() % group_size == 0
    B = rewards.numel() // group_size
    grouped = rewards.view(B, group_size)
    group_means = grouped.mean(dim=1, keepdim=True)
    adv_grouped = grouped - group_means
    return adv_grouped.view(-1)


def cleanup_old_checkpoints(output_dir: str, keep_last: int) -> None:
    """
    Keep only the newest `keep_last` step-* checkpoint folders in `output_dir`.
    Older ones are deleted.
    """
    if keep_last <= 0:
        return

    if not os.path.isdir(output_dir):
        return

    # Collect all step-* directories
    entries = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and name.startswith("step-"):
            # Extract numeric step if possible
            m = re.search(r"step-(\d+)", name)
            step = int(m.group(1)) if m else -1
            entries.append((step, name))

    # Sort by step number (ascending: oldest first)
    entries.sort(key=lambda x: x[0])

    if len(entries) <= keep_last:
        return

    # Keep the last `keep_last` entries (highest step numbers)
    to_delete = entries[:-keep_last]
    for step, name in to_delete:
        path = os.path.join(output_dir, name)
        try:
            shutil.rmtree(path)
            logger.info(f"Deleted old checkpoint: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete old checkpoint {path}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Learner for multi-task actor-learner (ABD/DED/SAT)")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base model name/path (e.g. Qwen3-4B).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grpo-actor-learner-multitask",
        help="Where to save checkpoints.",
    )
    parser.add_argument(
        "--replay_dir",
        type=str,
        default="replay",
        help="Directory used for local replay buffer (file-based).",
    )
    parser.add_argument(
        "--batch_groups",
        type=int,
        default=8,
        help="Number of prompt groups per batch.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Number of completions per prompt.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="LR for LoRA params.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Scheduler warmup steps.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=100000,
        help="Total training steps (for schedule).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 for the model weights.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1600,
        help="Max total tokens (prompt+completion) in learner.",
    )
    parser.add_argument(
        "--keep_last",
        type=int,
        default=3,
        help="Number of recent step-* checkpoints to keep. 0 = keep all.",
    )

    # LoRA config for fresh training
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )

    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Optional resume from LoRA checkpoint.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for clipping. 0.0 = no clipping.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_defaults()

    # IMPORTANT: no mixed_precision here, we keep model in bf16/fp16 as loaded.
    accelerator = Accelerator(
        mixed_precision="no",
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Learner saving to {args.output_dir}")
        logger.info(f"Using replay_dir={args.replay_dir}")
        logger.info(f"batch_groups={args.batch_groups}, group_size={args.group_size}")
        logger.info(f"max_seq_len={args.max_seq_len}, total_steps={args.total_steps}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=None,
    )

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    # PEFT / LoRA
    if args.resume_dir is not None:
        # Resolve to absolute path to avoid HuggingFace Hub confusion
        resume_path = os.path.abspath(args.resume_dir)
        if not os.path.exists(resume_path):
            raise ValueError(
                f"Resume directory does not exist: {resume_path}\n"
                f"Please check the path or remove --resume_dir to start fresh training."
            )
        if accelerator.is_main_process:
            logger.info(f"Resuming LoRA from {resume_path}")
        # Use local_files_only=True to prevent PEFT from trying to download from HuggingFace Hub
        model = PeftModel.from_pretrained(
            base_model, 
            resume_path,
            local_files_only=True
        )
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        lora_targets = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    if accelerator.is_main_process:
        try:
            model.print_trainable_parameters()
        except Exception:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Trainable params: {trainable} / {total}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    buffer = LocalReplayBuffer(replay_dir=args.replay_dir)

    # Extract starting step from resume_dir if resuming
    global_step = 0
    if args.resume_dir is not None:
        # Extract step number from resume_dir path (e.g., "output/step-250" -> 250)
        step_match = re.search(r"step-(\d+)", args.resume_dir)
        if step_match:
            global_step = int(step_match.group(1))
            if accelerator.is_main_process:
                logger.info(f"Resuming from step {global_step}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Could not extract step number from resume_dir {args.resume_dir}, starting from step 0")

    if accelerator.is_main_process:
        log_with_color("32;1", f"[LEARNER] Entering training loop... Starting from step {global_step}")

    while global_step < args.total_steps:
        # 1) Sample groups from replay buffer
        groups: List[Dict[str, Any]] = buffer.sample_batch_groups(
            batch_groups=args.batch_groups,
            timeout=20,
        )
        if len(groups) == 0:
            if accelerator.is_main_process:
                logger.info("[LEARNER] No groups in replay buffer yet, waiting...")
            continue

        # Flatten into sequences, rewards, and track task labels
        texts: List[str] = []
        prompt_lengths: List[int] = []
        rewards_list: List[float] = []
        task_names: List[str] = []

        group_size = args.group_size

        for g in groups:
            prompt = g["prompt"]
            completions = g["completions"]
            rewards = g["rewards"]
            task_name = g.get("task_name", "UNK")

            assert len(completions) == group_size
            assert len(rewards) == group_size

            # Tokenize prompt once to know its token length
            enc_prompt = tokenizer(prompt, add_special_tokens=False)
            plen = len(enc_prompt["input_ids"])

            for comp, r in zip(completions, rewards):
                full_text = prompt + comp  # must match actor decode behavior
                texts.append(full_text)
                prompt_lengths.append(plen)
                rewards_list.append(float(r))
                task_names.append(task_name)

        # Tokenize all full sequences together, pad
        enc_all = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        input_ids = enc_all["input_ids"].to(accelerator.device)
        pad_token_id = tokenizer.pad_token_id

        prompt_lengths_tensor = torch.tensor(
            prompt_lengths,
            dtype=torch.long,
            device=accelerator.device,
        )

        rewards_tensor = torch.tensor(
            rewards_list,
            dtype=torch.float32,
            device=accelerator.device,
        )

        # 2) Compute GRPO loss
        model.train()

        with accelerator.accumulate(model):
            logprob_sums = compute_logprobs_only(
                model=model,
                sequences=input_ids,
                prompt_lengths=prompt_lengths_tensor,
                pad_token_id=pad_token_id,
                chunk_size=1,  # tune if you want
            )

            advantages = compute_group_advantages(
                rewards=rewards_tensor,
                group_size=group_size,
            )

            adv_mean = advantages.mean()
            adv_std = advantages.std().clamp(min=1e-6)
            norm_adv = (advantages - adv_mean) / adv_std

            loss = -(norm_adv.detach() * logprob_sums).mean()

            accelerator.backward(loss)

            # Compute and clip gradient norm
            grad_norm = None
            try:
                if args.max_grad_norm > 0.0:
                    # Clip gradients - use unwrapped model for distributed training
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        accelerator.unwrap_model(model).parameters(), 
                        args.max_grad_norm
                    )
                else:
                    # Just compute norm without clipping
                    total_norm_sq = 0.0
                    for p in accelerator.unwrap_model(model).parameters():
                        if p.grad is not None:
                            pn = p.grad.data.norm(2)
                            total_norm_sq += pn.item() ** 2
                    grad_norm = (total_norm_sq ** 0.5) if total_norm_sq > 0 else 0.0
            except Exception:
                pass

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if accelerator.is_main_process:
            # Global stats
            avg_reward = rewards_tensor.mean().item()
            min_reward = rewards_tensor.min().item()
            max_reward = rewards_tensor.max().item()
            std_reward = rewards_tensor.std().item()
            avg_adv = advantages.mean().item()
            std_adv = advantages.std().item()
            lr = (
                scheduler.get_last_lr()[0]
                if hasattr(scheduler, "get_last_lr")
                else optimizer.param_groups[0]["lr"]
            )

            # Per-task reward stats
            per_task = {}
            for tname, r in zip(task_names, rewards_list):
                per_task.setdefault(tname, []).append(r)

            per_task_strs = []
            for tname, vals in per_task.items():
                if vals:
                    avg_t = sum(vals) / len(vals)
                    per_task_strs.append(f"{tname}={avg_t:.3f} (n={len(vals)})")
            per_task_report = " | ".join(per_task_strs) if per_task_strs else "no_task_stats"

            msg = (
                f"step={global_step} loss={loss.item():.4f} "
                f"reward={avg_reward:.4f} (min={min_reward:.4f}, max={max_reward:.4f}, std={std_reward:.4f}) "
                f"adv={avg_adv:.4f} (std={std_adv:.4f}) lr={lr:.3e} "
                f"tasks: {per_task_report}"
            )
            if grad_norm is not None:
                msg += f" grad_norm={grad_norm:.4f}"
            # logger.info(msg)
            log_with_color("32;1", msg)

        # 3) Save checkpoint periodically
        if accelerator.is_main_process and (global_step % args.save_every == 0):
            ckpt_dir = os.path.join(args.output_dir, f"step-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            log_with_color("32;1", f"[LEARNER] Saving checkpoint to {ckpt_dir}")
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

            # Write version.txt into this checkpoint directory
            version_path = os.path.join(args.output_dir, "version.txt")
            try:
                with open(version_path, "w") as f:
                    f.write(str(global_step))
            except Exception as e:
                logger.warning(f"Failed to write version.txt in {args.output_dir}: {e}")

            # Update "latest" symlink or marker for actors
            latest_dir = os.path.join(args.output_dir, "latest")
            try:
                if os.path.islink(latest_dir) or os.path.exists(latest_dir):
                    # If it's a symlink, unlink; if it's a stray directory/file, remove
                    try:
                        os.unlink(latest_dir)
                    except IsADirectoryError:
                        shutil.rmtree(latest_dir)
                    except Exception:
                        pass
                os.symlink(ckpt_dir, latest_dir)
            except Exception as e:
                # Not fatal – actors can still point directly to step-* directories
                logger.warning(f"Could not update 'latest' symlink: {e}")

        # Clean up older checkpoints
        cleanup_old_checkpoints(args.output_dir, args.keep_last)

    # Final save
    if accelerator.is_main_process:
        ckpt_dir = os.path.join(args.output_dir, f"final-step-{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        log_with_color("32;1", f"[LEARNER] Saving final checkpoint to {ckpt_dir}")
        accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)


if __name__ == "__main__":
    main()
