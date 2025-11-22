# file: learner.py

"""GRPO-style learner that consumes trajectories dumped by the actors."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from replay_buffer import ReplayBuffer, ReplayDirectoryReader, Transition
from utils.logging import setup_logging
from utils.metrics import AverageMeter
from utils.tokenization import build_prompt, join_prompt_completion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO learner")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--replay_dir", type=str, default="replay")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument("--min_buffer", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda_bc", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Truncate sequences to this many tokens")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    return parser.parse_args()


# ---------------------------------------------------------------------------
def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    base_model.to(args.device)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.train()
    return model, tokenizer


# ---------------------------------------------------------------------------
def build_batch_tensors(
    batch: List[Transition],
    tokenizer,
    device: str,
    max_length: int,
):
    prompts = [build_prompt(tr.task, tr.obs) for tr in batch]
    completions = [tr.action for tr in batch]
    texts = [join_prompt_completion(p, c, tokenizer.eos_token) for p, c in zip(prompts, completions)]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    prompt_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    prompt_lengths = (prompt_tokens["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    return inputs, prompt_lengths


# ---------------------------------------------------------------------------
def compute_losses(model, tokenizer, batch: List[Transition], args):
    inputs, prompt_lengths = build_batch_tensors(batch, tokenizer, args.device, args.max_seq_len)
    rewards = torch.tensor([tr.reward for tr in batch], device=args.device)
    offline_mask = torch.tensor([1.0 if tr.offline else 0.0 for tr in batch], device=args.device)

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    target_ids = inputs["input_ids"][:, 1:].contiguous()
    attn_mask = inputs["attention_mask"][:, 1:].to(logits.dtype)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * attn_mask

    rl_masks = torch.zeros_like(target_ids, dtype=token_log_probs.dtype)
    prompt_lens = prompt_lengths.detach().cpu().tolist()
    for idx, prompt_len in enumerate(prompt_lens):
        start = max(int(prompt_len) - 1, 0)
        rl_masks[idx, start:] = 1.0  # shift because of target alignment
    rl_token_log_probs = token_log_probs * rl_masks
    seq_log_prob = rl_token_log_probs.sum(dim=1)

    advantages = torch.zeros_like(rewards)
    task_to_indices: Dict[str, List[int]] = {}
    for i, tr in enumerate(batch):
        task_to_indices.setdefault(tr.task, []).append(i)
    for indices in task_to_indices.values():
        task_rewards = rewards[indices]
        mean = task_rewards.mean()
        std = task_rewards.std(unbiased=False)
        advantages[indices] = (task_rewards - mean) / (std + 1e-6)
    rl_loss = -(advantages.detach() * seq_log_prob).mean()

    token_counts = (rl_masks * attn_mask).sum(dim=1).clamp(min=1.0)
    avg_token_nll = -(rl_token_log_probs.sum(dim=1) / token_counts)
    if offline_mask.sum() > 0:
        bc_loss = (avg_token_nll * offline_mask).sum() / offline_mask.sum()
    else:
        bc_loss = torch.tensor(0.0, device=args.device)

    total_loss = rl_loss + args.lambda_bc * bc_loss
    return total_loss, rl_loss.detach(), bc_loss.detach(), rewards.mean().detach()


# ---------------------------------------------------------------------------
def save_checkpoint(model, tokenizer, output_dir: str | Path, step: int) -> None:
    ckpt_dir = Path(output_dir) / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    latest_dir = Path(output_dir) / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(ckpt_dir, latest_dir)


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging()

    model, tokenizer = load_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    replay = ReplayBuffer(capacity=500_000)
    reader = ReplayDirectoryReader(args.replay_dir)
    meters = AverageMeter()

    step = 0
    try:
        while step < args.total_steps:
            new_transitions = reader.fetch_new()
            if new_transitions:
                replay.extend(new_transitions)
            if len(replay) < args.min_buffer:
                time.sleep(args.poll_interval)
                continue

            batch = replay.sample(args.batch_size)
            loss, rl_loss, bc_loss, reward_mean = compute_losses(model, tokenizer, batch, args)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            meters.update("loss", float(loss.detach().cpu()))
            meters.update("rl", float(rl_loss.cpu()))
            meters.update("bc", float(bc_loss.cpu()))
            meters.update("reward", float(reward_mean.cpu()))

            step += 1
            if step % args.log_interval == 0:
                summary = meters.summary()
                logger.info(
                    "step=%d | loss=%.4f | rl=%.4f | bc=%.4f | reward=%.3f | buffer=%d",
                    step,
                    summary.get("loss", 0.0),
                    summary.get("rl", 0.0),
                    summary.get("bc", 0.0),
                    summary.get("reward", 0.0),
                    len(replay),
                )
                meters = AverageMeter()

            if step % args.save_interval == 0:
                save_checkpoint(model, tokenizer, args.output_dir, step)
    except KeyboardInterrupt:
        logger.info("Interrupted; saving final checkpoint")
    finally:
        save_checkpoint(model, tokenizer, args.output_dir, step)


if __name__ == "__main__":
    main()

