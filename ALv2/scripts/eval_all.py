# file: scripts/eval_all.py

"""Evaluate a trained adapter across all registered tasks."""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasks.registry import build_tasks
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on all tasks")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--tasks_config", default="configs/tasks.yaml")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_model(base_model: str, ckpt: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model = PeftModel.from_pretrained(model, ckpt).to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    setup_logging()

    model, tokenizer = load_model(args.base_model, args.ckpt, args.device)
    task_map = build_tasks(args.tasks_config)
    scores = {}
    for name, task in task_map.items():
        logging.info("Evaluating %s", name)
        score = task.evaluate(model, tokenizer, num_episodes=args.num_episodes)
        scores[name] = score
        logging.info("%s: %.3f", name, score)
    avg = sum(scores.values()) / max(1, len(scores))
    logging.info("Overall average: %.3f", avg)


if __name__ == "__main__":
    main()

