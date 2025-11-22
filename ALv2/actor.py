# file: actor.py

"""Multi-task actor that collects trajectories across Affine + AgentGym tasks."""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from replay_buffer import ReplayBuffer, ReplayDiskWriter, Transition
from tasks import BaseTask
from tasks.registry import build_tasks
from utils.logging import StatsTracker, setup_logging
from utils.tokenization import build_prompt, strip_prompt_from_completion
from utils.sampling import sample_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-task GRPO actor")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--tasks_config", type=str, default=None)
    parser.add_argument("--task_filter", type=str, nargs="*", default=None)
    parser.add_argument("--replay_dir", type=str, default="replay")
    parser.add_argument("--max_steps_per_episode", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=0, help="0 = infinite")
    parser.add_argument("--schedule", type=str, choices=["round_robin", "random"], default="round_robin")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help="Override device string, e.g. cuda:0")
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_args()


# ---------------------------------------------------------------------------
def resolve_device(arg_device: str | None) -> str:
    if arg_device:
        return arg_device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return f"cuda:{local_rank}"
    return "cpu"


# ---------------------------------------------------------------------------
def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
    ).to(args.device)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
def collect_episode(
    task: BaseTask,
    model,
    tokenizer,
    args: argparse.Namespace,
) -> List[Transition]:
    offline = getattr(task, "offline", False)
    obs = task.reset()
    transitions: List[Transition] = []
    for _ in range(args.max_steps_per_episode):
        if offline:
            next_obs, reward, done, info = task.step("")
            action_text = info.get("expert_action", "")
        else:
            prompt = build_prompt(task.name, obs)
            completion = sample_model(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            action_text = strip_prompt_from_completion(prompt, completion)
            next_obs, reward, done, info = task.step(action_text)
        transitions.append(
            Transition(
                obs=obs,
                action=action_text,
                reward=reward,
                next_obs=next_obs,
                done=done,
                task=task.name,
                offline=offline,
            )
        )
        obs = next_obs
        if done or not obs:
            break
    return transitions


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    setup_logging()

    model, tokenizer = load_model_and_tokenizer(args)
    task_map: Dict[str, BaseTask] = build_tasks(args.tasks_config)
    if args.task_filter:
        task_map = {k: v for k, v in task_map.items() if k in set(args.task_filter)}
        if not task_map:
            raise RuntimeError("Task filter removed all tasks")
    tasks = list(task_map.values())

    replay = ReplayBuffer(capacity=200_000)
    writer = ReplayDiskWriter(args.replay_dir)
    tracker = StatsTracker()

    logger.info("Starting actor with %d tasks", len(tasks))
    idx = 0
    episode_count = 0
    try:
        while True:
            if args.episodes and episode_count >= args.episodes:
                break
            if args.schedule == "round_robin":
                task = tasks[idx % len(tasks)]
                idx += 1
            else:
                task = random.choice(tasks)
            start = time.time()
            transitions = collect_episode(task, model, tokenizer, args)
            if not transitions:
                continue
            replay.extend(transitions)
            writer.write_episode(transitions)
            for t in transitions:
                tracker.record(t.task, t.reward, t.done)
            episode_count += 1
            if episode_count % args.log_interval == 0:
                logger.info("Episode %d | task=%s | steps=%d | reward=%.2f | dt=%.2fs",
                            episode_count,
                            task.name,
                            len(transitions),
                            sum(t.reward for t in transitions),
                            time.time() - start)
                tracker.report()
    except KeyboardInterrupt:
        logger.info("Actor interrupted; exiting cleanly")


if __name__ == "__main__":
    main()

