# file: utils/tokenization.py

"""Prompt/completion helpers used by both actor and learner."""

from __future__ import annotations

from typing import List


def build_prompt(task_name: str, observation: str) -> str:
    header = f"[TASK: {task_name.upper()}]\n"
    return f"{header}{observation.strip()}\nAnswer:"


def strip_prompt_from_completion(prompt: str, completion: str) -> str:
    if completion.startswith(prompt):
        return completion[len(prompt) :].strip()
    return completion.strip()


def join_prompt_completion(prompt: str, completion: str, eos_token: str | None = None) -> str:
    text = f"{prompt}{completion}"
    if eos_token and not text.endswith(eos_token):
        text += eos_token
    return text


def batch_join(prompts: List[str], completions: List[str], eos_token: str | None = None) -> List[str]:
    return [join_prompt_completion(p, c, eos_token) for p, c in zip(prompts, completions)]

