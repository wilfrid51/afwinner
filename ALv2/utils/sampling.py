# file: utils/sampling.py

"""Shared model sampling utilities."""

from __future__ import annotations

import torch


def sample_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0]
    prompt_len = inputs["input_ids"].shape[-1]
    completion_ids = generated[prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return prompt + completion

