# rl_grpo_multitask.py
import os
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model

import abd, ded, sat
from abd import ABDTask
from ded import DEDTask
from sat import SATTask

from models import Challenge


# ----------------- config ----------------- #

@dataclass
class RLConfig:
    # ---- model / generation ----
    model_name_or_path: str = "trongg/Affine_robertoCalories"  # TODO: set your HF model here
    max_prompt_len: int = 512
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0

    # ---- optimization ----
    lr: float = 5e-5
    weight_decay: float = 0.01
    total_steps: int = 5000
    global_batch_size: int = 64  # total across all GPUs

    # ---- RL / GRPO ----
    kl_coef: float = 0.05
    adv_norm_eps: float = 1e-8
    weight_clip: float = 5.0

    # ---- LoRA ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ---- logging / ckpt ----
    log_every: int = 10
    save_every: int = 1000
    output_dir: str = "./rl_lora_ckpts"


# ----------------- task interface ----------------- #

# class BaseTask:
#     """
#     You already have 8 task classes.
#     Wrap them so they satisfy:

#       - sample_prompt(self) -> str
#       - score(self, prompt: str, response: str) -> float
#     """

#     def sample_prompt(self) -> str:
#         raise NotImplementedError

#     def score(self, prompt: str, response: str) -> float:
#         raise NotImplementedError


def build_tasks() -> List[ABDTask | DEDTask | SATTask]:
    """
    Return list of 3 tasks
    """
    return [ABDTask(), DEDTask(), SATTask()]


# ----------------- DDP utils ----------------- #

def setup_ddp():
    if "RANK" not in os.environ:
        raise RuntimeError(
            "run with torchrun, e.g.: torchrun --nproc_per_node=8 rl_grpo_multitask.py"
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
    Adjust target_modules to match your model’s Linear module names.
    For llama/qwen-style this is usually OK.
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

    # freeze base weights, train only LoRA params
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
    Compute average logprob over response tokens for each sample.

    input_ids:      [B, T]
    attention_mask: [B, T]
    prompt_lengths: [B] (number of tokens in prompt, BEFORE generation)
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]

    log_probs = torch.log_softmax(logits, dim=-1)       # [B, T, V]
    log_probs = log_probs[:, :-1, :]                    # [B, T-1, V]
    target_ids = input_ids[:, 1:]                       # [B, T-1]

    token_logprobs = log_probs.gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)                                       # [B, T-1]

    B, Tm1 = token_logprobs.shape
    device = token_logprobs.device

    idxs = torch.arange(Tm1, device=device).unsqueeze(0).expand(B, -1)  # [B, T-1]
    start = (prompt_lengths - 1).clamp(min=0).unsqueeze(1)              # [B, 1]

    # mask for response tokens (positions >= prompt_length after shift)
    resp_mask = (idxs >= start).float() * attention_mask[:, 1:].float()  # [B, T-1]

    resp_logprob_sum = (token_logprobs * resp_mask).sum(dim=1)  # [B]
    resp_len = resp_mask.sum(dim=1).clamp(min=1.0)              # [B]

    avg_logprob = resp_logprob_sum / resp_len                   # [B]
    return avg_logprob


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    cfg: RLConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Generate responses for a batch of prompts.

    returns:
      all_input_ids:     [B, T]
      all_attention_mask:[B, T]
      prompt_lengths:    [B]
      responses_text:    list[str]
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
    kl = (logprob_old - logprob_ref)          # [B]
    adv = rewards - cfg.kl_coef * kl         # [B]

    mean = adv.mean()
    std = adv.std(unbiased=False)
    norm_adv = (adv - mean) / (std + cfg.adv_norm_eps)

    weights = torch.clamp(norm_adv, -cfg.weight_clip, cfg.weight_clip)
    return weights.detach()


# ----------------- main training ----------------- #

def main():
    cfg = RLConfig()

    rank, world_size, local_rank = setup_ddp()
    set_seed(42, rank)
    device = torch.device(f"cuda:{local_rank}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # dtype
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # base policy + reference
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
    )

    policy.to(device)
    ref_policy.to(device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    # LoRA on policy only
    policy = add_lora(policy, cfg)
    policy.to(device)

    ddp_policy = DDP(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    optim_params = get_trainable_params(ddp_policy)
    optimizer = torch.optim.AdamW(optim_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_steps)

    tasks = build_tasks()
    num_tasks = len(tasks)
    assert num_tasks == 8, f"expected 8 tasks, got {num_tasks}"

    per_device_batch = max(1, cfg.global_batch_size // world_size)

    if is_main(rank):
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"world_size={world_size}, per_device_batch={per_device_batch}")
        print(f"trainable params: {sum(p.numel() for p in optim_params)}")

    step = 0
    ddp_policy.train()

    try:
        while step < cfg.total_steps:
            # ----- 1) sample tasks + prompts -----
            chosen_task_ids = [random.randrange(num_tasks) for _ in range(per_device_batch)]
            prompts = [tasks[t].generate().prompt for t in chosen_task_ids]

            # ----- 2) rollout: generate responses -----
            input_ids, attn_mask, prompt_lens, responses = generate_responses(
                ddp_policy.module, tokenizer, prompts, cfg, device
            )

            # ----- 3) compute rewards -----
            async def compute_rewards() -> torch.Tensor:
                rewards_list = []
                for t_idx, prompt, resp in zip(chosen_task_ids, prompts, responses):
                    r = await tasks[t_idx].evaluate(resp, Challenge(prompt=prompt, env=tasks[t_idx].env))
                    rewards_list.append(r.score)
                return torch.tensor(rewards_list, dtype=torch.float32, device=device)  # [B]
            
            rewards = compute_rewards()

            # ----- 4) logprob_old and logprob_ref -----
            logprob_old = compute_logprobs_for_responses(
                ddp_policy.module, input_ids, attn_mask, prompt_lens
            ).to(device)

            logprob_ref = compute_logprobs_for_responses(
                ref_policy, input_ids, attn_mask, prompt_lens
            ).to(device)

            # ----- 5) GRPO weights -----
            weights = compute_grpo_weights(rewards, logprob_old, logprob_ref, cfg)  # [B]

            # ----- 6) training step (recompute logprobs with grad) -----
            ddp_policy.train()
            optimizer.zero_grad(set_to_none=True)

            out = ddp_policy(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits
            log_probs = torch.log_softmax(logits, dim=-1)      # [B,T,V]
            log_probs = log_probs[:, :-1, :]                   # [B,T-1,V]
            target_ids = input_ids[:, 1:]                      # [B,T-1]
            token_logprobs = log_probs.gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)                                      # [B,T-1]

            B, Tm1 = token_logprobs.shape
            idxs = torch.arange(Tm1, device=device).unsqueeze(0).expand(B, -1)
            start = (prompt_lens - 1).clamp(min=0).unsqueeze(1)
            resp_mask = (idxs >= start).float() * attn_mask[:, 1:].float()

            resp_logprob_sum = (token_logprobs * resp_mask).sum(dim=1)
            resp_len = resp_mask.sum(dim=1).clamp(min=1.0)
            avg_logprob_new = resp_logprob_sum / resp_len       # [B]

            loss = -(weights * avg_logprob_new).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            # ----- logging -----
            if is_main(rank) and step % cfg.log_every == 0:
                avg_reward = rewards.mean().item()
                avg_kl = (logprob_old - logprob_ref).mean().item()
                print(
                    f"step {step} | loss {loss.item():.4f} | reward {avg_reward:.4f} | "
                    f"kl {avg_kl:.4f} | lr {scheduler.get_last_lr()[0]:.3e}"
                )

            # ----- checkpoint -----
            if is_main(rank) and step % cfg.save_every == 0:
                ckpt_dir = os.path.join(cfg.output_dir, f"step-{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                ddp_policy.module.save_pretrained(ckpt_dir)

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
