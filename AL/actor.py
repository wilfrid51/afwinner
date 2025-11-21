# actor.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import argparse
import asyncio
import random
import logging
import itertools
import gc
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from replay_buffer import LocalReplayBuffer

# Your task envs (ABDTask, DEDTask, SATTask, etc.)
from env import *


logger = logging.getLogger("ACTOR")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


def log_with_color(color: str, message: str):
    """
    Log a message with a color.
    """
    print(f"\033[{color}m{message}\033[0m")


def decode_generations(
    tokenizer,
    sequences: torch.Tensor,
    prompts: List[str],
    group_size: int,
) -> List[str]:
    """
    Decode full sequences and remove the prompt prefix by string matching.
    This is robust to padding-side issues, etc.
    """
    sequences = sequences.detach().cpu()
    completions: List[str] = []

    for i, seq in enumerate(sequences):
        full_text = tokenizer.decode(
            seq,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        prompt = prompts[i // group_size]
        if full_text.startswith(prompt):
            comp = full_text[len(prompt):]
        else:
            idx = full_text.rfind(prompt)
            if idx != -1:
                comp = full_text[idx + len(prompt):]
            else:
                comp = full_text
        completions.append(comp.strip())
    return completions


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-actor process for ABD/DED/SAT RL")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base model name or path (e.g. Qwen3-4B)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional LoRA/PEFT adapter to load on top of the base model.",
    )
    parser.add_argument(
        "--replay_dir",
        type=str,
        default="replay",
        help="Directory used for local replay buffer (file-based).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts to generate per loop iteration.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Number of completions per prompt (GRPO group size).",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Max tokens for prompt.",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=512,
        help="Max new tokens.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device to run on (e.g. cuda, cuda:0). "
            "If not set, we use LOCAL_RANK to choose cuda:<local_rank>."
        ),
    )
    parser.add_argument(
        "--reload_every",
        type=int,
        default=120,
        help="Seconds between checking for updated weights on disk (optional).",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help=(
            "Optional path where learner saves latest model (for live reload). "
            "If provided, actor will periodically reload weights from here."
        ),
    )
    parser.add_argument(
        "--actor_id",
        type=int,
        default=None,
        help=(
            "ID for this actor. If None, uses LOCAL_RANK or 0. "
            "Used for task rotation and logging."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="ABD,DED,SAT",
        help=(
            "Comma-separated list of tasks to cycle through, e.g. "
            "'ABD,DED,SAT' or 'ABD,ABD,DED'. "
            "Each actor rotates this list based on its actor_id."
        ),
    )
    parser.add_argument(
        "--no_sampling",
        action="store_true",
        help="Disable sampling (use greedy decoding) for debugging.",
    )

    return parser.parse_args()


def resolve_device_and_actor_id(args: argparse.Namespace) -> Tuple[torch.device, int, int]:
    """
    Decide which GPU/device and actor_id to use, based on:
    - LOCAL_RANK (torchrun / accelerate)
    - optional --device
    - optional --actor_id

    Returns:
        device, actor_id, world_size
    """
    # Torchrun / accelerate usually set these
    local_rank_env = os.environ.get("LOCAL_RANK")
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        local_rank = 0

    if world_size_env is not None:
        world_size = int(world_size_env)
    else:
        world_size = 1

    # Device resolution:
    # - If user passed --device, trust it.
    # - Else, if LOCAL_RANK exists, use cuda:<local_rank>.
    # - Else, default to cuda:0.
    if args.device is not None:
        device_str = args.device
    else:
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)

    # Actor ID:
    # - If --actor_id passed, use it
    # - Else, use RANK (or local_rank) as actor_id
    if args.actor_id is not None:
        actor_id = args.actor_id
    elif rank_env is not None:
        actor_id = int(rank_env)
    else:
        actor_id = local_rank

    return device, actor_id, world_size


def build_task_cycle(args: argparse.Namespace, actor_id: int):
    """
    Build a task cycle per actor, rotating the task list
    so that different actors see different starting tasks.

    Example with tasks="ABD,DED,SAT":
        actor 0 -> [ABD, DED, SAT, ABD, DED, SAT, ...]
        actor 1 -> [DED, SAT, ABD, DED, SAT, ABD, ...]
        actor 2 -> [SAT, ABD, DED, SAT, ABD, DED, ...]
    """
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not task_names:
        raise ValueError("No valid tasks specified in --tasks")

    num_tasks = len(task_names)
    offset = actor_id % num_tasks
    task_order = task_names[offset:] + task_names[:offset]

    logger.info(f"Actor {actor_id}: base tasks={task_names}, rotated task_order={task_order}")
    return itertools.cycle(task_order)


def maybe_reload_model(model, tokenizer, args, device, last_reload_time):
    """
    Reload latest LoRA weights from args.weights_dir if version changed.
    Expect args.weights_dir to point to learner's 'latest' checkpoint dir.
    """
    if args.weights_dir is None:
        log_with_color("31;1", f"[ACTOR] No weights directory provided, using base model only")
        return model, last_reload_time

    now = time.time()
    if now - last_reload_time < args.reload_every:
        log_with_color("31;1", f"[ACTOR] Time is No new weights found, using last model")
        return model, last_reload_time

    marker_path = os.path.join(args.weights_dir, "version.txt")
    if not os.path.exists(marker_path):
        log_with_color("31;1", f"[ACTOR] No version file found, using base model only")
        return model, last_reload_time

    try:
        with open(marker_path, "r") as f:
            version = f.read().strip()
    except Exception:
        log_with_color("31;1", f"[ACTOR] Failed to read version file, using base model only")
        return model, last_reload_time

    log_with_color("32;1", f"[ACTOR] Detected new learner version {version}, reloading model from {args.weights_dir}...")
    reload_start = time.time()

    # CRITICAL: Check memory BEFORE deleting old model
    # Model needs ~8GB base + 2GB transfer overhead + 1GB buffer = ~11GB free
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        free_before = (total - allocated) / (1024**3)

        required_for_reload = 11.0  # 8GB model + 2GB transfer + 1GB buffer
        if free_before < required_for_reload:
            logger.warning(
                f"[ACTOR] Insufficient memory for safe reload ({free_before:.2f} GB free, need {required_for_reload:.2f} GB). "
                f"Skipping reload to avoid OOM."
            )
            return model, last_reload_time  # Keep old model, skip reload

        logger.info(f"[ACTOR] Memory check passed: {free_before:.2f} GB free (need {required_for_reload:.2f} GB), proceeding with reload")

    # Free old model memory aggressively to avoid OOM
    log_with_color("32;1", f"[ACTOR] Freeing old model memory...")
    old_model = model  # Keep reference in case we need to restore
    del model
    gc.collect()  # Force Python garbage collection

    if device.type == 'cuda':
        torch.cuda.synchronize(device)  # Ensure deletion completes
        torch.cuda.empty_cache()

        # Check available memory before reloading
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        free_memory = total - allocated
        free_memory_gb = free_memory / (1024**3)
        log_with_color("32;1", f"[ACTOR] GPU memory: {allocated/(1024**3):.2f} GB allocated, {reserved/(1024**3):.2f} GB reserved, {free_memory_gb:.2f} GB free")

        # Aggressively free memory if low
        if free_memory_gb < 3.0:  # Need at least 3GB free for safe reload
            logger.warning(f"[ACTOR] Low GPU memory ({free_memory_gb:.2f} GB free), aggressively clearing cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            gc.collect()  # Force another GC pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

            allocated = torch.cuda.memory_allocated(device)
            free_memory = total - allocated
            free_memory_gb = free_memory / (1024**3)
            logger.info(f"[ACTOR] After aggressive cache clear: {free_memory_gb:.2f} GB free")

            # If still low, wait a bit and retry
            if free_memory_gb < 2.0:
                logger.warning(f"[ACTOR] Still low memory ({free_memory_gb:.2f} GB), waiting 5s for other processes to free memory...")
                time.sleep(5)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_allocated(device)
                free_memory = total - allocated
                free_memory_gb = free_memory / (1024**3)
                logger.info(f"[ACTOR] After wait: {free_memory_gb:.2f} GB free")

    # Reload base model to CPU first to avoid OOM
    log_with_color("32;1", f"[ACTOR] Reloading base model to CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load to CPU first - safe, no GPU memory needed
    )
    log_with_color("32;1", f"[ACTOR] Base model loaded to CPU successfully")

    # Now wait for GPU memory to be available before moving to GPU
    if device.type == 'cuda':
        # Memory requirements for model reload:
        # - Base model (4B params in bfloat16): ~8GB (4B * 2 bytes)
        # - During .to(device) transfer: +2GB temporary overhead
        # - LoRA adapter: ~100MB (small)
        # - Safety buffer: +1GB
        # Total needed: ~11GB free
        model_size_gb = 8.0  # Base model size estimate
        transfer_overhead_gb = 2.0  # Temporary memory during .to(device)
        safety_buffer_gb = 1.0  # Safety margin
        required_free_gb = model_size_gb + transfer_overhead_gb + safety_buffer_gb  # ~11GB total

        max_wait_attempts = 10  # Wait up to 50 seconds total
        wait_interval = 5  # Check every 5 seconds

        for attempt in range(max_wait_attempts):
            torch.cuda.empty_cache()  # Clear cache before checking
            torch.cuda.synchronize(device)
            gc.collect()  # Force garbage collection

            allocated = torch.cuda.memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            free_memory_gb = (total - allocated) / (1024**3)

            if free_memory_gb >= required_free_gb:
                logger.info(f"[ACTOR] GPU memory available: {free_memory_gb:.2f} GB free (need {required_free_gb:.2f} GB). Moving model to GPU...")
                break
            else:
                if attempt < max_wait_attempts - 1:
                    logger.warning(
                        f"[ACTOR] Waiting for GPU memory (attempt {attempt + 1}/{max_wait_attempts}): "
                        f"{free_memory_gb:.2f} GB free, need {required_free_gb:.2f} GB. Waiting {wait_interval}s..."
                    )
                    time.sleep(wait_interval)
                else:
                    # Final attempt failed - give up and raise error
                    logger.error(
                        f"[ACTOR] Failed to get enough GPU memory after {max_wait_attempts} attempts. "
                        f"Current: {free_memory_gb:.2f} GB free, need {required_free_gb:.2f} GB. "
                        f"Model remains on CPU - this will cause issues. Consider reducing batch_size or increasing reload_every."
                    )
                    raise RuntimeError(
                        f"Insufficient GPU memory for model reload: {free_memory_gb:.2f} GB free after waiting, "
                        f"need {required_free_gb:.2f} GB. Model loaded to CPU but cannot move to GPU."
                    )
        
        # Move base model to device with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                base_model.to(device)
                log_with_color("32;1", f"[ACTOR] Base model moved to GPU successfully")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"[ACTOR] OOM on GPU move attempt {attempt + 1}/{max_retries}, clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                    
                    # Re-check memory before retry
                    allocated = torch.cuda.memory_allocated(device)
                    free_memory_gb = (total - allocated) / (1024**3)
                    logger.info(f"[ACTOR] After cleanup: {free_memory_gb:.2f} GB free")
                else:
                    raise
    log_with_color("32;1", f"[ACTOR] Base model reloaded")

    # Reload LoRA adapters from weights_dir (learner's latest checkpoint)
    try:
        log_with_color("32;1", f"[ACTOR] Loading LoRA adapter from {args.weights_dir}...")
        ckpt_path = os.path.join(args.weights_dir, f"step-{version}")
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        log_with_color("32;1", f"[ACTOR] LoRA adapter loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load PEFT from {args.weights_dir}: {e}, using base model only.")
        model = base_model

    model.eval().to(device)
    reload_time = time.time() - reload_start
    logger.info(f"Model reload completed in {reload_time:.2f}s (version: {version})")
    return model, now


def main():
    args = parse_args()

    device, actor_id, world_size = resolve_device_and_actor_id(args)

    logger.info("=" * 80)
    logger.info(f"Actor starting | actor_id={actor_id} | world_size={world_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA path: {args.lora_path or 'None (base model only)'}")
    logger.info(f"Replay dir: {args.replay_dir}")
    logger.info(f"Batch size: {args.batch_size}, Group size: {args.group_size}")
    logger.info(f"Max prompt length: {args.max_prompt_length}, Max completion: {args.max_completion_length}")
    logger.info(f"Tasks config: {args.tasks}")
    logger.info(f"Sampling: {'OFF (greedy)' if args.no_sampling else 'ON (do_sample=True)'}")
    logger.info("=" * 80)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # critical for decoder-only models
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}, pad_token_id={tokenizer.pad_token_id}")

    logger.info(f"Loading base model: {args.model_name} (dtype=bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    base_model.to(device)
    logger.info(f"Base model loaded: {base_model.config.model_type}")

    if args.lora_path is not None:
        logger.info(f"Loading LoRA adapter from: {args.lora_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        logger.info("LoRA adapter loaded successfully")
    else:
        logger.info("Using base model (no LoRA)")
        model = base_model

    model.eval().to(device)
    logger.info(f"Model moved to {device} and set to eval mode")

    logger.info(f"Initializing local replay buffer at: {args.replay_dir}")
    buffer = LocalReplayBuffer(replay_dir=args.replay_dir)
    logger.info("Local replay buffer initialized successfully")

    # Initialize event loop before task environments (they need it for async initialization)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info("Event loop initialized")

    # Task envs - initialize inside running event loop context
    # (R2Dataset creates async tasks during __init__, which requires a running loop)
    logger.info("Initializing task environments...")
    
    async def init_tasks():
        """Initialize tasks within running event loop context"""
        return ABDTask(), DEDTask(), SATTask()
    
    abd_task, ded_task, sat_task = loop.run_until_complete(init_tasks())
    logger.info("Task environments initialized: ABD, DED, SAT")

    last_reload_time = time.time()
    start_time = time.time()
    total_prompts = 0
    total_completions = 0

    task_cycle = build_task_cycle(args, actor_id)

    logger.info("=" * 80)
    logger.info("Starting main generation loop...")
    logger.info("=" * 80)

    step = 0
    generation_in_progress = False
    while True:
        step += 1
        step_start_time = time.time()

        # Optionally reload weights if learner updated them
        # CRITICAL: Only reload when NOT generating/evaluating to avoid OOM
        if args.weights_dir and not generation_in_progress:
            # Additional safety check: ensure we have enough GPU memory before reloading
            if device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(device)
                total = torch.cuda.get_device_properties(device).total_memory
                free_memory_gb = (total - allocated) / (1024**3)
                # Need at least 11GB free to safely reload (8GB model + 2GB transfer overhead + 1GB buffer)
                required_for_reload = 11.0
                if free_memory_gb < required_for_reload:
                    logger.warning(
                        f"[Actor {actor_id} | Step {step}] Skipping model reload - insufficient GPU memory "
                        f"({free_memory_gb:.2f} GB free, need {required_for_reload:.2f} GB)"
                    )
                else:
                    try:
                        model, last_reload_time = maybe_reload_model(model, tokenizer, args, device, last_reload_time)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "Insufficient GPU memory" in str(e):
                            logger.error(f"[Actor {actor_id} | Step {step}] Model reload failed due to OOM: {e}. Continuing with old model.")
                            # Continue with old model - don't crash
                        else:
                            raise
                    except Exception as e:
                        # Catch any other errors during reload and continue with old model
                        logger.error(f"[Actor {actor_id} | Step {step}] Model reload failed: {e}. Continuing with old model.")
                        # Don't update last_reload_time so we'll retry next time
            else:
                model, last_reload_time = maybe_reload_model(model, tokenizer, args, device, last_reload_time)

        log_with_color("32;1", f"[ACTOR {actor_id}] Model is loaded and ready to use")

        batch_size = args.batch_size
        group_size = args.group_size

        # Per-actor deterministic task pattern
        task_labels = [next(task_cycle) for _ in range(batch_size)]
        log_with_color("33;1", f"[Actor {actor_id} | Step {step}] Task labels: {task_labels}")

        instances = []
        prompts = []

        # 1) Generate challenges/prompts
        prompt_gen_start = time.time()
        for i, task_name in enumerate(task_labels):
            if task_name == "ABD":
                task_obj = abd_task
            elif task_name == "DED":
                task_obj = ded_task
            elif task_name == "SAT":
                task_obj = sat_task
            else:
                raise ValueError(f"Unknown task name: {task_name}")

            try:
                challenge = loop.run_until_complete(
                    asyncio.wait_for(task_obj.generate(), timeout=30.0)
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[Actor {actor_id} | Step {step}] {task_name}.generate() timed out (30s), retrying with 60s timeout..."
                )
                challenge = loop.run_until_complete(
                    asyncio.wait_for(task_obj.generate(), timeout=60.0)
                )
                logger.info(
                    f"[Actor {actor_id} | Step {step}] {task_name}.generate() succeeded on retry"
                )

            prompts.append(challenge.prompt)
            instances.append((task_name, task_obj, challenge))

        prompt_gen_time = time.time() - prompt_gen_start
        log_with_color("33;1", f"[Actor {actor_id} | Step {step}] Generated {len(prompts)} prompts in {prompt_gen_time:.2f}s")

        # 2) Tokenize prompts
        tokenize_start = time.time()
        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        tokenize_time = time.time() - tokenize_start

        prompt_tokens = input_ids.numel()
        logger.info(
            f"[Actor {actor_id} | Step {step}] Tokenized: {prompt_tokens} total tokens, "
            f"avg {prompt_tokens // len(prompts)} per prompt ({tokenize_time:.3f}s), "
            f"input_ids device={input_ids.device}, model device={next(model.parameters()).device}"
        )

        # Repeat prompts group_size times for generation
        input_ids_rep = input_ids.repeat_interleave(group_size, dim=0)
        attention_mask_rep = attention_mask.repeat_interleave(group_size, dim=0)
        total_sequences = input_ids_rep.shape[0]
        logger.info(
            f"[Actor {actor_id} | Step {step}] Expanded to {total_sequences} sequences "
            f"(batch_size={batch_size} × group_size={group_size}), shapes: input_ids={input_ids_rep.shape}, attn_mask={attention_mask_rep.shape}"
        )

        # 3) Generate sequences
        gen_start = time.time()
        log_with_color("33;1", f"[Actor {actor_id} | Step {step}] Starting generation: {total_sequences} sequences, device={device}")
        
        # Mark generation as in progress to prevent model reload during generation
        generation_in_progress = True
        
        # Ensure CUDA is synchronized before generation to avoid hangs
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            log_with_color("33;1", f"[Actor {actor_id} | Step {step}] CUDA synchronized, starting model.generate()...")
        
        try:
            with torch.no_grad():
                sequences = model.generate(
                    input_ids=input_ids_rep,
                    attention_mask=attention_mask_rep,
                    max_new_tokens=args.max_completion_length,
                    min_new_tokens=1,
                    do_sample=not args.no_sampling,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"[Actor {actor_id} | Step {step}] CUDA OOM during generation! Clearing cache...")
                torch.cuda.empty_cache()
                raise
            else:
                logger.error(f"[Actor {actor_id} | Step {step}] RuntimeError during generation: {e}", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"[Actor {actor_id} | Step {step}] Generation failed: {e}", exc_info=True)
            generation_in_progress = False  # Clear flag on error
            raise
        
        gen_time = time.time() - gen_start
        log_with_color("33;1", f"[Actor {actor_id} | Step {step}] Generation completed in {gen_time:.2f}s")
        
        # Keep flag True during evaluation - model reload should wait until entire step completes

        generated_tokens = sequences.numel() - input_ids_rep.numel()
        tokens_per_sec = generated_tokens / gen_time if gen_time > 0 else 0
        logger.info(
            f"[Actor {actor_id} | Step {step}] Generated {generated_tokens} tokens in {gen_time:.2f}s "
            f"({tokens_per_sec:.1f} tok/s)"
        )

        decode_start = time.time()
        completions = decode_generations(
            tokenizer=tokenizer,
            sequences=sequences,
            prompts=prompts,
            group_size=group_size,
        )
        decode_time = time.time() - decode_start
        logger.debug(
            f"[Actor {actor_id} | Step {step}] Decoded {len(completions)} completions ({decode_time:.3f}s)"
        )

        # 4) Compute rewards & push groups to local replay buffer
        num_sequences = len(completions)
        assert num_sequences == batch_size * group_size

        eval_start = time.time()
        idx = 0
        reward_stats = {}  # per-task stats

        for b in range(batch_size):
            task_name, task_obj, challenge = instances[b]
            group_comps = completions[idx : idx + group_size]
            idx += group_size

            rewards = []
            for comp_idx, comp in enumerate(group_comps):
                try:
                    r = loop.run_until_complete(
                        asyncio.wait_for(task_obj.evaluate(comp, challenge), timeout=60.0)
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[Actor {actor_id} | Step {step}] {task_name}.evaluate() timeout "
                        f"(comp {comp_idx+1}/{group_size}) -> reward=0.0"
                    )
                    r = 0.0
                except Exception as e:
                    logger.warning(
                        f"[Actor {actor_id} | Step {step}] {task_name}.evaluate() error "
                        f"(comp {comp_idx+1}/{group_size}): {e} -> reward=0.0"
                    )
                    r = 0.0

                rewards.append(float(r))
                reward_stats.setdefault(task_name, []).append(float(r))

            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            max_reward = max(rewards) if rewards else 0.0
            logger.info(
                f"[Actor {actor_id} | Step {step}] Task={task_name:3s} | "
                f"Rewards: {[f'{x:.3f}' for x in rewards]} | "
                f"Avg={avg_reward:.3f} Max={max_reward:.3f}"
            )

            item = {
                "actor_id": actor_id,
                "global_step": step,
                "task_name": task_name,
                "prompt": prompts[b],
                "completions": group_comps,
                "rewards": rewards,
            }
            buffer.push(item)

        eval_time = time.time() - eval_start
        step_time = time.time() - step_start_time

        # Clear generation flag AFTER evaluation completes - now safe to reload model
        generation_in_progress = False

        # Update totals
        total_prompts += batch_size
        total_completions += num_sequences
        elapsed_time = time.time() - start_time
        prompts_per_sec = total_prompts / elapsed_time if elapsed_time > 0 else 0

        # Log step summary
        task_summaries = []
        for tname, vals in reward_stats.items():
            if vals:
                avg_t = sum(vals) / len(vals)
                task_summaries.append(f"{tname}_avg={avg_t:.3f} (n={len(vals)})")
        task_summary_str = ", ".join(task_summaries)

        logger.info(
            f"[Actor {actor_id} | Step {step}] SUMMARY | "
            f"Time: {step_time:.2f}s (gen={gen_time:.2f}s, eval={eval_time:.2f}s) | "
            f"{task_summary_str} | "
            f"Total: {total_prompts} prompts, {prompts_per_sec:.2f} prompts/s"
        )

        if step % 10 == 0:
            logger.info("=" * 80)
            logger.info(f"[Actor {actor_id}] PERIODIC SUMMARY (Step {step})")
            logger.info(f"  Total runtime: {elapsed_time/60:.1f} minutes")
            logger.info(f"  Total prompts processed: {total_prompts}")
            logger.info(f"  Total completions: {total_completions}")
            logger.info(f"  Average prompts/second: {prompts_per_sec:.2f}")
            logger.info(f"  Average step time: {elapsed_time/step:.2f}s")
            logger.info("=" * 80)

        time.sleep(0.05)


if __name__ == "__main__":
    main()
