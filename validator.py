import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging

# Import from AL.env
import sys
import os
# Add current directory to path to import from AL.env
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from AL.env import ABDTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def evaluate_model_on_abd(
    model_name: str,
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
):
    """
    Evaluate a causal language model on ABD (Algorithm By Deduction) tasks.
    
    Args:
        model_name: Path or name of the model to evaluate
        num_samples: Number of ABD challenges to evaluate
        device: Device to run evaluation on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling (True) or greedy decoding (False)
    """
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    model.eval()
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Model loaded on {device}")
    
    # Initialize ABD task
    logger.info("Initializing ABD task...")
    loop = asyncio.get_event_loop()
    if not loop.is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    async def init_task():
        return ABDTask()
    
    task = await init_task()
    logger.info("ABD task initialized")
    
    # Evaluation loop
    correct = 0
    total = 0
    
    logger.info(f"Starting evaluation on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                # Generate challenge
                challenge = await asyncio.wait_for(task.generate(), timeout=30.0)
                prompt = challenge.prompt
                
                # Tokenize prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(device)
                
                input_length = inputs["input_ids"].shape[1]
                
                # Generate response
                generate_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs.get("attention_mask"),
                    "max_new_tokens": max_new_tokens,
                    "min_new_tokens": 1,  # Ensure at least 1 token is generated
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                # Add top_p for sampling if using sampling
                if do_sample:
                    generate_kwargs["top_p"] = 0.95
                
                outputs = model.generate(**generate_kwargs)
                
                # Decode full sequence
                full_text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                # Extract completion: decode only the new tokens
                generated_tokens = outputs[0][input_length:]
                num_generated_tokens = len(generated_tokens)
                
                # Check if model actually generated anything
                if num_generated_tokens == 0:
                    logger.warning(f"No tokens generated on sample {i+1} - model output same length as input")
                    response = ""
                else:
                    response = tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ).strip()
                    
                    # If response is empty but we have tokens, try without skip_special_tokens
                    if not response and num_generated_tokens > 0:
                        response = tokenizer.decode(
                            generated_tokens,
                            skip_special_tokens=False,
                        ).strip()
                        # Remove special tokens manually if needed
                        response = response.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip()
                
                # Debug: log generation details
                if (i + 1) % 10 == 0 or not response:
                    logger.info(f"Sample {i+1}: Generated {num_generated_tokens} tokens, response length: {len(response)}")
                    if not response:
                        logger.warning(f"Empty response! Input length: {input_length}, Output length: {len(outputs[0])}")
                        logger.debug(f"Generated tokens: {generated_tokens[:10] if len(generated_tokens) > 0 else 'None'}")
                        logger.debug(f"Full text (first 200 chars): {full_text[:200]}")
                
                # Evaluate response
                score = await task.evaluate(response, challenge)
                
                if score > 0:
                    correct += 1
                total += 1
                
                if (i + 1) % 10 == 0:
                    accuracy = correct / total
                    # print("=" * 40)
                    # print("Prompt:\n", prompt)
                    # print("-" * 20)
                    # print("Response:\n", response)
                    # print("=" * 40)
                    logger.info(
                        f"Progress: {i+1}/{num_samples} | "
                        f"Correct: {correct} | "
                        f"Accuracy: {accuracy:.4f}"
                    )
                    
            except asyncio.TimeoutError:
                logger.warning(f"Sample {i+1} timed out, skipping...")
                continue
            except Exception as e:
                logger.error(f"Error on sample {i+1}: {e}")
                continue
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info("=" * 60)
    logger.info(f"Final Results:")
    logger.info(f"  Total samples: {total}")
    logger.info(f"  Correct: {correct}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info("=" * 60)
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Validate model on ABD tasks")
    parser.add_argument(
        "--model_name",
        type=str,
        default="caphe/Affine_top1",
        # default="merged_model",
        help="Path or name of the model to evaluate",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=250,
        help="Number of samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    
    args = parser.parse_args()
    
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    asyncio.run(
        evaluate_model_on_abd(
            model_name=args.model_name,
            num_samples=args.num_samples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=not args.greedy,
        )
    )


if __name__ == "__main__":
    main()
