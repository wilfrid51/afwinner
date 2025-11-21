import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging

# Import from AL.env
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add current directory to path to import from AL.env
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from AL.env import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def evaluate_model_on_SAT(
    model_name: str,
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = False,
):
    """
    Evaluate a causal language model on SAT (Abstract Benchmarking Dataset) tasks.
    
    Args:
        model_name: Path or name of the model to evaluate
        num_samples: Number of SAT challenges to evaluate
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
    
    # Enable KV cache for faster generation (reduces memory but speeds up generation)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    
    # Optimize model for inference
    if device == "cuda":
        # Enable torch.compile for faster generation (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile for faster inference")
        except Exception as e:
            logger.debug(f"torch.compile not available or failed: {e}")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Model loaded on {device}")
    
    # Initialize SAT task
    logger.info("Initializing SAT task...")
    loop = asyncio.get_event_loop()
    if not loop.is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    async def init_task():
        return SATTask()
    
    task = await init_task()
    logger.info("SAT task initialized")
    
    # Evaluation loop
    correct = 0
    total = 0
    
    logger.info(f"Starting evaluation on {num_samples} samples...")
    
    # Clear GPU cache before starting to free up memory
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before evaluation: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                # Generate challenge
                challenge = await asyncio.wait_for(task.generate(), timeout=30.0)
                # print("Challenge generated")
                prompt = challenge.prompt
                
                # Tokenize prompt with optimized settings
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=False,  # No padding needed for single sequence
                )
                
                # Move to device
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                # print("Prompt tokenized")
                
                input_length = input_ids.shape[1]
                # print(f"Input length: {input_length}")
                
                # Clear cache before generation to free memory
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device)  # Ensure all operations complete
                
                # Generate response with optimizations
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "min_new_tokens": 1,  # Ensure at least 1 token is generated
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": True,  # Enable KV cache for faster generation (reuses computed states)
                }
                
                # Add top_p for sampling if using sampling
                if do_sample:
                    generate_kwargs["top_p"] = 0.95
                    generate_kwargs["temperature"] = temperature
                else:
                    # For greedy decoding, remove temperature (not needed and causes warning)
                    pass
                
                # print("Generating response...")
                import time
                gen_start = time.time()
                outputs = model.generate(**generate_kwargs)
                gen_time = time.time() - gen_start
                num_generated = outputs[0].shape[0] - input_length
                print(f"Response generated: {num_generated} tokens in {gen_time:.2f}s ({num_generated/gen_time:.1f} tok/s)")
                
                # Clear cache after generation to free memory for next iteration
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                # Decode full sequence
                full_text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                # print("full_text finished")
                
                # Extract completion: decode only the new tokens
                generated_tokens = outputs[0][input_length:]
                num_generated_tokens = len(generated_tokens)
                
                # Check if model actually generated anything
                if num_generated_tokens == 0:
                    logger.warning(f"No tokens generated on sample {i+1} - model output same length as input")
                    response = ""
                else:
                    # print("Decoding response...")
                    response = tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ).strip()
                    # print("Response decoded")

                    # If response is empty but we have tokens, try without skip_special_tokens
                    if not response and num_generated_tokens > 0:
                        # print("Decoding response without skip_special_tokens...")
                        response = tokenizer.decode(
                            generated_tokens,
                            skip_special_tokens=False,
                        ).strip()
                        # Remove special tokens manually if needed
                        response = response.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip()
                        # print("Response decoded without skip_special_tokens")
                # Debug: log generation details
                if (i + 1) % 10 == 0 or not response:
                    logger.info(f"Sample {i+1}: Generated {num_generated_tokens} tokens, response length: {len(response)}")
                    if not response:
                        logger.warning(f"Empty response! Input length: {input_length}, Output length: {len(outputs[0])}")
                        logger.debug(f"Generated tokens: {generated_tokens[:10] if len(generated_tokens) > 0 else 'None'}")
                        logger.debug(f"Full text (first 200 chars): {full_text[:200]}")
                
                # Evaluate response
                print("Evaluating response...")
                print(f"{"="*30} Challenge {"="*30}")
                print(f"{challenge.prompt}")
                print(f"{"="*30} Response {"="*30}")
                print(f"{response}")
                print(f"{"="*30} End of Response {"="*30}")
                score = await task.evaluate(response, challenge)
                print("Response evaluated")

                print("Score: ", score)
                if score > 0:
                    # print("Correct incremented")
                    correct += 1
                total += 1
                # print("Total incremented")
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
    parser = argparse.ArgumentParser(description="Validate model on SAT tasks")
    parser.add_argument(
        "--model_name",
        type=str,
        # default="merged_model-50",
        default="caphe/Affine_top1",
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
        default=True,
        help="Use greedy decoding instead of sampling",
    )
    
    args = parser.parse_args()
    
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    asyncio.run(
        evaluate_model_on_SAT(
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
