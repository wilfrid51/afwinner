# This script merges a LoRA-trained checkpoint with the base model into a single model file for upload to HuggingFace

import os
import sys
import tempfile
import subprocess

def ask(question):
    try:
        return input(question).strip()
    except EOFError:
        return ''

def check_python_dependencies():
    try:
        import torch
        import peft
    except ImportError:
        print("You need the packages 'torch' and 'peft' installed in your Python environment.")
        print("Install with: pip install torch peft")
        sys.exit(1)

def main():
    print("=== Merge LoRA Checkpoint Script ===")
    # baseModelPath = ask("Enter the path to the base HF model (folder or .bin): ")
    baseModelPath = "caphe/Affine_top1"
    # loraCheckpointPath = ask("Enter the path to the LoRA checkpoint: ")
    loraCheckpointPath = "AL/output/step-600"
    # outputPath = ask("Enter the output folder (for merged model): ")
    outputPath = "merged_model"

    check_python_dependencies()

    mergeScript = f'''
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = r"{baseModelPath}"
lora_path = r"{loraCheckpointPath}"
output_dir = r"{outputPath}"

print(f"Loading base model: {{base_model_id}}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print(f"Loading LoRA weights from: {{lora_path}}")
peft_config = PeftConfig.from_pretrained(lora_path)
model = PeftModel.from_pretrained(model, lora_path)

print("Merging LoRA with base model...")
model = model.merge_and_unload()

print("Converting merged model to bfloat16...")
model = model.to(torch.bfloat16)

print(f"Saving merged model to {{output_dir}} in bfloat16...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")
'''

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as f:
        tmpScriptPath = f.name
        f.write(mergeScript)

    try:
        subprocess.check_call([sys.executable, tmpScriptPath])
    except subprocess.CalledProcessError:
        print("An error occurred when running the merge script.")
        sys.exit(1)
    finally:
        os.unlink(tmpScriptPath)

    print("The merged model is ready for upload to HuggingFace.")

if __name__ == "__main__":
    main()
