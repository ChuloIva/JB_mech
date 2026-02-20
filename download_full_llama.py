#!/usr/bin/env python3
"""
Download the full-sized Meta Llama model from Hugging Face using hf_transfer.
"""

import os
from huggingface_hub import snapshot_download

# Enable hf_transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Official Meta Llama repository
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

print(f"Downloading full model from {repo_id}...")
print("Using hf_transfer for faster downloads...")
print("Note: This will download all model files (safetensors, config, tokenizer, etc.)")
print("This may take a while as it's the full-sized model (~16GB)...\n")

# Download the entire repository
local_dir = snapshot_download(
    repo_id=repo_id,
    local_dir=None,  # Download to default cache directory
    local_dir_use_symlinks=False,  # Use actual files instead of symlinks
    ignore_patterns=["*.md", "*.txt"],  # Skip documentation files to save space
)

print(f"\nDownload complete!")
print(f"Model saved to: {local_dir}")
print("\nTo use the model:")
print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
print(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
print(f"  tokenizer = AutoTokenizer.from_pretrained('{repo_id}')")
