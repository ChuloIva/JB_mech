#!/usr/bin/env python3
"""
Download a model from Hugging Face using hf_transfer for faster downloads.
"""

import os
from huggingface_hub import hf_hub_download

# Enable hf_transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
filename = "Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"

print(f"Downloading {filename} from {repo_id}...")
print("Using hf_transfer for faster downloads...")

# Download the file
file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=None,  # Download to default cache directory
    local_dir_use_symlinks=False,  # Use actual files instead of symlinks
)

print(f"\nDownload complete!")
print(f"File saved to: {file_path}")
