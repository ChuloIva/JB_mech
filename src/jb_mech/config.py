"""
Configuration constants for Jailbreak Mechanism Analysis.

This module contains all model configurations, paths, and constants
for the Llama 3.1 8B jailbreak analysis project.
"""

from pathlib import Path
import sys

# ==============================================================================
# Model Configuration for Llama 3.1 8B
# ==============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TOTAL_LAYERS = 32
TARGET_LAYER = 16  # Middle layer for axis computation
HIDDEN_SIZE = 4096

# Model family detection
MODEL_TYPE = "llama"
SHORT_NAME = "Llama-8B"

# ==============================================================================
# GLP Configuration
# ==============================================================================

GLP_MODEL = "generative-latent-prior/glp-llama8b-d6"
GLP_CHECKPOINT = "final"
GLP_U_TIMESTEP = 0.9  # Standard timestep for meta-neuron extraction

# Tracedict config for baukit activation extraction
TRACEDICT_CONFIG = {
    "layer_prefix": "model.layers",
    "layers": [TARGET_LAYER],
    "retain": "output",
}

# ==============================================================================
# Project Paths
# ==============================================================================

# Detect project root (works whether running from repo or as installed package)
_current_file = Path(__file__).resolve()
PROJECT_ROOT = _current_file.parent.parent.parent.parent

# Third-party dependencies
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
ASSISTANT_AXIS_DIR = THIRD_PARTY_DIR / "assistant-axis"
GLP_DIR = THIRD_PARTY_DIR / "generative_latent_prior"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llama-3.1-8b"
RESPONSES_DIR = OUTPUT_DIR / "responses"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
VECTORS_DIR = OUTPUT_DIR / "vectors"
JAILBREAKS_DIR = OUTPUT_DIR / "jailbreaks"
AXIS_PATH = OUTPUT_DIR / "axis.pt"

# ==============================================================================
# Data Paths
# ==============================================================================

# Assistant Axis data
AA_ROLES_DIR = ASSISTANT_AXIS_DIR / "data" / "roles" / "instructions"
AA_QUESTIONS_FILE = ASSISTANT_AXIS_DIR / "data" / "extraction_questions.jsonl"
AA_TRAITS_DIR = ASSISTANT_AXIS_DIR / "data" / "traits" / "instructions"

# GLP pre-computed persona vectors
GLP_PERSONA_VECTORS_DIR = GLP_DIR / "integrations" / "persona_vectors" / "cached_vectors" / "Llama-3.1-8B-Instruct"
EVIL_PERSONA_PATH = GLP_PERSONA_VECTORS_DIR / "evil_response_avg_diff.pt"
HALLUCINATING_PERSONA_PATH = GLP_PERSONA_VECTORS_DIR / "hallucinating_response_avg_diff.pt"
SYCOPHANTIC_PERSONA_PATH = GLP_PERSONA_VECTORS_DIR / "sycophantic_response_avg_diff.pt"

# ==============================================================================
# Generation Configuration
# ==============================================================================

GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
}

VLLM_CONFIG = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "dtype": "bfloat16",
    "max_model_len": 2048,
}

# ==============================================================================
# Analysis Configuration
# ==============================================================================

# Harmful persona names to focus on
HARMFUL_PERSONAS = [
    "criminal_mastermind",
    "unethical_hacker",
    "manipulative_con_artist",
    "cult_leader",
    "evil_AI",
]

# Helpful persona names for comparison
HELPFUL_PERSONAS = [
    "therapist",
    "teacher",
    "coach",
    "consultant",
    "mentor",
]

# nanoGCG configuration
NANOGCG_CONFIG = {
    "num_steps": 500,
    "search_width": 512,
    "topk": 256,
    "seed": 42,
}

# ==============================================================================
# Helper Functions
# ==============================================================================


def add_third_party_to_path():
    """Add third-party directories to Python path."""
    paths_to_add = [
        str(ASSISTANT_AXIS_DIR),
        str(GLP_DIR),
    ]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [RESPONSES_DIR, ACTIVATIONS_DIR, VECTORS_DIR, JAILBREAKS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_model_config():
    """Get model configuration dictionary."""
    return {
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "total_layers": TOTAL_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "short_name": SHORT_NAME,
        "model_type": MODEL_TYPE,
    }
