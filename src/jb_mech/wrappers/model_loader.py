"""
Shared model loader for managing HuggingFace and VLLM model instances.

This module provides a singleton pattern for loading and managing model instances
to avoid loading the same model multiple times and to handle memory management
between VLLM (for generation) and HuggingFace (for activation extraction).
"""

import gc
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import MODEL_NAME, VLLM_CONFIG


class SharedModelLoader:
    """
    Singleton for managing shared model instances.

    VLLM and HuggingFace models cannot coexist in memory efficiently,
    so this class provides methods to switch between them.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.hf_model = None
        self.hf_tokenizer = None
        self.vllm_model = None
        self._current_mode = None  # "hf" or "vllm"
        self._initialized = True

    def load_hf_model(
        self,
        model_name: str = MODEL_NAME,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load HuggingFace model for activation extraction.

        Args:
            model_name: HuggingFace model identifier
            device: Device specification ("auto", "cuda:0", etc.)
            dtype: Data type for model weights

        Returns:
            Tuple of (model, tokenizer)
        """
        # Unload VLLM if loaded
        if self.vllm_model is not None:
            self.unload_vllm()

        if self.hf_model is None:
            print(f"Loading HuggingFace model: {model_name}")

            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            self.hf_tokenizer.padding_side = "left"

            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
            )
            self.hf_model.eval()
            self._current_mode = "hf"

        return self.hf_model, self.hf_tokenizer

    def load_vllm_model(
        self,
        model_name: str = MODEL_NAME,
        **kwargs,
    ):
        """
        Load VLLM model for fast generation.

        Args:
            model_name: HuggingFace model identifier
            **kwargs: Override VLLM config options

        Returns:
            VLLM LLM instance
        """
        # Unload HuggingFace model if loaded
        if self.hf_model is not None:
            self.unload_hf()

        if self.vllm_model is None:
            from vllm import LLM

            print(f"Loading VLLM model: {model_name}")

            config = VLLM_CONFIG.copy()
            config.update(kwargs)

            self.vllm_model = LLM(model=model_name, **config)
            self._current_mode = "vllm"

        return self.vllm_model

    def unload_hf(self):
        """Free HuggingFace model memory."""
        if self.hf_model is not None:
            print("Unloading HuggingFace model...")
            del self.hf_model
            self.hf_model = None

        if self.hf_tokenizer is not None:
            del self.hf_tokenizer
            self.hf_tokenizer = None

        self._clear_cache()
        self._current_mode = None

    def unload_vllm(self):
        """Free VLLM model memory."""
        if self.vllm_model is not None:
            print("Unloading VLLM model...")
            del self.vllm_model
            self.vllm_model = None

        self._clear_cache()
        self._current_mode = None

    def unload_all(self):
        """Free all model memory."""
        self.unload_hf()
        self.unload_vllm()

    def _clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def current_mode(self) -> Optional[str]:
        """Get the current model mode ('hf', 'vllm', or None)."""
        return self._current_mode

    @property
    def is_hf_loaded(self) -> bool:
        """Check if HuggingFace model is loaded."""
        return self.hf_model is not None

    @property
    def is_vllm_loaded(self) -> bool:
        """Check if VLLM model is loaded."""
        return self.vllm_model is not None

    def get_tokenizer(self, model_name: str = MODEL_NAME) -> AutoTokenizer:
        """
        Get a tokenizer without loading the full model.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Tokenizer instance
        """
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer


def get_model_loader() -> SharedModelLoader:
    """Get the singleton model loader instance."""
    return SharedModelLoader()
