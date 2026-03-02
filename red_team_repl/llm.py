"""
LLM backends: OpenRouter (cloud) and Transformers (local)
"""
import os
from typing import Optional
from abc import ABC, abstractmethod

import requests


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], system: str, **kwargs) -> str:
        pass

    @abstractmethod
    def unload(self):
        pass


class OpenRouterBackend(LLMBackend):
    """OpenRouter API backend for cloud models."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(
        self,
        messages: list[dict],
        system: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        full_messages = [{"role": "system", "content": system}] + messages

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": full_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def unload(self):
        pass  # Nothing to unload for API


class TransformersBackend(LLMBackend):
    """Local transformers backend."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded.")

    def generate(
        self,
        messages: list[dict],
        system: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        import torch

        # Build conversation with system prompt
        full_messages = [{"role": "system", "content": system}] + messages

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self):
        import gc
        import torch

        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMBackend(LLMBackend):
    """vLLM backend for faster local inference."""

    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.85):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.tokenizer = None
        self._load()

    def _load(self):
        from vllm import LLM
        from transformers import AutoTokenizer

        print(f"Loading {self.model_name} with vLLM...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype="bfloat16",
            max_model_len=4096,
        )

        print("Model loaded.")

    def generate(
        self,
        messages: list[dict],
        system: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        from vllm import SamplingParams

        full_messages = [{"role": "system", "content": system}] + messages

        prompt = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def unload(self):
        import gc
        import torch

        if self.llm is not None:
            del self.llm
            self.llm = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_backend(config: dict) -> LLMBackend:
    """Factory function to create the appropriate backend."""
    backend_type = config.get("backend", "openrouter")
    model = config["model"]

    if backend_type == "openrouter":
        return OpenRouterBackend(model, config.get("api_key"))
    elif backend_type == "transformers":
        return TransformersBackend(model, config.get("device"))
    elif backend_type == "vllm":
        return VLLMBackend(model, config.get("gpu_memory_utilization", 0.85))
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
