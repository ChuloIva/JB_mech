"""
Unified LLM client.

All LLM calls go through this single interface.
"""
import os
from typing import Optional
from abc import ABC, abstractmethod

import requests


class LLMClient:
    """Simple LLM client for OpenRouter."""

    def __init__(
        self,
        model: str = "anthropic/claude-3-5-sonnet",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(
        self,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        """Generate a response."""
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": all_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class LLMBackend(ABC):
    """Abstract backend for different LLM providers."""

    @abstractmethod
    def generate(self, messages: list[dict], system: str = "", **kwargs) -> str:
        pass


class OpenRouterBackend(LLMBackend):
    """OpenRouter API backend."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.client = LLMClient(model=model, api_key=api_key)
        self.model = model

    def generate(
        self,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        return self.client.generate(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class TransformersBackend(LLMBackend):
    """Local transformers backend."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._loaded = True

    def generate(
        self,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        import torch
        self._ensure_loaded()

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

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
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def create_backend(config: dict) -> LLMBackend:
    """Factory function to create appropriate backend."""
    backend_type = config.get("backend", "openrouter")
    model = config["model"]

    if backend_type == "openrouter":
        return OpenRouterBackend(model, config.get("api_key"))
    elif backend_type == "transformers":
        return TransformersBackend(model, config.get("device"))
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
