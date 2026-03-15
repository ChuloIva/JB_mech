"""OpenRouter API client for SPRO training."""

import asyncio
import os
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

from .prompts import JUDGE_PROMPT
from .config import SPROConfig


class OpenRouterClient:
    """Async client for OpenRouter API."""

    def __init__(self, config: SPROConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key,
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, Exception)),
        reraise=True,
    )
    async def query_target(self, prompt: str, context: List[Dict] = None) -> str:
        """
        Query target model with conversation context.

        Args:
            prompt: The user prompt to send
            context: Previous conversation context

        Returns:
            Model response text
        """
        messages = context or []
        messages = messages + [{"role": "user", "content": prompt}]

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.target_model,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    raise  # Retry on rate limit
                return f"[ERROR: {e}]"

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, Exception)),
        reraise=True,
    )
    async def judge_conversation(
        self,
        intent: str,
        conversation: List[Tuple[str, str]],
    ) -> Tuple[int, str]:
        """
        Judge a multi-turn conversation.

        Args:
            intent: Original harmful intent
            conversation: List of (prompt, response) tuples

        Returns:
            Tuple of (score, reasoning)
        """
        # Format conversation
        conv_text = ""
        for i, (prompt, response) in enumerate(conversation, 1):
            conv_text += f"\n--- Turn {i} ---\n"
            conv_text += f"User: {prompt}\n"
            conv_text += f"Assistant: {response}\n"

        judge_input = JUDGE_PROMPT.format(intent=intent, conversation=conv_text)

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.judge_model,
                    messages=[{"role": "user", "content": judge_input}],
                    max_tokens=1000,
                    temperature=0.1,
                )
                reasoning = response.choices[0].message.content
                # Parse score from last line
                score = 2  # Default
                for line in reversed(reasoning.strip().split('\n')):
                    for char in line:
                        if char in '1234':
                            score = int(char)
                            break
                    if score != 2:
                        break
                return score, reasoning
            except Exception as e:
                if "429" in str(e):
                    raise
                return 2, f"[ERROR: {e}]"

    async def test_connection(self) -> str:
        """Test API connection."""
        response = await self.query_target("Say 'API working' and nothing else.")
        return response
