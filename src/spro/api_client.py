"""OpenRouter API client for SPRO training."""

import asyncio
import json
import os
import re
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

from .prompts import JUDGE_PROMPT
from .config import SPROConfig
from .data_structures import IDAScores


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

    def _parse_ida_json(self, text: str) -> IDAScores:
        """Parse IDA scores from judge response JSON."""
        # Try to extract JSON from response
        try:
            # First try direct parse
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Return default scores on parse failure
                    return IDAScores(
                        alignment=0, risk=0, detail=0,
                        rationale=f"[JSON parse error: {text[:200]}]"
                    )
            else:
                return IDAScores(
                    alignment=0, risk=0, detail=0,
                    rationale=f"[No JSON found: {text[:200]}]"
                )

        # Extract scores with defaults
        return IDAScores(
            alignment=int(data.get("intent_alignment_score", 0)),
            risk=int(data.get("compliance_risk_score", 0)),
            detail=int(data.get("detail_level_score", 0)),
            rationale=data.get("rationale", ""),
        )

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
    ) -> IDAScores:
        """
        Judge a multi-turn conversation using IDA scoring.

        Scores only the FINAL response against the original intent.

        Args:
            intent: Original harmful intent (the request)
            conversation: List of (prompt, response) tuples

        Returns:
            IDAScores with alignment, risk, detail scores
        """
        if not conversation:
            return IDAScores(alignment=0, risk=0, detail=0, rationale="Empty conversation")

        # Get the final response only
        final_response = conversation[-1][1]

        # Format judge input
        judge_input = JUDGE_PROMPT.format(
            request=intent,
            response=final_response
        )

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.judge_model,
                    messages=[{"role": "user", "content": judge_input}],
                    max_tokens=500,
                    temperature=0.1,
                )
                raw_response = response.choices[0].message.content
                return self._parse_ida_json(raw_response)
            except Exception as e:
                if "429" in str(e):
                    raise
                return IDAScores(
                    alignment=0, risk=0, detail=0,
                    rationale=f"[ERROR: {e}]"
                )

    async def test_connection(self) -> str:
        """Test API connection."""
        response = await self.query_target("Say 'API working' and nothing else.")
        return response
