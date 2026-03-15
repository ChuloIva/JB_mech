"""Parsing utilities for SPRO."""

import re
from typing import List, Tuple


def parse_sema_output(output: str) -> Tuple[str, List[str]]:
    """
    Parse SEMA output to extract thinking and numbered prompts.

    Args:
        output: Raw model output containing <think>...</think> and numbered prompts

    Returns:
        Tuple of (thinking_text, list_of_prompts)
    """
    thinking = ""
    prompts = []

    # Extract thinking section
    think_match = re.search(r"(.*?)</think>", output, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        after_think = output[think_match.end():].strip()
    else:
        after_think = output.strip()

    # Extract numbered prompts
    prompt_pattern = r"(\d+)\.\s*(.+?)(?=\n\d+\.|$)"
    matches = re.findall(prompt_pattern, after_think, re.DOTALL)

    if matches:
        prompts = [m[1].strip() for m in matches]

    return thinking, prompts
