"""Parsing utilities for SPRO."""

import re
from typing import List, Tuple, Dict


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


def find_reasoning_token_boundaries(
    tokenizer,
    token_ids,
    response_start_idx: int,
) -> Tuple[int, int]:
    """
    Find token boundaries for the reasoning section (<think>...</think>).

    Args:
        tokenizer: The tokenizer used
        token_ids: Full sequence token IDs
        response_start_idx: Where the response begins in tokens

    Returns:
        Tuple of (start_token_idx, end_token_idx) for reasoning section
    """
    import torch

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # Decode the full response
    response_tokens = token_ids[response_start_idx:]
    response_text = tokenizer.decode(response_tokens)

    # Build character-to-token mapping
    char_to_token = []
    for tok_idx, tok_id in enumerate(response_tokens):
        tok_text = tokenizer.decode([tok_id])
        for _ in range(len(tok_text)):
            char_to_token.append(tok_idx)

    # Find </think> tag (reasoning ends there)
    think_end_match = re.search(r"</think>", response_text, re.IGNORECASE)

    if think_end_match:
        # Reasoning starts at beginning of response
        start_char = 0
        end_char = think_end_match.end()

        if end_char < len(char_to_token):
            start_tok = response_start_idx  # Reasoning starts at response start
            end_tok = response_start_idx + char_to_token[min(end_char, len(char_to_token) - 1)]
            return (start_tok, end_tok)

    # No thinking section found, return empty
    return (response_start_idx, response_start_idx)


def find_query_token_boundaries(
    tokenizer,
    token_ids,
    response_start_idx: int,
    num_queries: int,
) -> List[Tuple[int, int]]:
    """
    Find token boundaries for each numbered query (1., 2., 3., etc.).

    Searches for patterns like "1.", "2." in the decoded text and maps
    them back to token positions.

    Args:
        tokenizer: The tokenizer used
        token_ids: Full sequence token IDs (can be tensor or list)
        response_start_idx: Where the response begins in tokens
        num_queries: Number of queries to find (from parsed prompts)

    Returns:
        List of (start_token_idx, end_token_idx) for each query (absolute positions)
    """
    import torch

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # Decode the full response
    response_tokens = token_ids[response_start_idx:]
    response_text = tokenizer.decode(response_tokens)

    # Build character-to-token mapping
    char_to_token = []
    current_char = 0

    for tok_idx, tok_id in enumerate(response_tokens):
        tok_text = tokenizer.decode([tok_id])
        for _ in range(len(tok_text)):
            char_to_token.append(tok_idx)
        current_char += len(tok_text)

    # Find each query marker "N." in the text
    query_boundaries = []

    for q in range(1, num_queries + 1):
        # Look for "N." pattern (with possible whitespace before)
        pattern = rf"(?:^|\n)\s*{q}\."
        match = re.search(pattern, response_text)

        if match:
            # Start is where the number begins
            start_char = match.end() - 1  # After the "N."

            # End is either the next query marker or end of text
            if q < num_queries:
                next_pattern = rf"(?:^|\n)\s*{q+1}\."
                next_match = re.search(next_pattern, response_text)
                end_char = next_match.start() if next_match else len(response_text) - 1
            else:
                end_char = len(response_text) - 1

            # Map to token indices (with bounds checking)
            if start_char < len(char_to_token) and end_char < len(char_to_token):
                start_tok = char_to_token[min(start_char, len(char_to_token) - 1)]
                end_tok = char_to_token[min(end_char, len(char_to_token) - 1)]

                # Convert to absolute positions
                query_boundaries.append((
                    response_start_idx + start_tok,
                    response_start_idx + end_tok
                ))

    return query_boundaries
