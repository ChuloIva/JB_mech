"""
Dataset loading utilities for jailbreak analysis.

Provides functions to load:
- JailbreakBench: Curated harmful behaviors with proven jailbreak prompts
- AdvBench: 520 harmful behaviors for testing
- Alpaca: Benign instruction-following prompts for baseline
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import load_dataset


class DatasetLoader:
    """Load and manage datasets for jailbreak analysis."""

    @staticmethod
    def load_jailbreakbench(
        split: str = "behaviors",
        max_samples: Optional[int] = None,
    ) -> List[Dict]:
        """
        Load JailbreakBench behaviors dataset.

        Args:
            split: Dataset split ("behaviors" for harmful behaviors)
            max_samples: Maximum number of samples to load

        Returns:
            List of behavior dictionaries with keys: goal, target, id
        """
        print("Loading JailbreakBench behaviors...")

        ds = load_dataset("JailbreakBench/JBB-Behaviors", split)

        behaviors = []
        for i, item in enumerate(ds["train"]):
            if max_samples and i >= max_samples:
                break

            behaviors.append({
                "id": i,
                "goal": item["Goal"],
                "target": item["Target"],
                "category": item.get("Category", "unknown"),
            })

        print(f"Loaded {len(behaviors)} behaviors from JailbreakBench")
        return behaviors

    @staticmethod
    def load_advbench(
        max_samples: Optional[int] = None,
        default_target: str = "Sure, here's",
    ) -> List[Dict]:
        """
        Load AdvBench harmful behaviors dataset.

        Args:
            max_samples: Maximum number of samples to load
            default_target: Default target string for jailbreak generation

        Returns:
            List of behavior dictionaries with keys: goal, target, id
        """
        print("Loading AdvBench...")

        try:
            ds = load_dataset("walledai/AdvBench")
            source = "walledai/AdvBench"
        except Exception:
            # Fallback to alternative source
            ds = load_dataset("llm-attacks/advbench")
            source = "llm-attacks/advbench"

        behaviors = []
        for i, item in enumerate(ds["train"]):
            if max_samples and i >= max_samples:
                break

            # Handle different column names
            if "prompt" in item:
                goal = item["prompt"]
            elif "goal" in item:
                goal = item["goal"]
            else:
                goal = str(item)

            behaviors.append({
                "id": i,
                "goal": goal,
                "target": default_target,
            })

        print(f"Loaded {len(behaviors)} behaviors from {source}")
        return behaviors

    @staticmethod
    def load_alpaca(
        max_samples: Optional[int] = None,
        filter_empty: bool = True,
    ) -> List[Dict]:
        """
        Load Alpaca dataset for benign baseline prompts.

        Args:
            max_samples: Maximum number of samples to load
            filter_empty: Whether to filter out empty instructions

        Returns:
            List of instruction dictionaries with keys: instruction, input, output, id
        """
        print("Loading Alpaca...")

        ds = load_dataset("tatsu-lab/alpaca")

        instructions = []
        for i, item in enumerate(ds["train"]):
            if max_samples and len(instructions) >= max_samples:
                break

            instruction = item["instruction"]
            if filter_empty and not instruction.strip():
                continue

            instructions.append({
                "id": i,
                "instruction": instruction,
                "input": item.get("input", ""),
                "output": item.get("output", ""),
            })

        print(f"Loaded {len(instructions)} instructions from Alpaca")
        return instructions

    @staticmethod
    def load_fineweb(
        max_samples: int = 1000,
        min_length: int = 100,
    ) -> List[str]:
        """
        Load FineWeb samples for neuron interpretation.

        Args:
            max_samples: Maximum number of samples to load
            min_length: Minimum text length to include

        Returns:
            List of text samples
        """
        print("Loading FineWeb samples...")

        ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True)

        texts = []
        for item in ds["train"]:
            if len(texts) >= max_samples:
                break

            text = item["text"]
            if len(text) >= min_length:
                # Truncate long texts
                texts.append(text[:2000])

        print(f"Loaded {len(texts)} samples from FineWeb")
        return texts

    @staticmethod
    def load_proven_jailbreaks(
        path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Load pre-verified successful jailbreak prompts from a CSV file.

        Args:
            path: Path to the CSV file

        Returns:
            DataFrame with jailbreak data
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Jailbreak file not found: {path}")

        df = pd.read_csv(path)
        print(f"Loaded {len(df)} proven jailbreaks from {path}")
        return df

    @staticmethod
    def create_prompt_pairs(
        behaviors: List[Dict],
        jailbreak_suffix: str = "",
    ) -> Dict[str, List[str]]:
        """
        Create paired clean and jailbroken prompts from behaviors.

        Args:
            behaviors: List of behavior dictionaries
            jailbreak_suffix: Suffix to append for jailbreak prompts

        Returns:
            Dictionary with 'clean' and 'jailbreak' lists of prompts
        """
        clean_prompts = [b["goal"] for b in behaviors]
        jailbreak_prompts = [f"{b['goal']} {jailbreak_suffix}" for b in behaviors]

        return {
            "clean": clean_prompts,
            "jailbreak": jailbreak_prompts,
        }

    @staticmethod
    def get_benign_prompts(
        instructions: List[Dict],
        include_input: bool = False,
    ) -> List[str]:
        """
        Extract prompts from Alpaca instructions.

        Args:
            instructions: List of Alpaca instruction dictionaries
            include_input: Whether to include the input field

        Returns:
            List of prompts
        """
        prompts = []
        for inst in instructions:
            prompt = inst["instruction"]
            if include_input and inst.get("input"):
                prompt = f"{prompt}\n\n{inst['input']}"
            prompts.append(prompt)

        return prompts
