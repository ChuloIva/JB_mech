#!/usr/bin/env python3
"""Download and save all jailbreak datasets to data/ directory.

Sources:
- OBLITERATUS built-in prompts (512 pairs across 7 tiers)
- AdvBench (~520 prompts) from walledai/AdvBench
- HarmBench (~510 prompts) from harmbench/behaviors
- Anthropic Red-Team (~2000 prompts) from Anthropic/hh-rlhf
- WildJailbreak (~2000 paired) from allenai/wildjailbreak
- JailbreakBench (100 behaviors) from JailbreakBench/JBB-Behaviors
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add third_party to path for OBLITERATUS imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "OBLITERATUS"))

from datasets import load_dataset
from huggingface_hub import login
import pandas as pd

# Output directory
DATA_DIR = PROJECT_ROOT / "data" / "jailbreak_datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {DATA_DIR}")

# Login to HuggingFace
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged into HuggingFace")
else:
    print("Warning: No HF_TOKEN found, some datasets may require authentication")


def save_dataset(name: str, data: list[dict], description: str = ""):
    """Save dataset as JSON and JSONL."""
    json_path = DATA_DIR / f"{name}.json"
    jsonl_path = DATA_DIR / f"{name}.jsonl"

    # Save as JSON
    with open(json_path, "w") as f:
        json.dump({"description": description, "count": len(data), "data": data}, f, indent=2)

    # Save as JSONL
    with open(jsonl_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"  Saved {len(data)} examples to {json_path.name} and {jsonl_path.name}")


# ============================================
# 1. Extract OBLITERATUS Built-in Prompts
# ============================================
print("\n" + "="*60)
print("1. Extracting OBLITERATUS Built-in Prompts (512 pairs)")
print("="*60)

try:
    from obliteratus.prompts import BUILTIN_HARMFUL, BUILTIN_HARMLESS, _HARMLESS_POOL

    obliteratus_data = []
    tier_ranges = [
        (0, 33, "Tier 1: Standard"),
        (33, 66, "Tier 2: Elevated"),
        (66, 99, "Tier 3: Maximum"),
        (99, 199, "Tier 4: Cyber & Digital"),
        (199, 299, "Tier 5: Weapons & Violence"),
        (299, 399, "Tier 6: Fraud & Manipulation"),
        (399, 512, "Tier 7: CBRN & Critical Infrastructure"),
    ]

    for i, (harmful, harmless) in enumerate(zip(BUILTIN_HARMFUL, BUILTIN_HARMLESS)):
        tier = "Unknown"
        for start, end, tier_name in tier_ranges:
            if start <= i < end:
                tier = tier_name
                break

        obliteratus_data.append({
            "id": i,
            "harmful": harmful,
            "harmless": harmless,
            "tier": tier,
            "source": "OBLITERATUS"
        })

    save_dataset("obliteratus_builtin", obliteratus_data,
                 "OBLITERATUS built-in 512 harmful/harmless prompt pairs across 7 severity tiers")

    # Also save the harmless pool separately
    harmless_pool = [{"id": i, "prompt": p} for i, p in enumerate(_HARMLESS_POOL)]
    save_dataset("obliteratus_harmless_pool", harmless_pool,
                 "OBLITERATUS harmless prompt pool for contrastive pairs")

except Exception as e:
    print(f"  Error loading OBLITERATUS: {e}")


# ============================================
# 2. Download AdvBench
# ============================================
print("\n" + "="*60)
print("2. Downloading AdvBench (~520 prompts)")
print("="*60)

try:
    ds = load_dataset("walledai/AdvBench", split="train")

    advbench_data = []
    for i, row in enumerate(ds):
        prompt = row.get("prompt") or row.get("goal") or row.get("behavior", "")
        if prompt and len(prompt.strip()) > 10:
            advbench_data.append({
                "id": i,
                "prompt": prompt.strip(),
                "source": "AdvBench"
            })

    save_dataset("advbench", advbench_data,
                 "AdvBench: Universal adversarial attacks benchmark (Zou et al. 2023)")

except Exception as e:
    print(f"  Error loading AdvBench: {e}")


# ============================================
# 3. Download HarmBench
# ============================================
print("\n" + "="*60)
print("3. Downloading HarmBench (~510 prompts)")
print("="*60)

try:
    # Load all configs and combine them
    harmbench_data = []
    for config in ["standard", "contextual", "copyright"]:
        try:
            ds = load_dataset("walledai/HarmBench", config, split="train")
            print(f"    Loaded {len(ds)} examples from '{config}' config")
            for i, row in enumerate(ds):
                prompt = row.get("prompt", "")
                if prompt and len(prompt.strip()) > 10:
                    harmbench_data.append({
                        "id": len(harmbench_data),
                        "prompt": prompt.strip(),
                        "category": row.get("category", config),
                        "config": config,
                        "source": "HarmBench"
                    })
        except Exception as e:
            print(f"    Warning: Could not load HarmBench config '{config}': {e}")

    if harmbench_data:
        save_dataset("harmbench", harmbench_data,
                     "HarmBench: Standardized red-teaming evaluation framework (Mazeika et al. 2024)")
    else:
        print("  No HarmBench data loaded")

except Exception as e:
    print(f"  Error loading HarmBench: {e}")


# ============================================
# 4. Download Anthropic Red-Team
# ============================================
print("\n" + "="*60)
print("4. Downloading Anthropic Red-Team (~2000 prompts)")
print("="*60)

try:
    # Try red-team-attempts split first
    try:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
    except:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")

    anthropic_data = []
    seen = set()

    for row in ds:
        text = row.get("transcript") or row.get("rejected") or row.get("chosen", "")
        if not text:
            continue

        if "Human:" in text:
            parts = text.split("Human:")
            if len(parts) >= 2:
                first_turn = parts[1].split("Assistant:")[0].strip()
                if first_turn and len(first_turn) > 15 and first_turn not in seen:
                    seen.add(first_turn)
                    anthropic_data.append({
                        "id": len(anthropic_data),
                        "prompt": first_turn,
                        "source": "Anthropic-RedTeam"
                    })

        if len(anthropic_data) >= 2000:
            break

    save_dataset("anthropic_redteam", anthropic_data,
                 "Anthropic HH-RLHF red-team attempts - refusal-triggering prompts")

except Exception as e:
    print(f"  Error loading Anthropic Red-Team: {e}")


# ============================================
# 5. Download WildJailbreak
# ============================================
print("\n" + "="*60)
print("5. Downloading WildJailbreak (~2000 paired prompts)")
print("="*60)

try:
    # Use eval config - train has schema issues
    ds = load_dataset("allenai/wildjailbreak", "eval", split="train")
    print(f"    Loaded {len(ds)} examples from WildJailbreak eval")

    wildjailbreak_data = []
    seen = set()

    for row in ds:
        adv = row.get("adversarial") or row.get("jailbreak", "") or row.get("prompt", "")
        van = row.get("vanilla") or row.get("benign", "") or ""

        if adv and len(adv.strip()) > 10 and adv.strip() not in seen:
            seen.add(adv.strip())
            wildjailbreak_data.append({
                "id": len(wildjailbreak_data),
                "adversarial": adv.strip(),
                "vanilla": van.strip() if van else "",
                "source": "WildJailbreak"
            })

        if len(wildjailbreak_data) >= 2000:
            break

    save_dataset("wildjailbreak", wildjailbreak_data,
                 "WildJailbreak: Synthetic jailbreaks with paired benign queries (Jiang et al. 2024)")

except Exception as e:
    print(f"  Error loading WildJailbreak: {e}")


# ============================================
# 6. Download JailbreakBench
# ============================================
print("\n" + "="*60)
print("6. Downloading JailbreakBench (100 behaviors)")
print("="*60)

try:
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")

    jbb_data = []
    for i, row in enumerate(ds):
        jbb_data.append({
            "id": i,
            "goal": row.get("Goal", ""),
            "target": row.get("Target", ""),
            "behavior": row.get("Behavior", ""),
            "category": row.get("Category", ""),
            "source": "JailbreakBench"
        })

    save_dataset("jailbreakbench", jbb_data,
                 "JailbreakBench: 100 harmful behaviors for jailbreak evaluation")

except Exception as e:
    print(f"  Error loading JailbreakBench: {e}")


# ============================================
# 7. Create Combined Dataset
# ============================================
print("\n" + "="*60)
print("7. Creating Combined Dataset")
print("="*60)

try:
    combined = []

    # Load all saved datasets
    for filename in DATA_DIR.glob("*.jsonl"):
        if "combined" in filename.name or "harmless_pool" in filename.name:
            continue

        with open(filename) as f:
            for line in f:
                item = json.loads(line)
                # Normalize to common format
                prompt = (item.get("harmful") or item.get("prompt") or
                         item.get("behavior") or item.get("adversarial") or
                         item.get("goal", ""))
                if prompt:
                    combined.append({
                        "prompt": prompt,
                        "source": item.get("source", filename.stem),
                        "original_id": item.get("id", -1)
                    })

    # Deduplicate by prompt
    seen = set()
    unique_combined = []
    for item in combined:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            item["id"] = len(unique_combined)
            unique_combined.append(item)

    save_dataset("combined_all", unique_combined,
                 f"Combined dataset from all sources, deduplicated ({len(unique_combined)} unique prompts)")

except Exception as e:
    print(f"  Error creating combined dataset: {e}")


# ============================================
# Summary
# ============================================
print("\n" + "="*60)
print("DOWNLOAD COMPLETE - Summary")
print("="*60)

total_size = 0
for filepath in sorted(DATA_DIR.glob("*.json")):
    size = filepath.stat().st_size
    total_size += size
    with open(filepath) as f:
        data = json.load(f)
        count = data.get("count", "?")
    print(f"  {filepath.name:<35} {count:>6} examples  ({size/1024:.1f} KB)")

print(f"\nTotal: {total_size/1024:.1f} KB in {DATA_DIR}")
print(f"\nFiles ready for training in: {DATA_DIR}")
