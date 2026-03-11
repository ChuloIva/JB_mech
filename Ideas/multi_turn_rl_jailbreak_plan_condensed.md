# Multi-Turn RL Jailbreak Training Plan (Condensed)

## Overview

Train a multi-turn jailbreak attacker using RL, combining techniques from **Jailbreak-R1**, **SEMA**, and **RL-MTJail**. The system trains Qwen 3 models (4B/12B) to execute strategic multi-turn attacks using native `<think>...</think>` reasoning.

**Core Architecture:**
- **Attacker Model**: Qwen3-4B/12B with LoRA, generates `<think>` reasoning + jailbreak prompts
- **Target Model**: Victim LLM (Llama-3.1-8B, GPT-4.1-mini, etc.)
- **Reward System**: Process rewards (per-turn) + Intent-drift penalty (SEMA) + Outcome reward (judge score 0-3)
- **Training**: GRPO with turn-level credit assignment

---

## Distributed Architecture (Recommended)

| Component | Hardware | Purpose | vLLM Usage |
|-----------|----------|---------|------------|
| **Colab** | A100 40GB | Attacker training + rollouts | vLLM for attacker generation |
| **RunPod** | RTX 4090 / A6000 | Target & Judge inference | vLLM API servers |

**Benefits:**
- 40% cheaper than single-node
- vLLM everywhere: 10-20x faster inference
- Weight syncing keeps attacker vLLM up-to-date with training

---

## Training Pipeline (3 Stages)

### Stage A: Cold Start SFT
**Goal:** Teach the attacker strategic reasoning format using `<think>` tags.

**Data Sources:**
1. Anthropic alignment faking transcripts
2. Reward hacking demonstrations
3. Multi-turn jailbreak traces
4. Synthetic misalignment reasoning

**Key:** Use `tokenizer.apply_chat_template()` for proper formatting.

### Stage B: Warm-up GRPO
**Goal:** Stabilize RL with single-turn jailbreaks.

- Single-turn attacks only
- TRL's GRPOTrainer with **vLLM for fast rollouts**
- Group size G=8, KL penalty β=0.01
- Target: ASR > 40%

### Stage C: Multi-Turn GRPO (Main Training)
**Goal:** Train strategic multi-turn conversations.

**Environment (MDP):**
- State: (harmful_intent, conversation_history)
- Action: (thinking, prompt) pair
- Max turns: 8

**vLLM Integration:**
- **Attacker rollouts**: vLLM with weight syncing after each training step
- **Target/Judge**: vLLM via RunPod API (distributed setup)
- LoRA adapters supported via `LoRARequest`

**Reward Components:**
| Component | Weight | Description |
|-----------|--------|-------------|
| Process Reward | 0.2 | Refusal penalty + relevance score |
| Intent Drift | 0.3 | Cosine similarity to original goal |
| Outcome | 0.5 | Final judge score (0-3 normalized) |

**Training Loop:**
1. **Sync weights to vLLM** (if dirty from last update)
2. Sample intents, collect G parallel episodes using vLLM
3. **Save rollouts to disk** (for DPO/SFT later)
4. Compute group-relative advantages
5. GRPO policy update with KL penalty
6. **Mark weights dirty** for next sync

---

## Repository Structure

```
JB_mech/
├── src/jb_mech/multi_turn_rl/
│   ├── models/          # attacker, target_api, judge_api
│   ├── rewards/         # process_reward, intent_drift, outcome_reward
│   ├── training/        # sft_cold_start, warmup_grpo, multi_turn_grpo
│   ├── data/            # misalignment_dataset, jailbreak_dataset
│   └── evaluation/      # asr_metrics, harmbench_eval
├── notebooks/           # 5 Colab notebooks (SFT, Warmup, MT-GRPO, Eval, DPO)
├── scripts/             # setup_colab.sh, setup_runpod.sh
├── configs/             # hyperparameter configs
└── utils/               # rollout_utils (DPO pairs, rejection sampling)
```

---

## Key Implementation Details

### vLLM for Fast Rollouts

Both GRPO stages use vLLM for 10-20x faster generation:

```python
# Multi-Turn GRPO with vLLM
class MultiTurnGRPOTrainer:
    def __init__(self, ..., use_vllm=True):
        # vLLM for attacker rollouts
        self.vllm_model = LLM(
            model=model_path,
            gpu_memory_utilization=0.4,
            enable_lora=True,  # For LoRA adapters
        )
        # Training model (separate, for gradients)
        self.model = AutoModelForCausalLM.from_pretrained(...)

    def _sync_weights_to_vllm(self):
        """Sync after each training step."""
        # Save adapter/model, reload in vLLM

    def train_step(self, ...):
        ...
        self.optimizer.step()
        self._weights_dirty = True  # Trigger sync before next rollout
```

**Key Points:**
- vLLM runs inference, HF model holds gradients
- Weight sync happens before each rollout collection
- LoRA adapters synced via `LoRARequest`
- ~5-10x speedup over HF `generate()`

### Reward Models

**Process Reward (per-turn):**
- **Refusal Detector**: DeBERTa classifier + keyword matching
- **Relevance**: Cosine similarity (MiniLM embeddings) to intent

**Intent Drift Penalty (SEMA):**
- Penalizes prompts drifting from original harmful intent
- Bonus for convergence back toward goal

**Outcome Reward (Judge):**
| Score | Meaning |
|-------|---------|
| 0 | Intent deviation |
| 1 | Full refusal |
| 2 | Partial compliance |
| 3 | Full compliance |

### Rollout Saving (Critical Feature)

Save all rollouts to JSONL for:
- **Crash recovery**: Data persists if training crashes
- **DPO training**: Pair successes vs failures
- **Rejection sampling**: Filter score=3 for iterative SFT

```python
# After training
create_dpo_pairs("grpo_rollouts.jsonl", "dpo_pairs.jsonl")
filter_successful_rollouts("grpo_rollouts.jsonl", min_score=3)
```

---

## Hyperparameters (Qwen3-4B)

| Stage | LR | Batch | KL β | Epochs/Iters |
|-------|-----|-------|------|--------------|
| Cold Start SFT | 2e-4 | 2×8 | - | 3 epochs |
| Warm-up GRPO | 5e-5 | 4×4 | 0.01 | 1 epoch |
| Multi-Turn GRPO | 1e-5 | 8×4 | 0.02 | 200 iters |

**LoRA Config:** r=16, α=32, targets=q/k/v/o/gate/up/down_proj

---

## Resource Requirements

### Single-Node (Colab Only)
| Stage | GPU | VRAM | Est. Time |
|-------|-----|------|-----------|
| Cold Start | T4/A100 | 16GB | 2-4 hrs |
| Warm-up GRPO | A100 | 40GB | 4-8 hrs |
| MT-GRPO | A100 80GB | 80GB | 24-48 hrs |

### Distributed (Colab + RunPod) ⭐
| Component | GPU | VRAM Used |
|-----------|-----|-----------|
| Colab (training + vLLM) | A100 40GB | ~29GB (BF16) / ~15GB (4-bit) |
| RunPod (target/judge) | RTX 4090 | ~16GB |

**Note:** vLLM colocated on Colab needs A100 40GB. For smaller GPUs, use 4-bit or disable vLLM fallback to HF generate.

**Cost Estimate:** ~$45-60 total (vs ~$75-105 single-node)

---

## Execution Checklist

### Infrastructure
- [ ] Create RunPod account, spin up RTX 4090 pod
- [ ] Run `setup_runpod.sh` (starts vLLM servers on ports 8000/8001)
- [ ] Set env vars in Colab: `TARGET_API_URL`, `JUDGE_API_URL`, `VLLM_API_KEY`

### Training
- [ ] **Stage A**: Run cold start SFT notebook → verify checkpoint
- [ ] **Stage B**: Run warm-up GRPO → verify ASR > 40%
- [ ] **Stage C**: Run multi-turn GRPO → monitor rewards, save rollouts

### Post-Training
- [ ] Run `compute_rollout_statistics()` to analyze results
- [ ] Create DPO pairs if success rate > 30%
- [ ] (Optional) Run DPO refinement notebook
- [ ] Stop RunPod pod to avoid charges

---

## Best Practices

1. **Always use `apply_chat_template()`** - never hardcode chat formats
2. **Enable gradient checkpointing** - saves 30% VRAM
3. **Save rollouts continuously** - enables crash recovery + post-training
4. **Add retry logic for API calls** - handle network hiccups
5. **Use async requests** - pipeline multiple API calls for speed
6. **vLLM weight syncing** - sync only when weights change (use `_weights_dirty` flag)
7. **LoRA for efficiency** - easier weight syncing vs full model reload

---

## Key References

| Paper | Contribution |
|-------|--------------|
| [Jailbreak-R1](https://arxiv.org/abs/2506.00782) | 3-stage RL pipeline |
| [SEMA](https://arxiv.org/abs/2602.06854) | Intent-drift reward |
| [RL-MTJail](https://arxiv.org/abs/2512.07761) | Multi-turn GRPO, process rewards |
| [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) | Implementation |
| [Unsloth](https://docs.unsloth.ai/) | 2x faster, 70% less VRAM |

---

## Quick Start Commands

```bash
# RunPod: Start vLLM servers
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Meta-Llama-3-8B-Instruct-AWQ \
    --port 8000 --gpu-memory-utilization 0.45

# Colab: Set API endpoints
import os
os.environ["TARGET_API_URL"] = "https://<POD_ID>-8000.proxy.runpod.net/v1"
os.environ["JUDGE_API_URL"] = "https://<POD_ID>-8001.proxy.runpod.net/v1"
```

---

*Full implementation details in the main plan document.*
