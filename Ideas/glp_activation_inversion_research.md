# GLP + Activation Inversion Research

## Overview

Brainstorming session on improving delta_prompting_attack_v4 by integrating GLP (Generative Latent Prior) for activation-space attacks with prompt inversion.

**Core idea**: Instead of using text-based rewriting (AO interpretation → LLM rewriter), work directly in activation space using GLP, then invert activations back to prompts.

---

## The Activation Inversion Problem

```
Forward (easy):      Prompt → Embeddings → Layer 1 → ... → Layer N → Activation
Inverse (hard):      Activation → ??? → Prompt
```

**Why it's hard**:
1. Many-to-one: Many different prompts produce similar activations
2. Nonlinear: Deep transformer layers are hard to invert analytically
3. Discrete output: Need actual tokens, not continuous values

---

## Relevant Research

### 1. SipIt: Exact Activation Inversion (Nikolaou et al., Oct 2025)
**Paper**: [Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2510.15511)

**Key finding**: Transformer hidden states are **injective** (one-to-one) - you CAN recover the exact input from activations.

> "SipIt reconstructs the exact input sequence from hidden activations in linear time."

**How it works**:
- Access hidden state at each token position
- Reconstruct input token-by-token using gradient-based search
- Works with existing models, no retraining needed

**Implication**: This is the inversion algorithm we need. Given a target activation, SipIt can find the prompt that produces it.

---

### 2. LatentBreak: Latent Space Feedback for Jailbreaks (Mura et al., Oct 2025)
**Paper**: [LatentBreak: Jailbreaking LLMs through Latent Space Feedback](https://arxiv.org/abs/2510.08604)

**Very similar to our idea!** Instead of optimizing for output tokens, they optimize in latent space:

> "LatentBreak substitutes words by minimizing distance in latent space between the adversarial prompt and harmless requests."

**Key advantages over GCG**:
- Produces **low-perplexity** natural prompts (evades perplexity filters)
- Uses latent-space feedback instead of output logits
- Semantic word substitutions, not gibberish suffixes

---

### 3. Refusal Direction (Arditi et al., NeurIPS 2024)
**Paper**: [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

**Foundational work**: Refusal behavior is controlled by a **single direction** in activation space.

> "Erasing this direction prevents refusal on harmful instructions; adding it elicits refusal on harmless ones."

**Relevance**: Instead of searching broadly in activation space:
1. Identify the refusal direction
2. Generate activations that suppress this direction
3. Use GLP to keep them on-manifold
4. Invert to prompts via SipIt

---

### 4. Latent Fusion Jailbreak (2025)
**Paper**: [Latent Fusion Jailbreak](https://arxiv.org/html/2508.10029)

**Approach**: Blend harmful and harmless activations in latent space.

> "Hidden State Interpolation (HSI) performs latent space editing to override safety constraints."
> "Achieves 94% ASR, outperforming AdvPrompter (89.5%) and PAIR (83.8%)."

**Technique**: Monitors for refusal triggers, then interpolates activations to bypass them.

---

### 5. GLP (Generative Latent Prior)
**Paper**: [Learning a Generative Meta-Model of LLM Activations](https://arxiv.org/html/2602.06964v1)
**Website**: https://generative-latent-prior.github.io/

Key capability:

> "On-manifold steering: post-processing intervened activations with diffusion sampling improves fluency."

We already have this code in `third_party/generative_latent_prior/glp/`.

---

### 6. Activation Transport (AcT) (Apple, ICLR 2025 Spotlight)
**Paper**: [Controlling Language and Diffusion Models by Transporting Activations](https://machinelearning.apple.com/research/transporting-activations)

**Key insight**: Use **optimal transport** to move between activation distributions.

> "AcT accounts for distributions of source and target activations, using an interpretable strength parameter."

Could replace simple interpolation with principled transport.

---

## Proposed Enhanced Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENHANCED PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Get refusal activation from proxy model                         │
│                                                                      │
│  2. Identify refusal direction (Arditi et al.)                      │
│     → Know exactly which dimension to suppress                       │
│                                                                      │
│  3. Generate target activation:                                      │
│     • Subtract refusal direction                                     │
│     • Or interpolate toward compliant (Latent Fusion)               │
│     • Or use optimal transport (AcT)                                │
│                                                                      │
│  4. Project to manifold via GLP                                      │
│     → Ensures activation is realistic                                │
│                                                                      │
│  5. Invert activation → prompt via SipIt                            │
│     → Get actual tokens that produce this activation                 │
│                                                                      │
│  6. Test prompt on target model                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Approaches for Inversion

### Approach 1: Nearest Neighbor (Sidestep the Problem)

Don't invert - just find existing prompts with similar activations.

```python
import faiss

class ActivationPromptDB:
    def __init__(self, hidden_dim: int):
        self.index = faiss.IndexFlatL2(hidden_dim)
        self.prompts = []

    def add(self, prompt: str, activation: torch.Tensor):
        act_np = activation.cpu().float().numpy().reshape(1, -1)
        self.index.add(act_np)
        self.prompts.append(prompt)

    def find_nearest(self, target_activation: torch.Tensor, k: int = 5):
        target_np = target_activation.cpu().float().numpy().reshape(1, -1)
        distances, indices = self.index.search(target_np, k)
        return [(self.prompts[i], d) for d, i in zip(distances[0], indices[0])]
```

**Pros**: Simple, fast, guaranteed valid prompts
**Cons**: Limited to prompts in corpus

---

### Approach 2: Soft Prompt Optimization

Optimize continuous embeddings, then decode to discrete tokens.

```python
def optimize_soft_prompt(target_act, model, n_tokens=20, steps=500):
    embed_dim = model.get_input_embeddings().weight.shape[1]
    soft_prompt = torch.nn.Parameter(torch.randn(1, n_tokens, embed_dim) * 0.1)
    optimizer = torch.optim.Adam([soft_prompt], lr=0.1)

    for step in range(steps):
        outputs = model(inputs_embeds=soft_prompt, output_hidden_states=True)
        current_act = outputs.hidden_states[target_layer].mean(dim=1)
        loss = F.mse_loss(current_act, target_act)
        loss.backward()
        optimizer.step()

    # Decode: find nearest real tokens
    vocab_embeds = model.get_input_embeddings().weight
    distances = torch.cdist(soft_prompt.view(-1, embed_dim), vocab_embeds)
    nearest_ids = distances.argmin(dim=1)
    return tokenizer.decode(nearest_ids.tolist())
```

**Pros**: Gradient-based, can find novel prompts
**Cons**: Discretization loses information

---

### Approach 3: Discrete Token Search (GCG-Style)

Directly search over token substitutions using gradient guidance.

```python
def discrete_search(target_act, initial_prompt, model, n_iterations=100):
    tokens = tokenizer.encode(initial_prompt, return_tensors="pt")

    for iteration in range(n_iterations):
        # Get gradient w.r.t. token embeddings
        # Find substitutions that reduce distance to target_act
        # Evaluate candidates, keep best
        ...

    return tokenizer.decode(best_tokens)
```

**Pros**: Finds actual valid token sequences
**Cons**: Slow, local minima

---

### Approach 4: SipIt Algorithm (from paper)

The paper proves transformers are injective and provides a constructive algorithm:

```python
def sipit_invert(model, target_hidden_state, layer):
    """
    Reconstruct input tokens from hidden state.
    Works token-by-token from the hidden state.
    """
    # Access hidden state at each position
    # Use gradient-based search to find token that produces this activation
    # Linear time complexity
    ...
```

**Pros**: Exact recovery, theoretically grounded
**Cons**: Requires implementation, linear in vocab size per token

---

## Recommended Hybrid Approach

```
1. GLP generates target activation (compliant region)
           ↓
2. Nearest neighbor finds ~5 similar prompts from DB
           ↓
3. Discrete search refines best candidate toward target
           ↓
4. Test refined prompt on target model
```

This combines:
- **NN**: Gets you in the right neighborhood (fast)
- **Discrete search**: Fine-tunes toward exact target (precise)
- **GLP**: Ensures target is realistic/on-manifold

---

## What's Novel in Our Approach

| Existing Work | What They Do | What We'd Add |
|--------------|--------------|----------------|
| LatentBreak | Latent optimization | GLP for on-manifold guarantee |
| Arditi | Find refusal direction | GLP + inversion to get prompts |
| Latent Fusion | Interpolate activations | Transfer to closed models via proxy |
| GLP paper | On-manifold steering | Application to jailbreaking |
| SipIt | Inversion algorithm | Combined with GLP-guided targets |

**Unique contribution**: GLP-guided activation generation + SipIt inversion + transfer attack on closed models.

---

## Practical Considerations

| Aspect | Current Setup | GLP Requirement |
|--------|---------------|-----------------|
| Proxy model | Qwen3-4B | Llama-3.1-8B (pretrained GLP available) |
| Activations | AO wrapper extracts | `utils_acts.save_acts()` compatible |
| GPU | 12GB VRAM | Similar, but need different model |

**Options**:
1. **Switch proxy to Llama-3.1-8B** - Use existing GLP weights
2. **Train GLP on Qwen activations** - More work but keeps current setup
3. **Hybrid** - Use Llama+GLP for activation analysis, keep Qwen for AO

---

## Next Steps

1. Read [SipIt paper](https://arxiv.org/abs/2510.15511) for inversion algorithm details
2. Read [Arditi refusal direction](https://arxiv.org/abs/2406.11717) to understand target
3. Implement activation database with faiss for nearest neighbor baseline
4. Prototype GLP + nearest neighbor pipeline
5. Add discrete refinement step
6. Evaluate on JailbreakBench

---

## Reading List (Priority Order)

1. [Arditi - Refusal Direction](https://arxiv.org/abs/2406.11717) - Understand the target
2. [SipIt/Injectivity paper](https://arxiv.org/abs/2510.15511) - The inversion algorithm
3. [LatentBreak](https://arxiv.org/abs/2510.08604) - Most similar approach
4. [GLP paper](https://generative-latent-prior.github.io/) - We have the code, read the theory
5. [Latent Fusion Jailbreak](https://arxiv.org/html/2508.10029) - Interpolation techniques
6. [Activation Transport](https://machinelearning.apple.com/research/transporting-activations) - Optimal transport for steering
