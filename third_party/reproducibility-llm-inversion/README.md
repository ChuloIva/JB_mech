# SipIt: Language Model Inversion

This repository implements the **SipIt** (Sequential Inverse Prompt via Iterative Updates) algorithm from the paper:

**"Language Models are Injective and Hence Invertible"**
https://arxiv.org/abs/2510.15511

THE IMPLEMENTATION IS ENTIRELY DONE BY CLAUDE.AI. THIS RESPOSITORY IS UNRELATED TO THE ORIGINAL AUTHORS. 

## Overview

SipIt is a novel algorithm that reconstructs input sequences from hidden activations in transformer language models. Given hidden states from any layer of a transformer, SipIt can recover the original input text with provable theoretical guarantees.

### Key Features

- **Sequential Token Recovery**: Iteratively recovers tokens position-by-position
- **Layer-Agnostic**: Works on hidden states from any transformer layer
- **Multiple Policies**: Supports random and gradient-guided candidate selection
- **Epsilon-Ball Matching**: Configurable tolerance for hidden state matching
- **Theoretical Guarantees**: Achieves exact recovery with probability 1 in at most T|V| steps

## Installation

### Requirements

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.50.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/reproducibility-llm-inversion.git
cd reproducibility-llm-inversion

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from sipit import SipIt

# Initialize SipIt with GPT-2
sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-5)

# Invert a text prompt
text = "Hello, world!"
recovered_text, avg_distance = sipit.invert_from_text(text, policy="random")

print(f"Original:  '{text}'")
print(f"Recovered: '{recovered_text}'")
print(f"Match: {text == recovered_text}")
```

### Running Examples

The repository includes comprehensive examples demonstrating various use cases:

```bash
# Run all examples
python example.py

# Or run the basic demo
python sipit.py
```

## Algorithm Details

### Core Algorithm (Algorithm 1)

SipIt operates by:

1. **Initialization**: Start with an empty sequence
2. **Sequential Recovery**: For each position t:
   - Iterate through vocabulary candidates according to a policy
   - Compute hidden state for prefix + candidate token
   - Check if hidden state matches observed state within ε-ball
   - If match found, append token to recovered sequence
3. **Termination**: Return recovered sequence

### Pseudocode

```
Input: Observed layer-ℓ states; vocabulary V; tolerance ε ≥ 0
Output: Recovered sequence

Initialize: s^ ← ⟨⟩
For t = 1 to T:
  For j = 1 to |V|:
    vj ← Policy(V, tested_candidates, s^, ℓ)
    If h^t ∈ A_π,t(vj; ε):
      s^ ← s^ ⊕ vj
      break
```

### Policy Strategies

#### Random Policy
- Randomly shuffles vocabulary candidates
- Simple and effective baseline
- No gradient computation required

#### Gradient-Guided Policy
- Orders candidates by predicted likelihood
- Computes distances for all vocabulary tokens
- Prioritizes candidates with smallest distance to target
- More computationally expensive but potentially faster convergence

## API Reference

### SipIt Class

#### `__init__(model_name, layer_idx, epsilon, device)`

Initialize the SipIt algorithm.

**Parameters:**
- `model_name` (str): HuggingFace model name (default: "gpt2")
- `layer_idx` (int): Layer index for hidden states (default: -1, last layer)
- `epsilon` (float): Tolerance for hidden state matching (default: 1e-5)
- `device` (str): Device to run on ('cuda' or 'cpu', auto-detected if None)

#### `invert_from_text(text, policy, verbose)`

Invert text by first encoding it and extracting hidden states.

**Parameters:**
- `text` (str): Input text to invert
- `policy` (str): 'random' or 'gradient' (default: 'random')
- `verbose` (bool): Show progress bar (default: True)

**Returns:**
- `recovered_text` (str): Reconstructed text
- `avg_distance` (float): Average distance across all positions

#### `invert_sequence(target_hidden_states, policy, max_iterations_per_position, verbose)`

Invert a sequence from hidden states (core algorithm).

**Parameters:**
- `target_hidden_states` (Tensor): Target hidden states [1, seq_len, hidden_dim]
- `policy` (str): Candidate selection policy (default: 'random')
- `max_iterations_per_position` (int): Max candidates per position (default: vocab_size)
- `verbose` (bool): Show progress (default: True)

**Returns:**
- `recovered_tokens` (List[int]): List of recovered token IDs
- `distances` (List[float]): Distance at each position

## Examples

### Example 1: Basic Inversion

```python
from sipit import SipIt

sipit = SipIt(model_name="gpt2")
text = "The quick brown fox jumps over the lazy dog."
recovered, distance = sipit.invert_from_text(text)
```

### Example 2: Different Layers

```python
# Test different layers
for layer_idx in [0, 3, 6, 9, 11]:
    sipit = SipIt(model_name="gpt2", layer_idx=layer_idx)
    recovered, distance = sipit.invert_from_text("Hello world", verbose=True)
    print(f"Layer {layer_idx}: distance={distance:.6f}")
```

### Example 3: Policy Comparison

```python
sipit = SipIt(model_name="gpt2")

# Random policy
recovered_random, dist_random = sipit.invert_from_text(
    "Test prompt",
    policy="random"
)

# Gradient-guided policy
recovered_grad, dist_grad = sipit.invert_from_text(
    "Test prompt",
    policy="gradient"
)
```

### Example 4: Epsilon Tolerance

```python
# Test different epsilon values
for epsilon in [1e-6, 1e-5, 1e-4, 1e-3]:
    sipit = SipIt(model_name="gpt2", epsilon=epsilon)
    recovered, distance = sipit.invert_from_text("Test", verbose=True)
    print(f"ε={epsilon}: distance={distance:.6f}")
```

## Implementation Notes

### Computational Complexity

- **Time Complexity**: O(T × |V|) in worst case, where T is sequence length and |V| is vocabulary size
- **Space Complexity**: O(T × d) where d is hidden dimension
- **Practical Performance**: Average inversion time ~28 seconds for prompts of 20-200 tokens (paper result)

### Hidden State Matching

The algorithm uses L2 distance for hidden state matching:

```python
distance = ||target_hidden - candidate_hidden||_2
match = (distance <= epsilon)
```

### Model Support

Tested with:
- GPT-2 (all variants: small, medium, large, xl)
- GPT-Neo
- GPT-J

Other causal language models from HuggingFace should work with minimal modifications.

## Theoretical Foundations

### Theorem 3.1 (Correctness)

SipIt recovers the true input sequence with probability one in at most T|V| steps.

### Key Insights

1. **Injectivity**: Transformers are locally injective at each position
2. **Causal Structure**: Position t depends only on prefix and current token
3. **Sequential Recovery**: Can recover tokens one-by-one using causality

## Reproducibility

This implementation follows the specifications from the paper:

- Python 3.11
- PyTorch 2.0+
- Transformers 4.50.0
- Epsilon-ball matching criterion
- Random and gradient-guided policies

### Expected Results

- **Exact Recovery**: Should achieve exact token recovery on clean hidden states
- **Robustness**: Works across different layers with varying epsilon values
- **Efficiency**: Scales linearly with sequence length

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@misc{nikolaou2025languagemodelsinjectiveinvertible,
      title={Language Models are Injective and Hence Invertible}, 
      author={Giorgos Nikolaou and Tommaso Mencattini and Donato Crisostomi and Andrea Santilli and Yannis Panagakis and Emanuele Rodolà},
      year={2025},
      eprint={2510.15511},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.15511}, 
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This implementation is based on the paper "Language Models are Injective and Hence Invertible" (https://arxiv.org/abs/2510.15511).
