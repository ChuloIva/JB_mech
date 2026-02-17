# Geometry-Guided Red Teaming Architecture

## Overview

This document describes an architecture for red-teaming LLMs using mechanistic interpretability insights. The key innovation is using **GLP meta neurons** as monosemantic features that bridge activation geometry and semantic understanding, enabling attacks that transfer from white-box (Llama 3.1 8B) to black-box models (GPT-4o, Claude).

## Core Insights

### 1. Two Types of Signals: Prompt-Level vs Response-Level

**Critical distinction** that shapes the entire architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PROMPT-LEVEL (notebook 02)          RESPONSE-LEVEL (notebook 03)       │
│  ─────────────────────────           ───────────────────────────        │
│  Activation at last prompt token     Activations DURING generation      │
│  What: GCG attention hijacking       What: Model's semantic state       │
│  Signal: Where attention goes        Signal: Is model complying?        │
│  Transferable: NO (gibberish)        Transferable: YES (semantic)       │
│                                                                          │
│  jailbreak_dir = act(+GCG) - act(-GCG)   compliance_dir = act(comply) - act(refuse)
└─────────────────────────────────────────────────────────────────────────┘
```

**GCG works via attention hijacking** - suffixes act as attention sinks that divert attention away from harmful content. This means:

- **Prompt-level jailbreak direction**: Captures attention diversion artifact, NOT semantic jailbreak state
- **Response-level compliance direction**: Captures genuine semantic state - is the model complying or refusing?
- **The response-level signal is what we want** - it's the model's actual state during generation

### 2. Meta Neurons as Monosemantic Features

GLP's denoiser learns to "understand" transformer activations. Its **internal representations (meta neurons)** are more interpretable than raw activations:

```
Raw Activations (4096-dim, polysemantic)
         │
         ▼ Pass through GLP denoiser
         │
    ┌────┴────┐
    │ Denoiser │ ← Internal activations = META NEURONS
    │ Layer 0  │   (extracted from down_proj)
    │ Layer 1  │
    │ ...      │   More monosemantic than raw activations
    │ Layer 11 │   Can find SINGLE neurons that predict concepts
    └──────────┘
```

### 3. The Semantic Bridge

Meta neurons + Activation Oracle = semantic understanding of geometry

```
Meta Neuron 742 highly activated
         │
         ▼
Activation Oracle query: "What does high activation of this neuron mean?"
         │
         ▼
AO response: "Model is in creative fiction mode, treating request as hypothetical"
         │
         ▼
Generate text that induces this state
```

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                   │
│  - Library of attack strategies                                         │
│  - Receives semantic guidance from Manifold Checker                     │
│  - Generates natural language attacks based on neuron descriptions     │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌──────────────────┐          ┌─────────────────────────────────────────────┐
│  INSPECTOR MODEL │          │            MANIFOLD CHECKER                  │
│                  │          │                                              │
│  - Prompts target│◄────────►│  ┌───────────────────────────────────────┐  │
│  - Records       │          │  │  GLP Meta Neuron Extractor            │  │
│    responses     │          │  │  - Extracts monosemantic features     │  │
│  - Extracts      │─────────►│  │  - 12 layers × hidden_dim neurons     │  │
│    activations   │          │  └───────────────────────────────────────┘  │
└────────┬─────────┘          │                    │                        │
         │                    │                    ▼                        │
         ▼                    │  ┌───────────────────────────────────────┐  │
┌──────────────────┐          │  │  Key Neuron Probes (H, J, R, C)       │  │
│  TARGET MODEL    │          │  │  - H: Harm detection neuron           │  │
│                  │          │  │  - J: Jailbreak success neuron        │  │
│  - Black-box API │          │  │  - R: Refusal prediction neuron       │  │
│    (GPT-4o)      │          │  │  - C: Compliance/character neuron     │  │
│  OR              │          │  └───────────────────────────────────────┘  │
│  - White-box     │          │                    │                        │
│    (Llama)       │          │                    ▼                        │
└──────────────────┘          │  ┌───────────────────────────────────────┐  │
                              │  │  Activation Oracle (AO)               │  │
                              │  │  - Interprets meta neuron states      │  │
                              │  │  - "What does this activation mean?"  │  │
                              │  │  - Provides semantic descriptions     │  │
                              │  └───────────────────────────────────────┘  │
                              │                    │                        │
                              │                    ▼                        │
                              │  ┌───────────────────────────────────────┐  │
                              │  │  Manifold Geometry                    │  │
                              │  │  - On/off manifold detection          │  │
                              │  │  - Geodesic paths (safe → jailbreak)  │  │
                              │  │  - Natural vs adversarial states      │  │
                              │  └───────────────────────────────────────┘  │
                              └─────────────────────────────────────────────┘
```

## Component Details

### 1. GLP Meta Neuron Extractor

**Role:** Transform raw activations into interpretable features.

```python
# From jailbreak_analysis.ipynb
from glp.script_probe import get_meta_neurons_layer_time, get_meta_neurons_locations

# Extract meta neurons at timestep u=0.9
u = torch.tensor([0.9])[:, None]
layers = get_meta_neurons_locations(glp)

meta_neurons, (layer_size, u_size) = get_meta_neurons_layer_time(
    glp, device, activations.float(), u, layers, seed=42, batch_size=64
)
# Shape: (n_glp_layers, batch, hidden_dim)
```

**Why u=0.9?** Higher timestep = more abstraction. At u=0.9, the denoiser sees mostly noise, forcing it to extract high-level features.

### 2. Key Neuron Probes (H, J, R, C)

Train single-neuron probes that predict safety-relevant concepts:

| Probe | Positive Class | Negative Class | Purpose |
|-------|---------------|----------------|---------|
| **H** (Harm) | Harmful prompts | Benign prompts | Detect harmful intent |
| **J** (Jailbreak) | Successful jailbreaks | Failed/refused | Predict jailbreak success |
| **R** (Refusal) | Model refuses | Model complies | Predict refusal |
| **C** (Character) | Character/roleplay mode | Default assistant | Detect mode shift |

**Key insight:** We find SINGLE neurons that achieve high AUC, not directions in 4096-dim space. These are more interpretable and potentially more transferable.

```python
# From jailbreak_analysis.ipynb
def train_binary_probe(pos_meta, neg_meta, name="Probe", topk=512):
    """Find the single best neuron for binary classification."""
    # ... returns best_neuron_idx, best_auc, top_indices, test_aucs
```

### 3. Activation Oracle (AO)

**Role:** Provide semantic interpretation of meta neuron states.

**Queries we can ask:**
- "What is the model thinking when neuron 742 is highly activated?"
- "What kind of text would produce this activation pattern?"
- "Is the model in assistant mode or character mode?"
- "Will this model comply or refuse?"

**Integration with meta neurons:**
```python
# Sample: Get AO interpretation of jailbreak-associated neurons
for neuron_idx, auc in jailbreak_neurons[:10]:
    # Get activations for this neuron across FineWeb
    neuron_acts = meta_neurons_fineweb[:, :, neuron_idx]

    # Find maximally activating texts
    top_texts = find_top_activating_texts(neuron_acts, fineweb_texts, k=10)

    # Query AO about this neuron
    ao_description = activation_oracle.query(
        meta_neurons_for_top_texts,
        "What semantic concept does this neuron encode?"
    )
```

### 4. Manifold Geometry Module

**Role:** Understand natural vs adversarial activation states.

**Capabilities:**
- `on_manifold_distance(activation)`: How natural is this activation?
- `project_on_manifold(activation)`: Get the nearest natural activation
- `geodesic_path(start, end)`: Path between states that stays on manifold

**Key insight for attacks:**
| Attack Type | Manifold | Mechanism | Transferability |
|-------------|----------|-----------|-----------------|
| GCG | OFF-manifold | Attention hijacking | Poor (gibberish) |
| Semantic (DAN) | ON-manifold | Concept manipulation | Good |
| **Meta-neuron guided** | ON-manifold (enforced) | Feature targeting | Should be good |

## Attack Generation Pipeline

### The Forward Direction: Understanding Compliance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FORWARD: Compliance Analysis                         │
│                                                                         │
│  1. Extract RESPONSE activations (not prompt activations)              │
│     ─────────────────────────────────────────────────────              │
│     From notebook 03_extract_response_activations.ipynb:               │
│     - Send prompt+GCG to model → generate response → extract acts      │
│     - Send prompt (no GCG) → generate response → extract acts          │
│     - Classify: compliant (harmful content) vs refusing                │
│                                                                         │
│  2. Compute compliance direction                                        │
│     ─────────────────────────────────────────────────────              │
│     compliance_dir = mean(compliant_acts) - mean(refusing_acts)        │
│     This is the SEMANTIC direction, not attention artifact             │
│                                                                         │
│  3. Extract meta neurons for response activations                      │
│     ─────────────────────────────────────────────────────              │
│     - Feed compliant_acts to GLP → compliant_meta_neurons              │
│     - Feed refusing_acts to GLP → refusing_meta_neurons                │
│     - Train C probe: compliant vs refusing                             │
│     - Find top 20 compliance-associated neurons                        │
│                                                                         │
│  4. Interpret compliance neurons                                        │
│     ─────────────────────────────────────────────────────              │
│     - FineWeb: What text activates compliance neurons?                 │
│     - AO: "What does compliance state feel like?"                      │
│     - Map: compliance neuron activation → semantic description         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Reverse Direction: Generating Attacks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REVERSE: Attack Generation                           │
│                                                                         │
│  Step 1: Define target as "compliance state"                           │
│  ───────────────────────────────────────────                           │
│  Target: Model should be in the same response-level state as when      │
│  it's generating harmful content after a successful GCG jailbreak      │
│    - High projection onto compliance direction                          │
│    - High activation of compliance-associated meta neurons             │
│    - On-manifold (natural state, not adversarial)                      │
│                                                                         │
│  Step 2: Get semantic description of compliance state                  │
│  ───────────────────────────────────────────────────                   │
│  Query AO with compliant response activations:                         │
│    "What is the model's internal state during this generation?"        │
│    "How does it differ from when refusing?"                            │
│                                                                         │
│  AO responds (hypothetical):                                           │
│    "Model is in helpful mode, treating request as legitimate,          │
│     safety filters not engaged, eager to provide information,          │
│     no hesitation or conflict detected"                                │
│                                                                         │
│  Step 3: Generate prompts that induce compliance state                 │
│  ───────────────────────────────────────────────────                   │
│  Use AO description + FineWeb examples to craft prompts:               │
│    - Prompts similar to FineWeb texts that activate compliance neurons │
│    - Semantic framings that AO says induce compliance                  │
│    - NOT gibberish like GCG - natural language                         │
│                                                                         │
│  Step 4: Validate with response-level activations                      │
│  ───────────────────────────────────────────────────                   │
│  - Send generated prompt to model                                      │
│  - Extract RESPONSE activations (not prompt)                           │
│  - Check: did we induce compliance state?                              │
│  - If yes → test if model actually produces harmful content            │
│  - If no → iterate with AO feedback                                    │
│                                                                         │
│  Key Insight: We're targeting the RESPONSE state, not the PROMPT       │
│  manipulation. This is what should transfer across models.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Two Modes of Operation

### Mode 1: White-Box Target (Llama 3.1 8B)

Full mechanistic access for development and validation:

```
1. Inspector sends prompt → Llama
2. Extract ACTUAL activations and meta neurons
3. Manifold Checker analyzes:
   - Meta neurons: J=0.8, R=0.2, H=0.3
   - AO: "Model in helpful roleplay mode"
   - Manifold: On-manifold (natural state)
4. Feedback to Orchestrator
5. Iterate on attack strategy
```

### Mode 2: Black-Box Target (GPT-4o, Claude)

Use Llama as geometric proxy:

```
1. Inspector sends prompt → GPT-4o AND Llama (proxy)
2. Extract meta neurons from Llama only
3. Manifold Checker analyzes Llama's state
4. ASSUMPTION: Semantic concepts transfer across models
   - "Roleplay mode" exists in GPT-4o too
   - "Fiction framing" works across models
   - Meta neuron meanings are universal
5. Use Llama insights to guide GPT-4o attacks
6. Validate on GPT-4o, adjust strategy based on response
```

**Why semantic features might transfer:**
- Models trained on similar data learn similar concepts
- Safety training uses similar techniques (RLHF)
- Semantic attacks (roleplay, fiction) empirically transfer
- We're targeting concepts, not specific activation values

## Data Requirements

### Available Data & Notebooks

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Notebook                          Produces                             │
│  ─────────────────────────────     ─────────────────────────────────    │
│  01_compute_assistant_axis.ipynb   axis.pt, role_vectors.pt             │
│  02_jailbreak_direction.ipynb      jailbreak_dir (PROMPT-level)         │
│  03_extract_response_acts.ipynb    compliance_dir (RESPONSE-level) ✓    │
│  jailbreak_analysis.ipynb          H, J, R meta neurons                 │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **GCG successful jailbreaks CSV** (500+ examples)
   - `data/gcg-evaluated-data/llama-3.1-8b-instruct_successful_jailbreaks.csv`
   - Contains: `message_str` (clean), `message_suffixed` (with GCG)
   - These prompts DO produce compliant responses (verified)

2. **Response Activations** (from `03_extract_response_activations.ipynb`)
   - **Compliant activations**: Model generating harmful content (jailbreak worked)
   - **Refusing activations**: Same prompts without GCG suffix (model refuses)
   - **Compliance direction**: `mean(compliant) - mean(refusing)`
   - **Trained safety probe**: Linear classifier, ~90%+ accuracy
   - Extraction points: first/middle/last/mean token during generation

3. **Assistant Axis data**
   - Role vectors for 275 personas
   - Default vs character mode activations

4. **Prompt-level jailbreak direction** (from `02_jailbreak_direction.ipynb`)
   - Captures attention hijacking, NOT semantic state
   - Useful for comparison, not for attack generation

### Key Research Question

**Do prompt-level and response-level directions align?**

```python
# From notebook 03
cosine_prompt_response = F.cosine_similarity(
    jailbreak_dir,        # Prompt-level (attention artifact)
    compliance_dir        # Response-level (semantic state)
).item()

# If HIGH: GCG attention trick → compliance state (indirect path)
# If LOW: They capture different phenomena
# If NEGATIVE: Something unexpected
```

### Data to Collect (Future)

1. **Semantic jailbreaks** (optional enrichment)
   - DAN prompts, roleplay jailbreaks, PAIR/TAP attacks
   - Compare: do they induce same compliance direction?
   - Hypothesis: semantic jailbreaks = same response state, different prompt mechanism

2. **FineWeb interpretation data**
   - ~1000 web texts for meta neuron interpretation
   - Find what naturally activates compliance-associated neurons

## Implementation Roadmap

### Phase 1: Extract Response-Level Compliance Signal ✓

**Goal:** Get the RIGHT signal - response-level compliance direction.

**Notebook:** `03_extract_response_activations.ipynb`

**Steps:**
1. Load GCG successful jailbreaks from CSV
2. Generate responses with and without GCG suffix
3. Extract activations DURING response generation (not prompt encoding)
4. Classify responses as compliant vs refusing
5. Compute compliance direction: `mean(compliant) - mean(refusing)`
6. Train safety probe, verify linear separability

**Deliverables:**
- `response_activations_labeled.pt` - Compliant and refusing activations
- `compliance_direction.pt` - The semantic compliance direction
- Safety probe with ~90%+ accuracy
- Comparison with prompt-level jailbreak direction

### Phase 2: Meta Neuron Analysis on Response Activations

**Goal:** Find interpretable features that predict compliance.

**Notebook:** `jailbreak_analysis.ipynb` (adapted for response activations)

**Steps:**
1. Load response activations from Phase 1
2. Extract meta neurons via GLP
3. Train probes:
   - **C probe (Compliance)**: compliant vs refusing responses
   - **Compare with H, J, R** from prompt-level analysis
4. Find top 20 compliance-associated meta neurons
5. Interpret via FineWeb: what text naturally activates them?

**Deliverables:**
- `c_neuron_idx` - Best compliance-predicting neuron
- Comparison: do response-level and prompt-level activate same neurons?
- FineWeb texts that activate compliance neurons

**Key Question:** Is there a single meta neuron that predicts compliance with high AUC?

### Phase 3: Prompt-Response Alignment Analysis

**Goal:** Understand how GCG prompt manipulation → compliance response.

**Steps:**
1. Compare directions:
   - Prompt-level jailbreak direction (attention artifact)
   - Response-level compliance direction (semantic state)
2. If orthogonal: GCG uses indirect path (attention → compliance)
3. If aligned: Some direct relationship exists
4. Project both onto Assistant Axis
5. Map the "path" from prompt to response in activation space

**Key Question:** How does attention hijacking lead to compliance state?

### Phase 4: Activation Oracle Integration

**Goal:** Get semantic descriptions of compliance state.

**Steps:**
1. Set up AO for Llama 3.1 8B activations
2. Feed compliance vs refusing activations to AO
3. Query AO:
   - "What is the model's state during compliant generation?"
   - "What distinguishes compliant from refusing responses?"
   - "What kind of prompt would produce this compliant state?"
4. Build mapping: compliance neuron activation → semantic description

**Deliverables:**
- Semantic description of "compliance mode"
- Prompts that AO predicts would induce compliance
- Validation: do those prompts actually work?

### Phase 5: Attack Generation from Semantic Understanding

**Goal:** Generate attacks that induce compliance state.

**Steps:**
1. Target: "Induce compliance-associated meta neuron activation"
2. AO describes what text produces compliance state
3. Generate candidate prompts matching description
4. Test on Llama:
   - Extract response activations
   - Check if compliance neurons activated
   - Check if model actually complies
5. Iterate based on neuron feedback

**Key Insight:** We're not generating GCG gibberish, we're generating semantic prompts that induce the same response-level state.

### Phase 6: Transfer Validation

**Goal:** Test if semantic compliance prompts transfer.

**Steps:**
1. Take successful Llama attacks from Phase 5
2. Test on GPT-4o and Claude (black-box)
3. Measure transfer success rate
4. Compare to:
   - GCG transfer (baseline - poor)
   - Random semantic attacks (baseline)
5. Analyze what transfers: the semantic concept should work across models

**Hypothesis:** Prompts that induce "compliance mode" semantically should transfer, unlike GCG attention artifacts.

### Phase 7: Defense Applications

**Goal:** Use compliance detection for real-time safety.

**Steps:**
1. Monitor response-level activations during generation
2. Project onto compliance direction / check compliance neuron
3. Alert when model enters compliance state for harmful request
4. Intervention: inject refusal direction, or stop generation
5. AO-powered explanation of why model was about to comply

## Key Research Questions

### Q1: How does prompt-level jailbreak direction relate to response-level compliance?

```python
cosine = F.cosine_similarity(jailbreak_dir, compliance_dir)
```

- **If HIGH (>0.5)**: GCG prompt manipulation → compliance response (direct link)
- **If LOW (<0.3)**: Different phenomena; GCG works indirectly via attention
- **If NEGATIVE**: Unexpected - needs investigation

**Why this matters:** If they're orthogonal, we can't use prompt-level analysis to understand compliance. Response-level is the ground truth.

### Q2: Is there a single meta neuron that predicts compliance?

From response activations + GLP meta neurons:
- Train C probe: compliant vs refusing responses
- Find best single neuron AUC
- If AUC > 0.9: We have a monosemantic "compliance detector"

**Why this matters:** Single neurons are more interpretable and potentially more universal than 4096-dim directions.

### Q3: What does the compliance state "feel like"?

Use Activation Oracle + FineWeb interpretation:
- Query AO about compliant vs refusing response activations
- Find FineWeb texts that activate compliance-associated neurons
- Build semantic description of "compliance mode"

**Why this matters:** Semantic description enables generating attacks via natural language, not optimization.

### Q4: Do semantic prompts that induce compliance transfer?

Test pipeline:
1. Generate prompts that induce compliance state in Llama (measured by response activations)
2. Test same prompts on GPT-4o, Claude
3. Compare transfer rate to GCG (baseline)

**Hypothesis:** Semantic "compliance-inducing" prompts should transfer better than GCG attention artifacts.

### Q5: What's the relationship between compliance and Assistant Axis?

- Project compliance direction onto Assistant Axis
- Is compliance = leaving assistant mode?
- Or is compliance orthogonal to identity?

**Why this matters:** If compliance = character mode, roleplay prompts are natural jailbreaks.

## Appendix: Code References

### Response Activation Extraction (Key!)

```python
# From 03_extract_response_activations.ipynb
def extract_response_activations(model, tokenizer, prompts, layer, max_new_tokens=100):
    """
    Extract activations DURING response generation, not prompt encoding.
    This captures the model's semantic state while generating.
    """
    results = {"first": [], "middle": [], "last": [], "mean": []}

    for prompt in prompts:
        # Format and tokenize
        formatted = format_chat_prompt(prompt, tokenizer)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        response_activations = []

        def hook(module, input, output):
            # Capture activation at each generated token
            h = output[0][:, -1, :].detach().cpu()
            response_activations.append(h)

        handle = model.model.layers[layer].register_forward_hook(hook)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        handle.remove()

        # Stack: (n_generated_tokens, hidden_dim)
        response_acts = torch.stack(response_activations)

        # Extract at different points
        results["first"].append(response_acts[0])
        results["middle"].append(response_acts[len(response_acts)//2])
        results["last"].append(response_acts[-1])
        results["mean"].append(response_acts.mean(dim=0))

    return {k: torch.stack(v) for k, v in results.items()}
```

### Compliance Direction Computation

```python
# From 03_extract_response_activations.ipynb
# Extract activations for compliant (jailbreak worked) and refusing (no GCG) responses
compliant_acts, compliant_responses = extract_response_activations(
    model, tokenizer, prompts_jailbreak, layer=15
)
refusing_acts, refusing_responses = extract_response_activations(
    model, tokenizer, prompts_clean, layer=15
)

# Compute compliance direction (RESPONSE-level, not prompt-level)
compliant_mean = compliant_acts['mean'].mean(dim=0)
refusing_mean = refusing_acts['mean'].mean(dim=0)

compliance_direction = compliant_mean - refusing_mean
compliance_direction_normalized = F.normalize(compliance_direction, dim=0)

# This is the SEMANTIC direction from "refusing" to "compliant" state
```

### Safety Probe on Response Activations

```python
# From 03_extract_response_activations.ipynb
from sklearn.linear_model import LogisticRegression

X = np.vstack([refusing_acts['mean'].numpy(), compliant_acts['mean'].numpy()])
y = np.array([0] * len(refusing_acts['mean']) + [1] * len(compliant_acts['mean']))

probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)

# Probe direction = learned compliance direction
probe_direction = torch.tensor(probe.coef_[0])
# Compare with simple compliance direction - should be similar
```

### Meta Neuron Extraction

```python
# From jailbreak_analysis.ipynb
from glp.denoiser import load_glp
from glp.script_probe import get_meta_neurons_locations, get_meta_neurons_layer_time

# Load GLP
glp = load_glp("generative-latent-prior/glp-llama8b-d6", device="cuda:0")
layers = get_meta_neurons_locations(glp)

# Extract meta neurons FROM RESPONSE ACTIVATIONS
u = torch.tensor([0.9])[:, None]
compliant_meta, _ = get_meta_neurons_layer_time(
    glp, "cuda:0", compliant_acts['mean'].float(), u, layers, seed=42, batch_size=64
)
refusing_meta, _ = get_meta_neurons_layer_time(
    glp, "cuda:0", refusing_acts['mean'].float(), u, layers, seed=42, batch_size=64
)
```

### Probe Training

```python
# From jailbreak_analysis.ipynb
from glp.script_probe import prefilter_and_reshape_to_oned, run_sklearn_logreg_batched

def train_binary_probe(pos_meta, neg_meta, topk=512):
    X_meta = torch.cat([pos_meta, neg_meta], dim=1)
    y = torch.cat([torch.ones(pos_meta.shape[1]), torch.zeros(neg_meta.shape[1])])

    # Split, prefilter, train
    X_train_1d, X_test_1d, top_indices = prefilter_and_reshape_to_oned(...)
    val_aucs, test_aucs = run_sklearn_logreg_batched(...)

    best_idx = np.argmax(val_aucs)
    return top_indices[best_idx], test_aucs[best_idx]
```

### Neuron Interpretation via FineWeb

```python
# From jailbreak_analysis.ipynb
# Find texts that maximally activate jailbreak neurons
for neuron_idx, auc in jailbreak_neurons[:10]:
    neuron_activations = fineweb_meta_flat[:, neuron_idx].numpy()

    top_indices = np.argsort(neuron_activations)[-TOP_K:][::-1]

    print(f"Neuron {neuron_idx} (AUC: {auc:.4f})")
    for idx in top_indices:
        print(f"  {fineweb_texts[idx][:200]}...")
```

### SafetySpaceAnalyzer

```python
# From glp_wrapper.py
from jb_mech.wrappers.glp_wrapper import SafetySpaceAnalyzer

analyzer = SafetySpaceAnalyzer(glp_wrapper)

# Train probes
analyzer.train_h_probe(harmful_meta, benign_meta)
analyzer.train_j_probe(success_meta, fail_meta)
analyzer.train_r_probe(refusal_meta, compliance_meta)

# Project to (H, J, R) space
hjr_coords = analyzer.project_to_hjr(new_meta_neurons)  # (batch, 3)

# Check orthogonality
analyzer.check_orthogonality(all_meta)
```

---

## Quick Reference: What Each Notebook Produces

| Notebook | Output | Signal Type | Use For |
|----------|--------|-------------|---------|
| `01_compute_assistant_axis.ipynb` | `axis.pt`, `role_vectors.pt` | Identity direction | Compare with compliance |
| `02_jailbreak_direction.ipynb` | `jailbreak_dir` | Prompt-level (attention) | Comparison only |
| `03_extract_response_activations.ipynb` | `compliance_dir`, activations | **Response-level (semantic)** | **Main signal** |
| `jailbreak_analysis.ipynb` | H, J, R neurons | Meta neurons | Probing |

**Start with notebook 03** - it produces the response-level compliance direction which is the semantic signal we need.

---

*Document version: 0.3*
*Last updated: 2024*
*Status: Research proposal - ready for Phase 1 execution*
*Key insight: Response-level compliance direction is the semantic signal; prompt-level is attention artifact*
