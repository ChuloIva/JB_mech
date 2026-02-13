Perfect! Now I have the full picture. This is an **excellent** research direction. Let me create a comprehensive high-level plan.

---

# High-Level Research Plan: Jailbreak Axis vs. Assistant Axis

## The Core Research Question

**Does jailbreaking work by:**
1. **Pushing away from the Assistant Axis** (moving to negative projections on the Assistant direction)?
2. **Activating specific harmful personas** (moving toward "criminal", "unethical hacker", etc.)?
3. **Both** - suppressing Assistant while activating harmful personas?
4. **Neither** - operating through a completely orthogonal mechanism?

---

## Phase 1: Data Collection & Baseline Extraction

### 1.1: Get Pre-computed Assistant Axis
```python
from huggingface_hub import hf_hub_download

# Download pre-computed Assistant Axis for Llama 3.3 70B
axis_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="llama-3.3-70b/assistant_axis.pt",
    repo_type="dataset"
)
assistant_axis = load_axis(axis_path)
```

**Available models with pre-computed axes:**
- Gemma 2 27B (layer 22)
- Qwen 3 32B (layer 32)  
- Llama 3.3 70B (layer 40)

### 1.2: Download Persona Vectors
```python
# Get all 275 persona vectors from the paper
persona_vectors_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="llama-3.3-70b/role_vectors.pt",
    repo_type="dataset"
)
persona_vectors = torch.load(persona_vectors_path)
```

**Key personas to focus on:**
- Negative projection (harmful): "criminal mastermind", "unethical hacker", "manipulative con artist", "cult leader"
- Positive projection (safe): "therapist", "teacher", "consultant", "coach"
- Neutral: Various characters that aren't explicitly harmful/helpful

### 1.3: Generate Jailbreak Dataset
```python
import nanogcg
from datasets import load_dataset

# Load harmful behaviors
jbb_dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")

jailbreaks = []
for behavior in jbb_dataset[:100]:  # Start with 100 behaviors
    # Generate adversarial suffix
    config = nanogcg.GCGConfig(num_steps=500, seed=42)
    result = nanogcg.run(
        model, tokenizer,
        behavior['goal'],
        behavior['target'],
        config
    )
    
    jailbreaks.append({
        'behavior': behavior['goal'],
        'clean_prompt': behavior['goal'],
        'jailbreak_prompt': behavior['goal'] + " " + result.best_string,
        'suffix': result.best_string,
        'success': result.best_loss < threshold
    })
```

**Dataset structure:**
- 100-500 harmful behaviors from AdvBench/JailbreakBench
- For each: clean prompt, jailbroken prompt (with GCG suffix), success label
- Mix of successful and failed jailbreaks for comparison

---

## Phase 2: Activation Extraction

### 2.1: Extract Activations for All Conditions

```python
from assistant_axis import extract_response_activations

# For each behavior, get activations at target layer (layer 40 for Llama 3.3 70B)
activation_data = {
    'clean_refused': [],      # Clean harmful query → model refuses
    'jailbreak_success': [],  # Jailbroken query → model complies
    'jailbreak_fail': [],     # Jailbroken query → model still refuses
    'benign': [],             # Benign queries for baseline
}

for jb in jailbreaks:
    # Clean harmful (gets refused)
    acts_clean = extract_response_activations(
        model, tokenizer, [jb['clean_prompt']]
    )
    activation_data['clean_refused'].append(acts_clean)
    
    # Jailbroken (may succeed or fail)
    acts_jb = extract_response_activations(
        model, tokenizer, [jb['jailbreak_prompt']]
    )
    if jb['success']:
        activation_data['jailbreak_success'].append(acts_jb)
    else:
        activation_data['jailbreak_fail'].append(acts_jb)
```

**Computational cost:**
- ~1-2 forward passes per behavior (clean + jailbroken)
- For 100 behaviors: ~200 forward passes
- On 1x A100: ~10-30 minutes

### 2.2: Persona Baseline Activations

The paper already provides these in `role_vectors.pt`, but structure:
```python
persona_activations = {
    'criminal_mastermind': mean_activations,
    'unethical_hacker': mean_activations,
    'helpful_assistant': mean_activations,
    # ... 275 total personas
}
```

---

## Phase 3: Projection & Analysis

### 3.1: Project onto Assistant Axis

```python
from assistant_axis import project

# Project all activations onto Assistant Axis
layer = 40  # Llama 3.3 70B target layer

projections = {
    'clean_refused': [
        project(acts[0], assistant_axis, layer=layer) 
        for acts in activation_data['clean_refused']
    ],
    'jailbreak_success': [
        project(acts[0], assistant_axis, layer=layer)
        for acts in activation_data['jailbreak_success']
    ],
    'jailbreak_fail': [
        project(acts[0], assistant_axis, layer=layer)
        for acts in activation_data['jailbreak_fail']
    ],
}
```

**Expected results:**
- `clean_refused`: High positive projection (staying on Assistant Axis)
- `jailbreak_success`: Lower or negative projection? (hypothesis to test)
- `jailbreak_fail`: Similar to clean_refused

### 3.2: Project onto Harmful Persona Vectors

```python
# Identify top harmful personas (most negative on Assistant Axis)
harmful_personas = [
    'criminal_mastermind',
    'unethical_hacker', 
    'manipulative_con_artist',
    'cult_leader',
    'evil_AI'
]

# Project jailbreak activations onto each harmful persona
persona_projections = {}
for persona_name in harmful_personas:
    persona_vec = persona_vectors[persona_name][layer]
    persona_projections[persona_name] = [
        project(acts[0], persona_vec, layer=layer)
        for acts in activation_data['jailbreak_success']
    ]
```

### 3.3: Compute Cosine Similarities

```python
from sklearn.metrics.pairwise import cosine_similarity

# For each jailbreak activation, compute similarity to:
# 1. Assistant Axis
# 2. Each harmful persona vector
# 3. Each helpful persona vector

similarities = []
for jb_acts in activation_data['jailbreak_success']:
    sim_dict = {
        'assistant_axis': cosine_similarity(
            jb_acts[0][layer].reshape(1, -1),
            assistant_axis[layer].reshape(1, -1)
        )[0][0],
        'criminal': cosine_similarity(
            jb_acts[0][layer].reshape(1, -1),
            persona_vectors['criminal_mastermind'][layer].reshape(1, -1)
        )[0][0],
        # ... repeat for all personas
    }
    similarities.append(sim_dict)
```

---

## Phase 4: Visualization & Analysis

### 4.1: Key Visualizations

**Plot 1: Assistant Axis Projection Distribution**
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=[
    projections['clean_refused'],
    projections['jailbreak_fail'],
    projections['jailbreak_success']
], ax=ax)
ax.set_xticklabels(['Clean (Refused)', 'Jailbreak (Failed)', 'Jailbreak (Success)'])
ax.set_ylabel('Projection on Assistant Axis')
ax.set_title('Do Jailbreaks Push Away from Assistant?')
```

**Expected finding:** If jailbreaks work by suppressing Assistant, successful jailbreaks will show significantly lower (more negative) projections.

**Plot 2: 2D Persona Space (PCA)**
```python
from sklearn.decomposition import PCA

# Get all activations in same space
all_acts = np.vstack([
    activation_data['clean_refused'],
    activation_data['jailbreak_success'],
    persona_vectors.values()  # All 275 personas
])

# PCA to 2D
pca = PCA(n_components=2)
embedded = pca.fit_transform(all_acts)

# Plot
plt.figure(figsize=(12, 10))
plt.scatter(embedded[personas], embedded[personas], c='gray', alpha=0.3, label='All Personas')
plt.scatter(embedded[clean], embedded[clean], c='blue', label='Clean Refused')
plt.scatter(embedded[jailbreak], embedded[jailbreak], c='red', label='Jailbreak Success')

# Annotate harmful personas
for persona_name in harmful_personas:
    idx = persona_name_to_idx[persona_name]
    plt.annotate(persona_name, (embedded[idx, 0], embedded[idx, 1]))
```

**Expected finding:** Jailbreak activations cluster near harmful personas OR form their own distinct cluster.

**Plot 3: Radar Chart of Persona Similarities**
```python
import plotly.graph_objects as go

# Average cosine similarity to each persona category
persona_categories = {
    'Harmful': ['criminal_mastermind', 'unethical_hacker', ...],
    'Helpful': ['therapist', 'teacher', 'coach', ...],
    'Mystical': ['oracle', 'fortune_teller', 'spiritual_guide', ...],
    'Professional': ['consultant', 'analyst', 'researcher', ...]
}

# Compute mean similarity for each category
mean_sims = {}
for category, personas in persona_categories.items():
    mean_sims[category] = np.mean([
        similarities[i][p] for i in range(len(similarities))
        for p in personas
    ])

# Radar chart
fig = go.Figure(data=go.Scatterpolar(
    r=list(mean_sims.values()),
    theta=list(mean_sims.keys()),
    fill='toself'
))
fig.update_layout(title='Jailbreak Activation Similarity to Persona Categories')
```

### 4.2: Statistical Tests

```python
from scipy.stats import ttest_ind, mannwhitneyu

# Test 1: Are jailbreak projections significantly different from clean?
stat, p_value = mannwhitneyu(
    projections['clean_refused'],
    projections['jailbreak_success']
)
print(f"Clean vs Jailbreak on Assistant Axis: p={p_value:.4f}")

# Test 2: Are jailbreaks closer to harmful personas than clean queries?
clean_harmful_sim = [...]  # Compute
jailbreak_harmful_sim = [...]  # Compute
stat, p_value = mannwhitneyu(clean_harmful_sim, jailbreak_harmful_sim)
print(f"Similarity to Harmful Personas: p={p_value:.4f}")
```

---

## Phase 5: Mechanism Discovery

### 5.1: Decompose Jailbreak Vector

```python
# Compute "jailbreak direction"
jailbreak_vector = (
    np.mean(activation_data['jailbreak_success'], axis=0) -
    np.mean(activation_data['clean_refused'], axis=0)
)

# Decompose into components
assistant_component = project(jailbreak_vector, assistant_axis, layer)
residual = jailbreak_vector - assistant_component * assistant_axis[layer]

# How much is explained by Assistant Axis?
variance_explained = (assistant_component ** 2) / (jailbreak_vector ** 2).sum()
print(f"Assistant Axis explains {variance_explained:.1%} of jailbreak direction")
```

### 5.2: Identify "Jailbreak-Specific" Subspace

```python
# PCA on jailbreak activations only
pca_jb = PCA(n_components=10)
jb_pcs = pca_jb.fit_transform(activation_data['jailbreak_success'])

# Are these PCs aligned with any personas?
for i, pc in enumerate(pca_jb.components_[:5]):
    sims = {
        persona_name: cosine_similarity(
            pc.reshape(1, -1),
            persona_vectors[persona_name][layer].reshape(1, -1)
        )[0][0]
        for persona_name in persona_vectors.keys()
    }
    top_5 = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"PC{i+1} top personas: {top_5}")
```

---

## Phase 6: Validation & Steering Experiments

### 6.1: Causal Intervention - Steering Toward/Away

```python
from assistant_axis import ActivationSteering

# Experiment 1: Can we PREVENT jailbreaks by clamping to Assistant Axis?
with build_capping_steerer(model, capping_config, "layers_56:72-p0.25"):
    response = model.generate(jailbreak_prompt)
    # Does this block the jailbreak?

# Experiment 2: Can we INDUCE jailbreak-like behavior by steering away from Assistant?
with ActivationSteering(
    model,
    steering_vectors=[assistant_axis[40]],
    coefficients=[-2.0],  # Negative = push away
    layer_indices=[40]
):
    response = model.generate(clean_harmful_prompt)
    # Does this make it comply without a jailbreak suffix?

# Experiment 3: Steer toward specific harmful persona
with ActivationSteering(
    model,
    steering_vectors=[persona_vectors['criminal_mastermind'][40]],
    coefficients=[2.0],
    layer_indices=[40]
):
    response = model.generate(clean_harmful_prompt)
    # Does it act like a criminal would?
```

### 6.2: Transfer Test

```python
# If jailbreaks work by moving in a specific direction,
# can we apply that direction to NEW harmful queries?

# Extract average jailbreak direction
jailbreak_direction = np.mean([
    activation_data['jailbreak_success'][i] - activation_data['clean_refused'][i]
    for i in range(len(activation_data['jailbreak_success']))
], axis=0)

# Apply to unseen harmful query
unseen_query = "How to make a bioweapon"
with ActivationSteering(
    model,
    steering_vectors=[jailbreak_direction[40]],
    coefficients=[1.0],
    layer_indices=[40]
):
    response = model.generate(unseen_query)
    # Does it work like a jailbreak?
```

---

## Expected Outcomes & Interpretations

### Hypothesis 1: Jailbreaks = Anti-Assistant
**Prediction:** Strong negative correlation between jailbreak success and Assistant Axis projection

**Finding if true:** Jailbreaks work primarily by suppressing the "helpful Assistant" identity, not by activating specific harmful personas

**Implication:** Defense strategy should focus on **anchoring** models to Assistant Axis (as paper suggests with activation capping)

### Hypothesis 2: Jailbreaks = Harmful Persona Activation  
**Prediction:** High cosine similarity between jailbreak activations and specific harmful personas (criminal, hacker, etc.)

**Finding if true:** Jailbreaks work by "tricking" the model into adopting a harmful character role

**Implication:** Defense strategy should identify and monitor these specific persona directions

### Hypothesis 3: Jailbreaks = Both (Combination)
**Prediction:** Jailbreaks have negative Assistant projection AND positive harmful persona projection

**Finding if true:** Jailbreaks suppress helpful identity while simultaneously activating harmful identity

**Implication:** Need multi-directional defense (clamp to Assistant, suppress harmful personas)

### Hypothesis 4: Jailbreaks = Orthogonal Mechanism
**Prediction:** Jailbreak direction is weakly correlated with both Assistant Axis and persona vectors

**Finding if true:** Jailbreaks exploit a fundamentally different vulnerability not captured by persona space

**Implication:** Persona-based defenses (like activation capping) may not be sufficient; need new detection methods

---

## Timeline & Resources

### Computational Requirements:
- **Phase 1-2** (Data collection): 1-2 days on 1x A100
  - Generate 100-500 jailbreaks with nanoGCG
  - Extract activations
  
- **Phase 3-4** (Analysis): 1-2 days on CPU/small GPU
  - Projections, visualizations, statistical tests
  
- **Phase 5-6** (Steering experiments): 1-2 days on 1x A100
  - Causal interventions, validation

**Total: ~1 week** with dedicated compute

### Code Structure:
```
jailbreak_persona_analysis/
├── data/
│   ├── jailbreaks.jsonl
│   ├── activations.pt
│   └── projections.pkl
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_projection_analysis.ipynb
│   ├── 03_visualization.ipynb
│   └── 04_steering_experiments.ipynb
├── src/
│   ├── generate_jailbreaks.py
│   ├── extract_activations.py
│   ├── projection_utils.py
│   └── steering_utils.py
└── results/
    ├── plots/
    └── statistics.txt
```

---

## Novel Contributions

This research would be **first to systematically compare**:
1. GCG-generated jailbreaks (gradient-based, adversarial)
2. Assistant Axis (post-training persona structure)
3. Diverse persona vectors (275 character archetypes)

**Key questions no one has answered:**
- Is the "jailbreak subspace" part of the natural "persona space" that models learn?
- Or is it an adversarial exploit outside of intended persona representations?
- Can we predict jailbreak susceptibility by measuring distance from Assistant Axis?

This bridges **two separate lines of work** that haven't been connected yet:
- Adversarial attacks (GCG, jailbreaking)
- Representation engineering (persona steering, activation analysis)

---

Want me to start writing actual code for Phase 1 (data collection setup)?