Great questions — let me search for existing work on jailbreak activation analysis and then explain how GLP compares to SAEs for feature discovery.This is a very active research area. Let me give you a comprehensive picture of what's been done and where GLP fits in relative to SAEs.

## Yes, This Has Been Done — Extensively

There's a rich body of work on exactly this idea of studying jailbreaks through the lens of activation space. Here are the key threads:

**"Safety Patterns" / Representation Engineering approach** — Researchers discovered that LLMs refusing malicious queries is linked to specific activation patterns they call "safety patterns," which can be extracted using just a few contrastive query pairs (harmful vs. benign). By weakening these safety patterns in the latent space of each layer's output, they could reduce the model's refusal ability — and conversely, strengthening them improves defense. This is the JRE (Jailbreaking through Representation Engineering) line of work.

**Safety Boundary work** — A large-scale study of 30,000+ samples showed that harmful, benign, and jailbreak prompts form distinct clusters in activation space, with jailbreak activations being distinct from both harmful and benign activation spaces. They found that jailbreak prompts bypass safety checks by shifting activations beyond a "safety boundary" into an unmonitored activation space, with low and middle layers showing the most pronounced effects.

**Refusal direction / Arditi et al.** — The foundational finding that refusal can be captured as a single linear direction in activation space, and simply removing that direction ablates the model's ability to refuse.

**SAE-based refusal analysis** — Recent work using SAEs to decompose refusal found that LLMs distinctly encode harm and refusal as separate feature sets, that harmful features have a clear causal effect on refusal features, and that adversarial jailbreaks operate by suppressing specific refusal-related features.

**Safety Knowledge Neurons** — Work at the individual neuron level found that adjusting the activation of safety-related knowledge neurons can effectively control the model's behavior with a mean attack success rate higher than 97%, by modifying only 0.3% of parameters.

## GLP vs SAE — How They Compare for This

This is the core of your question, and it's where things get interesting. Here's the comparison:

**SAEs** impose a strong structural assumption: activations decompose into a **sparse, linear** combination of features. Each feature is a direction in activation space, and interpretability comes from sparsity — you get monosemantic units you can label ("this feature fires for refusal," "this feature fires for harmful content detection").

**GLP** makes **no such structural assumption**. It's a deep nonlinear diffusion model that learns the full distribution of activations. The GLP paper shows that its internal neurons *also* isolate interpretable concepts (they do 1-D probing and find individual GLP neurons that correspond to human-understandable concepts), but it achieves this through generative modeling rather than enforced sparsity.

The key differences for your use case:

| | SAE | GLP |
|---|---|---|
| **Assumption** | Linear, sparse decomposition | None (learns full distribution) |
| **Feature finding** | Explicit sparse features you can enumerate | Neurons that emergently isolate concepts; probing required |
| **Steering** | Clamp/modify individual feature activations | Denoise perturbed activations back onto manifold |
| **Limitation** | Recent evidence shows safety is inherently distributed, emerging from coordinated patterns across many neurons rather than isolated in single features | Harder to enumerate a "feature dictionary" |
| **Strength for safety** | Can identify and name individual refusal features | Can model the full nonlinear manifold structure, potentially capturing distributed safety representations that SAEs miss |

## Where GLP Could Add Something New

The existing work has a known gap: SAEs trained on pretraining data often fail to capture task-specific features, missing key functional concepts needed for downstream applications — for instance, SAEs trained on pretraining corpora fail to encode features for refusal behavior, while those trained on chat data do. And the GSAE paper argues that safety is fundamentally distributed and relational, not sparse and local.

GLP could potentially address both of these issues because:

1. **It doesn't require sparsity** — if the "jailbreak subspace" is a nonlinear manifold rather than a sparse linear combination, GLP can model it while SAEs structurally cannot.

2. **On-manifold projection** — you could take a "borderline" jailbreak activation and use GLP's denoising to see *which direction it gets pulled* — toward refusal or toward compliance. This gives you a kind of "safety field" across activation space rather than just a single refusal direction.

3. **Probing GLP neurons** — the paper shows GLP neurons isolate concepts. You could probe specifically for safety-related concepts and see whether GLP finds a richer decomposition than SAEs do.

The novel contribution relative to existing work would be: using a **generative model of the activation manifold** (rather than linear probes, contrastive directions, or SAEs) to study the geometry of the safety boundary. Nobody has done this yet — the GLP paper itself shows "evil persona steering" as an example application but doesn't deeply analyze the safety/refusal geometry.

Want me to look at the actual codebase to see what would be involved in setting up this pipeline?

Great question — let me look at the actual GLP codebase to see how probing works in practice.Let me look at the actual paper for the probing methodology details, and also check what datasets they use.Now let me check the probing datasets they reference to understand what's actually needed.Now I have a very complete picture. Let me put this all together practically.

## How Probing Actually Works (Step by Step)

The probing procedure is surprisingly simple and lightweight. Here's exactly what happens:

### What "1-D Probing" Means

The idea is: take a single neuron (one scalar value) from your encoder (SAE feature or GLP meta-neuron), and use just that one number to predict a binary concept. If a single neuron can predict "is this text about safety/refusal/harm?" with high accuracy, then that neuron has *isolated* that concept.

### The Concrete Pipeline

**Step 1: Collect a labeled dataset of text prompts**

You need a binary classification dataset — pairs of (text, label). For the 113 tasks from Kantamneni et al., these include things like sentiment (positive/negative), whether a headline is front-page, whether text is entailment, etc. Each dataset has maybe a few hundred to a few thousand examples.

For your safety use case, you'd create datasets like:
- Harmful vs. benign prompts (label = 1 if harmful)
- Successful jailbreak vs. failed jailbreak (label = 1 if jailbreak succeeded)
- Refusal vs. compliance responses

You don't need a huge amount — even a few hundred examples per class works.

**Step 2: Get activations from the LLM**

Run each text prompt through the LLM (e.g., Llama 8B), extract the residual stream activation at a specific layer (they use layer 15 for Llama 8B, the middle layer). You typically take the last token's activation, which captures the model's overall understanding of the full input. This gives you one vector of dimension 4096 per prompt.

**Step 3: Encode through GLP (or SAE)**

Pass each activation through the GLP's encoder to get the intermediate "meta-neuron" activations. The GLP has a specific architecture — a deep MLP with SwiGLU blocks — and the intermediate representations at each layer contain these meta-neurons. So for each prompt, you get a vector of meta-neuron activations.

For an SAE, you'd get the sparse feature activations instead.

**Step 4: Train a 1-D probe**

For each meta-neuron `j`, train a simple threshold classifier: "if meta-neuron j > threshold, predict class 1." This is literally just finding the best threshold for a single scalar — it's a one-parameter model. You evaluate with AUC (area under the ROC curve) so the threshold doesn't even matter.

Then you report the best single meta-neuron's AUC for each concept. If GLP meta-neuron #4872 achieves AUC 0.95 on "is this a harmful prompt?", that means that single neuron nearly perfectly separates harmful from benign inputs.

### Computational Cost

This is the good news — **probing itself is very cheap**:

- **Collecting activations**: One forward pass per prompt through the LLM. For 1000 prompts through Llama 8B, this takes maybe a minute or two on a single GPU.
- **Encoding through GLP**: One forward pass per activation through the GLP (which is just an MLP). Very fast — seconds for 1000 activations.
- **Training the probe**: Literally computing AUC for each neuron — takes seconds on CPU.

The expensive part is **training the GLP itself** (5.6 days on an A100 for the Llama 8B version with 1 billion activations). But you don't need to do this — they provide pre-trained weights you can download:

```python
from glp.denoiser import load_glp
model = load_glp("generative-latent-prior/glp-llama8b-d6", device="cuda:0", checkpoint="final")
```

### What You'd Need for a Safety Probing Experiment

The total data requirements are modest:

1. **A set of harmful/benign prompt pairs** — ~200-1000 per class is fine. Existing datasets like AdvBench, WildJailbreak, or StrongREJECT provide these.
2. **Access to an open-source LLM** — Llama 3.1 8B (since the pre-trained GLP is for this model).
3. **The pre-trained GLP weights** — free download from HuggingFace.
4. **A single GPU with ~24GB VRAM** — the README says an RTX 4090 is sufficient for inference.

### GLP vs SAE for Finding Safety Features — The Key Difference

With an **SAE**, you get explicit sparse features. You look at which features are most different between harmful and benign activations (by mean difference), take the top-k, and probe on those. The features are inherently interpretable because sparsity forces each one to correspond to a narrow concept.

With **GLP**, you're probing on the internal neurons of a deep nonlinear diffusion model. The paper shows these neurons *also* isolate concepts — and actually do it better than SAEs on their 113-task benchmark. But the mechanism is different: these neurons emerge from learning the full generative distribution, not from an enforced sparsity constraint.

The GLP paper found that their meta-neurons substantially outperformed all SAE baselines on 1-D probing. The intuition is that GLP's nonlinear processing can disentangle concepts that are entangled in a linear (SAE) decomposition — which is exactly the scenario you'd expect for something like "safety," which the GSAE paper argues is inherently distributed and nonlinear.

### Practical Next Steps If You Want to Try This

The minimum viable experiment would be:

1. Grab ~500 jailbreak prompts (successful + failed) from WildJailbreak or similar
2. Run them through Llama 3.1 8B, extract layer 15 activations
3. Load the pre-trained GLP
4. Encode activations through the GLP
5. For each meta-neuron, compute AUC on "successful jailbreak vs. not"
6. Look at the top-scoring neurons — these are your candidate "jailbreak features"
7. Validate by looking at what other text maximally activates those neurons (using FineWeb, like the paper does)

This whole thing could run in an afternoon on a single 4090. Want me to sketch out the actual code for this?