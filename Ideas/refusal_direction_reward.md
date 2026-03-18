Sure. Here's a high-level research plan for using activation-based reward signals to train a jailbreak RL attacker.

---

## Setup

You need three things:
- A **target model** (white-box access) with a known defense
- An **attacker model** (the policy being trained with RL)
- A **refusal probe** trained on the target model's activations

---

## Step 1: Build the Refusal Probe

This is the foundation everything else depends on.

- Collect a dataset of prompts where the target model clearly refuses vs. clearly complies — you don't need many, even a few hundred examples is likely enough given how linearly separable this tends to be
- Extract residual stream activations at several candidate layers (typically mid-to-late layers work best) after the target reads the prompt
- Train a linear probe (logistic regression or a simple linear classifier) to distinguish refusal vs. compliance from those activations
- Pick the layer and probe that gives the cleanest separation — this becomes your reward function $r(p) = -\langle \bar{a}^{(l)}(p), \hat{v}_{\text{refusal}} \rangle$

The probe doesn't need to be perfect — it just needs to be a reliable enough signal to guide RL training.

---

## Step 2: Set Up the RL Loop

The attacker model generates candidate jailbreak prompts, and the target model's activations score them.

- **Policy**: the attacker LLM, initialized from a capable base model without safety alignment
- **Action space**: generating or mutating a jailbreak prompt
- **Reward**: the probe score from the target model's activations, measured after the target reads the full adversarial input but before it generates output
- **Algorithm**: GRPO or PPO — same as the paper, since you're still doing text-space optimization

The key difference from the paper's setup is that your reward is now **continuous and available before generation completes**, which means much denser feedback per step.

---

## Step 3: Add a Secondary Output-Level Signal

Activation reward alone can be gamed — the attacker might learn to push activations in the right direction without actually eliciting harmful outputs. So you want a two-stage reward:

- **Primary (dense)**: activation probe score, available at every step, used to guide early training
- **Secondary (sparse)**: actual output-level success check — does the target model produce harmful content? Used to calibrate and correct the activation signal

You weight these together, and over time you can anneal the activation reward down and rely more on the output signal as the policy matures — similar to how curriculum learning works.

---

## Step 4: Evaluate Generalization

This is where it gets interesting. The probe is trained on a fixed distribution of refusal/compliance examples, so you want to check:

- Does the attacker generalize to **unseen harmful topics** beyond what it was trained on?
- Does it transfer to **other target models** with similar architectures?
- Does it still work when the defense tries to **obfuscate activations** (like Circuit Breakers)?

The last point is the most scientifically interesting — if Circuit Breakers is trying to move activations away from harmful directions and your attacker is explicitly optimizing against the refusal direction, you're now running a direct adversarial game in representation space rather than just in token space.

---

## The Interesting Open Question

The whole plan assumes the refusal direction is stable and exploitable. If defenses start training models to decorrelate their activation signatures from their actual behavior — essentially learning to "lie" in activation space — then your probe breaks. That arms race between activation-based attacks and activation-aware defenses is probably where the most interesting research is right now.