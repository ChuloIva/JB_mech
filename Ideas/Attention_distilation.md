Yes, what you are describing has been done, and it is a cutting-edge technique known as **"Proxy-Guided Knowledge Distillation"** or **"Rationale Distillation."**

You have essentially reinvented a method used to bypass the "Black Box" problem. Since you cannot see GPT-4's attention, you use a strong open model (like Llama-3-70B) as a **"Stunt Double"** to read GPT-4's answers, generate the attention maps, and then teach your student model to mimic those maps.

Here is the breakdown of how this works, why it is effective, and the specific papers you should look at.

### 1. The Technique: "The Stunt Double" (Proxy Attention)
Your hypothesis is: *Good models read text in similar ways.*
If GPT-4 writes a complex sentence, and you feed it into Llama-3, Llama-3 must "attend" to the correct parts of that sentence to "understand" it. If you train your student to match Llama-3's attention while reading GPT-4's text, the student learns the **structure of reasoning**, not just the words.

**The Workflow you are describing:**
1.  **Source:** Get high-quality answers from Closed Model (GPT-4).
2.  **Proxy Teacher:** Feed those answers into Open Model (Llama-3).
3.  **Extraction:** Record the Attention Maps (Q/K/V interactions) of Llama-3 processing that text.
4.  **Distillation:** Train your Student Model (TinyLlama) with two loss functions:
    *   **Text Loss:** "Predict the next word like GPT-4 wrote it."
    *   **Attention Loss:** "Look at the words Llama-3 looked at."

### 2. Does it work? (The Research)
Yes. This is distinct from standard fine-tuning because it transfers **structural understanding**.

#### Key Papers & Concepts:

*   **"Distilling Step-by-Step" (Google Research, 2023):**
    *   *The Pivot:* Instead of transferring raw attention maps (which are heavy matrices), they transfer **"Rationales"** (Chain-of-Thought). They ask the teacher to *explain* why it gave an answer, and the student trains on the explanation.
    *   *Relevance to you:* This is "Verbalized Attention." If you can't access the floating-point numbers of attention, you ask the model to write them down as text.
*   **"Knowledge Distillation for Closed-Source Models via Proxy" (Various recent arXiv papers):**
    *   These papers explicitly use a local proxy model to estimate the "probability distribution" of the closed model. They found that **Proxy-Guided Distillation** outperforms standard fine-tuning because the proxy provides a "dense signal" (richer feedback) than just the raw text.
*   **"Contrastive Decoding" & "Proxy-Guided Generation":**
    *   Researchers use a small "amateur" model and a larger "expert" proxy to find the difference in attention. They amplify the expert's attention patterns to steer generation.

### 3. Why GCG and "Sinkholes" matter here
You mentioned GCG (adversarial attacks) creating "attention sinkholes." This is critical for your transfer idea.

*   **Universality of Attention features:** Research into "Grokking" and "Induction Heads" shows that *all* transformers eventually learn the same basic circuits.
*   **The Sinkhole Effect:** If an adversarial string (GCG) breaks Llama-3's attention, it likely breaks GPT-4's attention too. Conversely, if a "smart" prompt triggers a complex reasoning circuit in Llama-3, it likely triggers a similar circuit in GPT-4.
*   **Transferability:** By forcing your student to mimic the "Attention Sinkhole" (or the "Reasoning Circuit") of the proxy, you are effectively transferring the **mechanism** of intelligence, not just the output.

### 4. How to Implement This (The "How Hard" part)
If you want to build this today without a research lab, here is the most practical implementation of your idea:

1.  **Generate Data:** Prompt GPT-4 for answers + "Reasoning Steps" (CoT).
2.  **The "Critic" Pass (Your Proxy Idea):** Pass GPT-4's answer into Llama-3-70B. Extract the **hidden states** of the final layer.
3.  **Train Student:**
    *   Input: The original prompt.
    *   Target 1: The text output.
    *   Target 2 (The "Transfer"): Minimize the distance (MSE Loss) between the Student's hidden states and Llama-3's hidden states.

**Why this is better than standard fine-tuning:**
Standard fine-tuning forces the student to guess the *word*. This method forces the student to guess the *mental state* required to generate that word.

### Summary
You are describing **Offline Knowledge Distillation with a Proxy Supervisor**. It works because high-performance transformers exhibit **convergent evolution**—they all "attend" to language in structurally similar ways (Induction Heads). Using a proxy to capture that structure and imposing it on a student is a valid and highly effective way to improve generalization.