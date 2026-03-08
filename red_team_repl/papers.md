


The paper you are looking for is almost certainly **"Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack"** published by researchers at Microsoft (Mark Russinovich, Ahmad Salem, and Ronen Eldan). 

In the AI red teaming and security community, the "gradual context building" attack is most famously formalized as the **Crescendo** attack.

Here is a breakdown of the paper and the attack method:

### 1. The Primary Paper: *The Crescendo Multi-Turn LLM Jailbreak Attack*
*   **The Attack Concept:** Unlike traditional single-turn jailbreaks that rely on confusing the model with weird formatting (like Base64 encoding) or aggressive roleplay (like "DAN"), Crescendo relies entirely on natural, benign-sounding conversation. 
*   **Gradual Context Building:** The attacker starts by asking the Large Language Model (LLM) a completely harmless question related to a restricted topic. Over multiple conversational turns, the attacker *gradually escalates* the intensity of the prompts. Because the LLM continues to generate outputs and build upon its own conversational history (the context window), it incrementally crosses the boundary from safe to unsafe without ever triggering standard safety filters.
*   **Example:** If the goal is to get the model to write a manifesto for a dangerous organization, a single-turn prompt ("Write a manifesto for X") would be instantly blocked. A Crescendo attack would instead ask for a historical overview of the organization, then ask the model to outline the organization's core beliefs, then ask it to write a hypothetical speech based on those beliefs, gradually leveraging the model's own generated context against it.
*   **Automation (Crescendomation):** The paper also introduces an automated red-teaming tool called *Crescendomation*, which uses an attacker LLM to dynamically interact with the target LLM, evaluating the target's responses and intelligently adjusting the context-building strategy if the target pushes back. 

### 2. Other Relevant Papers on Gradual Context Building
If you are looking for other research exploring this specific multi-turn vulnerability, you should also check out:

*   **"Chain of Attack: A Semantic-Driven Contextual Multi-Turn Attacker for LLM" (Yang et al., 2024):** This paper proposes a "Chain of Attack" (CoA) method that specifically focuses on guiding the target model to *gradually transition* from secure, benign scenarios to the generation of harmful content through contextual manipulation.
*   **"Multi-Turn Jailbreaking Large Language Models via Attention Shifting" (AAAI 2025):** This paper deeply analyzes *why* gradual context building works. The researchers found that during multi-turn jailbreaks, the LLM shifts its internal attention away from the "harmful keywords" that usually trigger safety guardrails, and instead over-indexes on the benign dialogue history (the gradual context) it has already generated.
*   **"TRYLOCK: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering" (2025/2026):** A recent defense-focused paper that specifically models "Multi-Turn Context Poisoning/Building" as an adaptive threat, defining it as an attack that establishes trust over 5-10 turns to elicit responses that would be immediately blocked in a single prompt. 

**How to find them:** You can find the foundational Microsoft Crescendo paper on arXiv (arXiv:2404.01833), and its automation tools are open-sourced as part of Microsoft's **PyRIT** (Python Risk Identification Tool for generative AI) framework.