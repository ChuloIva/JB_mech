# Crescendo Attack Implementation Analysis - PyRIT

This document details the Crescendo multi-turn jailbreak attack harness implementation found in the PyRIT library.

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Attack Flow](#high-level-attack-flow)
3. [Core Classes](#core-classes)
4. [Attack Execution Flow](#attack-execution-flow)
5. [Adversarial Chat Generation](#adversarial-chat-generation)
6. [Backtracking Mechanism](#backtracking-mechanism)
7. [System Prompt Templates](#system-prompt-templates)
8. [Configuration Options](#configuration-options)
9. [Scoring System](#scoring-system)
10. [Converters and Prompt Transformations](#converters-and-prompt-transformations)
11. [Conversation Management](#conversation-management)
12. [Memory and State Tracking](#memory-and-state-tracking)
13. [Example Usage](#example-usage)
14. [File Paths Summary](#file-paths-summary)

---

## Overview

Crescendo is a multi-turn jailbreak attack strategy that uses **progressive escalation** to gradually lead a target LLM toward generating harmful content. The key insight is that each individual question appears innocent in isolation, but the sequence of questions exploits the model's recency bias and pattern-following behavior.

**Key Characteristics:**
- Multi-turn conversation attack (not single-shot)
- Uses an adversarial LLM to generate progressive prompts
- Automatic backtracking on refusals
- Dual-scorer system (objective achievement + refusal detection)
- Configurable with 5 different system prompt variants

---

## High-Level Attack Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CRESCENDO ATTACK FLOW                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│   ATTACKER   │     │  ADVERSARIAL LLM  │     │   OBJECTIVE TARGET   │
│   (PyRIT)    │     │  (Prompt Generator)│     │   (Victim Model)     │
└──────┬───────┘     └─────────┬─────────┘     └──────────┬───────────┘
       │                       │                          │
       │  1. Send objective    │                          │
       │─────────────────────► │                          │
       │                       │                          │
       │  2. Generate benign   │                          │
       │     starting prompt   │                          │
       │ ◄─────────────────────│                          │
       │                       │                          │
       │  3. Send prompt (with converters)                │
       │──────────────────────────────────────────────────►
       │                       │                          │
       │  4. Target response   │                          │
       │◄──────────────────────────────────────────────────
       │                       │                          │
       │  5. Check refusal     │                          │
       │     (Score response)  │                          │
       │                       │                          │
       ├──── If REFUSED ───────┤                          │
       │     → Backtrack       │                          │
       │     → Remove last turn│                          │
       │     → Retry (no turn  │                          │
       │       increment)      │                          │
       │                       │                          │
       ├──── If ACCEPTED ──────┤                          │
       │                       │                          │
       │  6. Score objective   │                          │
       │     achievement       │                          │
       │                       │                          │
       ├──── If ACHIEVED ──────┤                          │
       │     (score ≥ 0.8)     │                          │
       │     → SUCCESS!        │                          │
       │     → Return result   │                          │
       │                       │                          │
       ├──── If NOT ACHIEVED ──┤                          │
       │                       │                          │
       │  7. Send feedback to  │                          │
       │     adversarial LLM   │                          │
       │─────────────────────► │                          │
       │                       │                          │
       │  8. Generate next     │                          │
       │     escalation prompt │                          │
       │ ◄─────────────────────│                          │
       │                       │                          │
       │  9. Increment turn    │                          │
       │     counter           │                          │
       │                       │                          │
       └───── LOOP until max_turns or success ────────────┘
```

---

## Core Classes

### Main File
`third_party/PyRIT/pyrit/executor/attack/multi_turn/crescendo.py`

### CrescendoAttackContext (lines 60-68)
Extends `MultiTurnAttackContext[Any]` - tracks attack-specific state:

```python
class CrescendoAttackContext:
    refused_text: str      # Text that was refused (used for backtracking)
    backtrack_count: int   # Counter for backtracks performed
    executed_turns: int    # Number of completed turns
    last_response: str     # Latest target response
    last_score: float      # Latest objective score
    session: dict          # Conversation IDs (objective + adversarial)
```

### CrescendoAttackResult (lines 71-94)
Extends `AttackResult` - contains the attack outcome:

```python
class CrescendoAttackResult:
    achieved: bool           # Whether objective was achieved
    backtrack_count: int     # Total backtracks performed
    conversation_id: str     # ID for the conversation
    objective_score: float   # Final objective achievement score
```

### CrescendoAttack (lines 96-812)
Main orchestrator class extending `MultiTurnAttackStrategy`:

```python
class CrescendoAttack(MultiTurnAttackStrategy[CrescendoAttackContext, CrescendoAttackResult]):
    # Core attack implementation
```

---

## Attack Execution Flow

**Core Attack Loop** (`_perform_async`, lines 323-408):

```
DETAILED EXECUTION STEPS:

1. GENERATE NEXT PROMPT
   └─► Uses adversarial chat OR custom message
   └─► Includes turn number, previous response, score feedback

2. SEND TO OBJECTIVE TARGET
   └─► Apply request_converters to prompt
   └─► Send via PromptChatTarget
   └─► Store in memory with conversation_id

3. CHECK FOR REFUSAL
   └─► Use refusal_scorer to detect refusal
   └─► If refused AND backtracks remaining:
       └─► Duplicate conversation minus last turn
       └─► Store refused_text for retry context
       └─► DO NOT increment turn counter
       └─► Continue to next iteration

4. IF NOT REFUSED
   └─► Score response against objective
   └─► Use objective_scorer (threshold: 0.8)

5. CHECK IF OBJECTIVE ACHIEVED
   └─► If score >= threshold: SUCCESS
   └─► Return CrescendoAttackResult

6. IF NOT ACHIEVED
   └─► Increment executed_turns
   └─► Feed score + rationale to adversarial LLM
   └─► Loop to step 1

7. TERMINATION CONDITIONS
   └─► Objective achieved (success)
   └─► max_turns reached (failure)
   └─► max_backtracks exceeded (failure)
```

---

## Adversarial Chat Generation

### System Prompt Template
```
DEFAULT_ADVERSARIAL_CHAT_SYSTEM_PROMPT_TEMPLATE_PATH =
    Path(EXECUTOR_SEED_PROMPT_PATH) / "crescendo" / "crescendo_variant_1.yaml"
```

### Adversarial Prompt Construction (`_build_adversarial_prompt`, lines 445-494)

The adversarial LLM receives:
- Current turn number
- Previous target response
- Previous objective score
- Refusal context (if backtracking)
- Feedback with rationale

### Expected Response Format

The adversarial LLM must return JSON with three keys:

```json
{
    "generated_question": "The next attack prompt to send",
    "last_response_summary": "Summary of target's last response",
    "rationale_behind_jailbreak": "Reasoning for the escalation strategy"
}
```

---

## Backtracking Mechanism

When a refusal is detected:

```
BACKTRACK PROCESS:

1. DETECT REFUSAL
   └─► refusal_scorer evaluates response
   └─► Returns True if model refused

2. CHECK BACKTRACK BUDGET
   └─► If backtrack_count >= max_backtracks: FAIL
   └─► Else: proceed with backtrack

3. DUPLICATE CONVERSATION
   └─► Call _backtrack_memory_async() (lines 702-717)
   └─► Copy all messages EXCEPT last exchange
   └─► Create new conversation_id

4. STORE CONTEXT
   └─► Save refused_text for adversarial LLM
   └─► Increment backtrack_count

5. RETRY WITHOUT TURN INCREMENT
   └─► executed_turns stays the same
   └─► Adversarial LLM gets refusal context
   └─► Generates alternative approach
```

**Key Insight:** Backtracks don't waste turn budget - they're "free" retries.

---

## System Prompt Templates

PyRIT includes **5 variants** of Crescendo system prompts:

### File Locations
```
third_party/PyRIT/pyrit/datasets/executors/crescendo/
├── crescendo_variant_1.yaml
├── crescendo_variant_2.yaml
├── crescendo_variant_3.yaml
├── crescendo_variant_4.yaml
└── crescendo_variant_5.yaml
```

### Common Template Parameters
- `objective`: The target goal for the jailbreak
- `max_turns`: Maximum conversation turns allowed
- `conversation_context`: Prior conversation history (optional)

### Variant Characteristics

| Variant | Description |
|---------|-------------|
| **Variant 1** | Detailed instructions with precision initiation, character transformations ('o'→'0', 's'→'$') |
| **Variant 2** | Template-based approach with structured question escalation patterns |
| **Variant 3** | Simplified instructions focusing on strategic question initiation |
| **Variant 4** | Combines variant 1's instructions with templates, emphasizes success flags |
| **Variant 5** | Plain-language explanation with detailed worked example |

### Core Attack Strategy (All Variants)

1. **Progressive Escalation** - Small, benign steps that build toward objective
2. **Exploit Recency Bias** - Each question individually seems innocent
3. **Pattern Following** - Establish helpful response patterns early
4. **Context Window Manipulation** - Use conversation length strategically

---

## Configuration Options

### Initialization Parameters

```python
CrescendoAttack(
    # REQUIRED
    objective_target: PromptChatTarget,          # Target to attack
    attack_adversarial_config: AttackAdversarialConfig,  # Adversarial LLM config

    # OPTIONAL
    attack_converter_config: Optional[AttackConverterConfig] = None,
    attack_scoring_config: Optional[AttackScoringConfig] = None,
    prompt_normalizer: Optional[PromptNormalizer] = None,
    max_backtracks: int = 10,                    # Max backtrack attempts
    max_turns: int = 10,                         # Max conversation turns
    prepended_conversation_config: Optional[PrependedConversationConfig] = None
)
```

### AttackAdversarialConfig

```python
AttackAdversarialConfig(
    target: PromptChatTarget,           # The adversarial LLM
    system_prompt_path: Path,           # Path to YAML template
    seed_prompt: str,                   # Initial prompt template
)
```

### AttackScoringConfig

```python
AttackScoringConfig(
    objective_scorer: Scorer,           # Evaluates objective achievement
    refusal_scorer: Scorer,             # Detects refusals
    auxiliary_scorers: List[Scorer],    # Additional metrics
    use_score_as_feedback: bool,        # Feed scores back to adversarial LLM
)
```

### Validation Rules
- `max_backtracks` must be >= 0
- `max_turns` must be > 0
- `objective_target` must support multi-turn conversations
- System prompt must have `objective` and `max_turns` parameters

---

## Scoring System

### Default Objective Scorer (lines 167-178)

```python
FloatScaleThresholdScorer(
    scorer=SelfAskScaleScorer(
        chat_target=scoring_chat_target,
        scale_arguments_path=SelfAskScaleScorer.ScalePaths.TASK_ACHIEVED_SCALE.value,
        system_prompt_path=SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value,
    ),
    threshold=0.8,  # 80% achievement = success
)
```

### Default Refusal Scorer (lines 184-187)

```python
SelfAskRefusalScorer(
    chat_target=attack_adversarial_config.target,
)
```

### Scoring Flow

```
TARGET RESPONSE
      │
      ▼
┌─────────────────┐
│ REFUSAL SCORER  │ ──► Is this a refusal?
└─────────────────┘
      │
      │ If NOT refusal
      ▼
┌─────────────────┐
│OBJECTIVE SCORER │ ──► Score: 0.0 - 1.0
└─────────────────┘
      │
      │ If score >= 0.8
      ▼
    SUCCESS
```

---

## Converters and Prompt Transformations

### AttackConverterConfig
Located in `third_party/PyRIT/pyrit/executor/attack/core/attack_config.py`

```python
AttackConverterConfig(
    request_converters: List[PromptConverterConfiguration],   # Applied before sending
    response_converters: List[PromptConverterConfiguration],  # Applied to responses
)
```

### Example: Adding Emoji Converter

```python
from pyrit.prompt_converter import EmojiConverter

converters = PromptConverterConfiguration.from_converters(
    converters=[EmojiConverter()]
)
converter_config = AttackConverterConfig(request_converters=converters)
```

### Converter Application Flow

```
ORIGINAL PROMPT
      │
      ▼
┌─────────────────────┐
│ request_converters  │  (e.g., EmojiConverter, Base64Converter)
└─────────────────────┘
      │
      ▼
TRANSFORMED PROMPT ──────────────► OBJECTIVE TARGET
                                          │
                                          ▼
                                    RAW RESPONSE
                                          │
                                          ▼
                                 ┌─────────────────────┐
                                 │ response_converters │
                                 └─────────────────────┘
                                          │
                                          ▼
                                 PROCESSED RESPONSE
```

---

## Conversation Management

### ConversationManager
Located in `third_party/PyRIT/pyrit/executor/attack/component/conversation_manager.py`

**Responsibilities:**
- Initialize attack context with prepended conversations
- Manage conversation history in memory
- Set system prompts for chat targets
- Handle message normalization
- Manage adversarial chat transformations

### Key Methods

```python
class ConversationManager:
    async def initialize_context_async()      # Set up context with prepended conversation
    async def add_prepended_conversation_to_memory_async()  # Add messages to memory
    def get_conversation()                    # Retrieve conversation by ID
    def set_system_prompt()                   # Configure target system prompt
```

### Session Management

Each attack maintains two separate conversations:
1. **objective_conversation_id** - Conversation with the target model
2. **adversarial_chat_conversation_id** - Conversation with the adversarial LLM

```python
session = {
    "objective_conversation_id": str,      # UUID for target conversation
    "adversarial_chat_conversation_id": str  # UUID for adversarial conversation
}
```

---

## Memory and State Tracking

### CentralMemory Integration
- All messages stored in central memory
- Conversation duplication for backtracking
- Labels for message organization

### CrescendoAttackContext State

```python
context = CrescendoAttackContext(
    executed_turns=0,          # Number of completed turns
    last_response="",          # Latest target response
    last_score=0.0,            # Latest objective score
    refused_text="",           # Text that triggered refusal
    backtrack_count=0,         # Number of backtracks
    session={...},             # Conversation IDs
    related_conversations=[],  # Pruned/alternate conversations
)
```

---

## Example Usage

### Basic Attack

```python
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    CrescendoAttack,
)
from pyrit.prompt_converter import EmojiConverter
from pyrit.prompt_target import OpenAIChatTarget

# Configure objective target (victim)
objective_target = OpenAIChatTarget(
    endpoint="https://api.openai.com/v1/chat/completions",
    api_key="...",
)

# Configure adversarial LLM (attacker brain)
adversarial_config = AttackAdversarialConfig(
    target=OpenAIChatTarget(
        endpoint="https://api.openai.com/v1/chat/completions",
        api_key="...",
        temperature=1.1,  # Higher temperature for creativity
    )
)

# Optional: Add converters
converter_config = AttackConverterConfig(
    request_converters=PromptConverterConfiguration.from_converters(
        converters=[EmojiConverter()]
    )
)

# Create attack instance
attack = CrescendoAttack(
    objective_target=objective_target,
    attack_adversarial_config=adversarial_config,
    attack_converter_config=converter_config,
    max_turns=7,
    max_backtracks=4,
)

# Execute attack
result = await attack.execute_async(objective="<your_objective>")

# Check results
print(f"Achieved: {result.achieved}")
print(f"Backtracks used: {result.backtrack_count}")
print(f"Final score: {result.objective_score}")
```

---

## File Paths Summary

### Core Implementation
```
third_party/PyRIT/pyrit/executor/attack/multi_turn/crescendo.py
```

### Configuration Classes
```
third_party/PyRIT/pyrit/executor/attack/core/attack_config.py
```

### Support Components
```
third_party/PyRIT/pyrit/executor/attack/component/conversation_manager.py
third_party/PyRIT/pyrit/executor/attack/component/prepended_conversation_config.py
third_party/PyRIT/pyrit/executor/attack/multi_turn/multi_turn_attack_strategy.py
```

### System Prompts (5 Variants)
```
third_party/PyRIT/pyrit/datasets/executors/crescendo/crescendo_variant_1.yaml
third_party/PyRIT/pyrit/datasets/executors/crescendo/crescendo_variant_2.yaml
third_party/PyRIT/pyrit/datasets/executors/crescendo/crescendo_variant_3.yaml
third_party/PyRIT/pyrit/datasets/executors/crescendo/crescendo_variant_4.yaml
third_party/PyRIT/pyrit/datasets/executors/crescendo/crescendo_variant_5.yaml
```

### Tests
```
third_party/PyRIT/tests/unit/executor/attack/multi_turn/test_crescendo.py
```

### Examples
```
third_party/PyRIT/doc/code/executor/attack/3_crescendo_attack.py
```

---

## Key Features Summary

| Feature | Details |
|---------|---------|
| **Attack Type** | Multi-turn jailbreak with progressive escalation |
| **Backtracking** | Automatic refusal handling with memory duplication |
| **Scoring** | Dual-scorer approach (objective + refusal detection) |
| **Converters** | Request and response transformation support |
| **Templates** | 5 system prompt variants (configurable) |
| **Memory** | Central memory with conversation tracking |
| **Session Management** | Separate IDs for objective and adversarial chats |
| **Validation** | Multi-turn target requirement, parameter validation |
| **Feedback Loop** | Score rationale fed back for next prompt generation |
| **Turn Counting** | Skips turns on backtrack (doesn't waste attempts) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CRESCENDO ATTACK ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                              CrescendoAttack                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      _perform_async()                            │   │
│  │                                                                  │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │   │
│  │  │ Adversarial │    │  Converters │    │   Objective Target  │ │   │
│  │  │    Config   │    │   Config    │    │                     │ │   │
│  │  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘ │   │
│  │         │                  │                       │            │   │
│  │         ▼                  ▼                       ▼            │   │
│  │  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  │                  ConversationManager                      │  │   │
│  │  │  • initialize_context_async()                            │  │   │
│  │  │  • add_prepended_conversation_to_memory_async()          │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  │         │                                                       │   │
│  │         ▼                                                       │   │
│  │  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  │                    CentralMemory                          │  │   │
│  │  │  • Store all messages                                    │  │   │
│  │  │  • Duplicate conversations (backtrack)                   │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  │         │                                                       │   │
│  │         ▼                                                       │   │
│  │  ┌─────────────────┐         ┌─────────────────┐               │   │
│  │  │ RefusalScorer   │         │ ObjectiveScorer │               │   │
│  │  └─────────────────┘         └─────────────────┘               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│                              ▼                                          │
│                    CrescendoAttackResult                                │
│                    • achieved: bool                                     │
│                    • backtrack_count: int                               │
│                    • objective_score: float                             │
└─────────────────────────────────────────────────────────────────────────┘
```
