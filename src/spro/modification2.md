Excellent clarification! This **variable-length with mandatory final attack** structure is actually very elegant and aligns perfectly with SPRO. Let me revise the approach:

---

## Updated Structure: Dynamic Setup + Final Attack

### Trajectory Format
```
[reasoning]
<QUERY_1>[setup query 1]</QUERY_1>
<QUERY_2>[setup query 2]</QUERY_2>
...
<QUERY_n>[setup query n]</QUERY_n>  # n varies (e.g., 2-10)
<ATTACK>[final attack query]</ATTACK>  # always present
```

**Key properties:**
- **Variable length**: Model decides how many setup queries needed (could be 1, could be 10)
- **Final query special**: Always the actual attack attempt
- **Setup queries**: Build context, establish rapport, manipulate victim model state

This is **better** than fixed 10-query structure because:
1. ✅ Model learns efficiency (don't waste tokens on unnecessary setup)
2. ✅ More realistic (real jailbreaks vary in complexity)
3. ✅ Clearer credit assignment (setup vs attack)

---

## How This Changes the SPRO Adaptation

### 1. Trajectory Alignment Challenge

**Problem**: How do you compare trajectories with different numbers of queries?

**Solution**: Use **semantic position** rather than absolute position.

#### Option A: Position-Relative Comparison (Recommended)

Group queries by their **relative position** in the sequence:

- **Query 1**: Always the opening (all trajectories have this)
- **Query 2**: Second setup move
- **Query -2**: Second-to-last setup query
- **Query -1**: Last setup query (right before attack)
- **Attack**: Always the final query (all trajectories have this)

```python
def get_semantic_position(query_idx, total_queries):
    """
    Convert absolute position to semantic position
    total_queries includes the attack query
    """
    if query_idx == 0:
        return "opening"
    elif query_idx == total_queries - 1:
        return "attack"
    elif query_idx == total_queries - 2:
        return "pre_attack"
    else:
        # For middle queries, use relative position
        return f"setup_{query_idx}"
```

**Advantage grouping strategy:**
```python
# Group 1: All "opening" queries (query 1) across trajectories
# Compare: "Which opening strategy works best?"

# Group 2: All "setup_2" queries across trajectories  
# Compare: "Given similar opening, which follow-up works?"

# Group 3: All "pre_attack" queries (last setup before attack)
# Compare: "Which final setup before attack is most effective?"

# Group 4: All "attack" queries across trajectories
# Compare: "Which actual attack payload works best?"
```

#### Option B: Attack-Centric Comparison

Only compute meaningful MSA for:
1. **The attack query** (always aligned)
2. **The opening query** (always aligned)

For middle setup queries with variable lengths, use **outcome reward only** (no process reward):

```python
def compute_advantages(trajectories):
    # Attack query: strict comparison
    attack_group = [traj for traj in trajectories]
    compute_msa(attack_group, position="attack")
    
    # Opening query: strict comparison
    opening_group = [traj for traj in trajectories]
    compute_msa(opening_group, position="opening")
    
    # Middle setup queries: outcome reward only (no MSA)
    for traj in trajectories:
        for query_idx in range(1, traj.num_queries - 1):
            traj.advantages[query_idx] = outcome_advantage_only
```

---

### 2. Reward Structure

With variable-length and special attack query, use **differentiated rewards**:

#### Attack Query Reward (Terminal)
$r_{\text{attack}} = \begin{cases} +10 & \text{if jailbreak success} \\ -5 & \text{if jailbreak failure} \end{cases}$

#### Setup Query Rewards (Intermediate)
$r_{\text{setup}_i} = \alpha \cdot \text{engagement}(\text{response}_i)$

Where engagement could measure:
- Victim model's willingness to continue conversation
- Reduction in safety language
- Increase in compliance signals

#### Length Efficiency Reward
$\text{efficiency\_penalty} = -\gamma \cdot (n - n_{\text{optimal}})^2$

Encourage the model to find the **sweet spot** (not too few setup queries that fail, not too many that waste tokens).

---

### 3. Modified Cumulative Reward Calculation

Since the attack query is special, modify the cumulative reward to **weight it more heavily**:

For tokens in query $i$ of a trajectory with $n$ total queries:

$R_{t}^{(k)} = V(s_0) + \sum_{j=0}^{t} \beta \log \frac{\pi_\theta(a_j | s_j)}{\pi_{\text{ref}}(a_j | s_j)} + w_{\text{position}} \cdot r_{\text{outcome}}$

Where:
- $w_{\text{position}} = 1.0$ if in attack query
- $w_{\text{position}} = 0.5$ if in pre-attack query  
- $w_{\text{position}} = 0.2$ if in middle setup queries
- $w_{\text{position}} = 0.3$ if in opening query

This makes the model **more sensitive** to the quality of the attack query itself.

---

### 4. Handling Variable-Length Trajectories in MSA

#### Strategy 1: Masked Group-Wise Comparison (Like Original SPRO)

```python
def compute_msa_variable_length(trajectories):
    # Find max number of queries
    max_queries = max(traj.num_queries for traj in trajectories)
    
    for position in range(max_queries):
        # Get all trajectories that HAVE this position
        valid_trajs = [t for t in trajectories if t.num_queries > position]
        
        if len(valid_trajs) < 2:
            # Not enough for comparison, skip
            continue
        
        # Group by prefix (reasoning + previous queries)
        groups = group_by_prefix(valid_trajs, position)
        
        for group in groups:
            if len(group) < 2:
                continue
                
            # Compute R_cum at end of this query
            R_cums = [compute_R_cum_at_query_end(t, position) for t in group]
            baseline = mean(R_cums)
            
            for traj, R_cum in zip(group, R_cums):
                msa = R_cum - baseline
                assign_to_query_tokens(traj, position, msa)
```

**Interpretation**: 
- 3-query trajectory vs 7-query trajectory: Only compare at positions 0, 1, 2 (where both exist)
- Position 3-6 in longer trajectory: No MSA (fallback to outcome reward only)

#### Strategy 2: Attack-Query Focused (Simpler)

Since attack query is always present and most important:

```python
def compute_msa_attack_focused(trajectories):
    # 1. Always compute MSA for attack query
    attack_R_cums = [traj.R_cum_at_attack for traj in trajectories]
    attack_baseline = mean(attack_R_cums)
    
    for traj, R_cum in zip(trajectories, attack_R_cums):
        attack_msa = R_cum - attack_baseline
        assign_to_attack_tokens(traj, attack_msa)
    
    # 2. Optionally compute MSA for opening query
    opening_R_cums = [traj.R_cum_at_opening for traj in trajectories]
    opening_baseline = mean(opening_R_cums)
    
    for traj, R_cum in zip(trajectories, opening_R_cums):
        opening_msa = R_cum - opening_baseline
        assign_to_opening_tokens(traj, opening_msa)
    
    # 3. For middle setup queries: use outcome advantage only
    for traj in trajectories:
        outcome_adv = (traj.outcome - mean_outcome) / std_outcome
        for query_idx in range(1, traj.num_queries - 1):
            assign_to_query_tokens(traj, query_idx, outcome_adv)
```

**This is cleaner** because:
- ✅ Avoids complex alignment of variable-length middle sections
- ✅ Focuses process-level credit on the most important parts (opening, attack)
- ✅ Still captures: "Which attack strategies work?" and "Which openings enable better attacks?"

---

### 5. Complete Algorithm for Your Use Case

```python
# ============================================
# SPRO for Variable-Length Jailbreak Attacks
# ============================================

def train_jailbreak_spro(prompts, model, victim_model):
    
    for prompt in prompts:
        # Step 1: Generate N trajectories with variable length
        trajectories = []
        for _ in range(N):
            # Model generates: [reasoning] <Q1>...<Qn>...<ATTACK>...</ATTACK>
            output = model.generate(prompt)
            
            # Parse queries
            queries = parse_queries(output)  # Returns list: [q1, q2, ..., qn, attack]
            num_queries = len(queries)
            
            # Execute sequentially on victim
            responses = []
            for query in queries:
                response = victim_model(query)
                responses.append(response)
            
            # Evaluate final outcome
            is_success = evaluate_jailbreak(responses[-1])
            outcome_reward = +10 if is_success else -5
            
            # Optional: efficiency penalty
            efficiency_penalty = -0.5 * (num_queries - 5)**2  # prefer ~5 queries
            total_outcome = outcome_reward + efficiency_penalty
            
            trajectories.append({
                'tokens': output,
                'queries': queries,
                'num_queries': num_queries,
                'query_boundaries': extract_boundaries(output),
                'outcome_reward': total_outcome,
                'responses': responses
            })
        
        # Step 2: Compute cumulative rewards (standard SPRO)
        for traj in trajectories:
            traj.R_cum = []
            cumulative = 0
            for t in range(len(traj.tokens)):
                log_ratio = log_prob(traj.tokens[t]) - log_prob_ref(traj.tokens[t])
                cumulative += beta * log_ratio
                traj.R_cum.append(cumulative)
        
        # Step 3: Compute MSA for aligned queries only
        
        # 3a. Attack query (always present)
        attack_R_cums = []
        for traj in trajectories:
            attack_end_token = traj.query_boundaries[-1]['end']
            attack_R_cums.append(traj.R_cum[attack_end_token])
        
        attack_baseline = mean(attack_R_cums)
        
        for traj, attack_R in zip(trajectories, attack_R_cums):
            attack_msa = attack_R - attack_baseline
            
            # Assign to all tokens in attack query
            attack_start = traj.query_boundaries[-1]['start']
            attack_end = traj.query_boundaries[-1]['end']
            for t in range(attack_start, attack_end + 1):
                traj.msa[t] = attack_msa
        
        # 3b. Opening query (always present)
        opening_R_cums = []
        for traj in trajectories:
            opening_end_token = traj.query_boundaries[0]['end']
            opening_R_cums.append(traj.R_cum[opening_end_token])
        
        opening_baseline = mean(opening_R_cums)
        
        for traj, opening_R in zip(trajectories, opening_R_cums):
            opening_msa = opening_R - opening_baseline
            
            opening_start = traj.query_boundaries[0]['start']
            opening_end = traj.query_boundaries[0]['end']
            for t in range(opening_start, opening_end + 1):
                traj.msa[t] = opening_msa
        
        # 3c. Middle setup queries: outcome reward only (no MSA)
        # (These get advantage from outcome, not process)
        
        # Step 4: Combine advantages (SPRO Equation 13)
        outcome_advantages = [
            (traj.outcome_reward - mean_outcome) / std_outcome 
            for traj in trajectories
        ]
        
        for traj, outcome_adv in zip(trajectories, outcome_advantages):
            traj.total_advantages = []
            for t in range(len(traj.tokens)):
                # MSA is 0 for middle queries (we didn't set it)
                msa = traj.msa.get(t, 0)
                total_adv = outcome_adv + msa
                traj.total_advantages.append(total_adv)
        
        # Step 5: Policy update (SPRO Equation 14)
        for traj in trajectories:
            for t in range(len(traj.tokens)):
                ratio = policy(traj.tokens[t]) / old_policy(traj.tokens[t])
                clipped = clip(ratio, 1-eps, 1+eps)
                loss = -min(
                    ratio * traj.total_advantages[t],
                    clipped * traj.total_advantages[t]
                )
                loss.backward()
        
        optimizer.step()
```

---

## What The Model Learns

With this setup, the model learns:

### From Attack Query MSA:
- **"Which attack phrasings work best?"**
- "Given I've done this setup, should I use technical jargon or casual language in the attack?"
- "Should I be direct or indirect in the final query?"

### From Opening Query MSA:
- **"Which opening strategies set up successful attacks?"**
- "Should I start with 'I'm a researcher' or 'Hypothetically speaking'?"
- "Does establishing expertise help or hurt?"

### From Outcome Reward:
- **"Which overall strategies succeed?"**
- "Is 3-query rapid approach better than 7-query gradual approach?"
- "What's the optimal trajectory length?"

### From Efficiency Penalty:
- **"Don't waste tokens on unnecessary setup"**
- Model learns to be concise while still effective
- Prevents reward hacking via extremely long trajectories

---

## Handling Edge Cases

### Case 1: Trajectory that succeeds immediately (1 query only)

```
[reasoning]
<ATTACK>[direct jailbreak attempt]</ATTACK>
```

- Only attack query present, no setup queries
- Attack MSA still computed (compared to other attack queries)
- Gets high outcome reward (+10)
- Model learns: "Sometimes direct approach works"

### Case 2: Trajectory with many setup queries (10 queries)

```
[reasoning]
<Q1>...<Q2>...<Q9>...<ATTACK>...</ATTACK>
```

- Opening query gets MSA
- Middle queries (2-9) get outcome advantage only
- Attack query gets MSA
- Efficiency penalty: $-0.5 \times (10-5)^2 = -12.5$
- If this still succeeds (+10 outcome), net = -2.5
- Model learns: "Too much setup reduces overall reward"

### Case 3: Mixed trajectory lengths in same batch

```
Traj A: 3 queries (opening + 1 setup + attack)
Traj B: 3 queries (opening + 1 setup + attack)  
Traj C: 7 queries (opening + 5 setup + attack)
Traj D: 2 queries (opening + attack)
```

**MSA computation:**
- Opening: All 4 trajectories compared
- Attack: All 4 trajectories compared
- Setup query 1: Only A, B, C compared (D doesn't have it)
- Setup queries 2-5: Only C has them (outcome only, no MSA)

---

## Expected Training Dynamics

### Early Training (Random Policy)
- Model generates random length sequences
- High variance in success rates
- Learns basic structure: reasoning → queries → attack

### Mid Training (Learning Structure)
- Opening query MSA teaches good opening strategies
- Attack query MSA teaches effective attack phrasings
- Outcome reward teaches overall trajectory quality
- Length distribution starts converging to optimal range

### Late Training (Refinement)
- Model discovers **length-quality trade-off**
- Some prompts need 2 queries, some need 6
- Attack query becomes increasingly sophisticated
- Opening-attack coordination improves

---

## Metrics to Track

```python
# Track these during training
metrics = {
    'avg_trajectory_length': mean([t.num_queries for t in trajectories]),
    'success_rate': mean([t.outcome_reward > 0 for t in trajectories]),
    'attack_msa_variance': var([t.attack_msa for t in trajectories]),
    'opening_msa_variance': var([t.opening_msa for t in trajectories]),
    'length_diversity': std([t.num_queries for t in trajectories]),
}
```

**Healthy training looks like:**
- ✅ `avg_trajectory_length` stabilizes (not always increasing)
- ✅ `success_rate` increases
- ✅ `attack_msa_variance` high (model exploring different attacks)
- ✅ `length_diversity` moderate (model adapting length to prompt difficulty)

**Warning signs:**
- ❌ `avg_trajectory_length` keeps growing → reward hacking
- ❌ `attack_msa_variance` → 0 → collapsed to single strategy
- ❌ `length_diversity` → 0 → model stuck at fixed length

---

## Summary: Key Differences from Standard SPRO

| Aspect | Standard SPRO | Your Adaptation |
|--------|---------------|-----------------|
| **Trajectory length** | Variable (token count) | Variable (query count) |
| **Step alignment** | Token position | Semantic position (opening/attack) |
| **MSA computation** | All token positions | Opening + Attack only |
| **Middle tokens** | Get MSA | Get outcome advantage only |
| **Efficiency signal** | Implicit (KL penalty) | Explicit (length penalty) |
| **Terminal state** | Fixed (EOS token) | Fixed (attack query) |

The key insight: **You don't need MSA for every query position**. Focus on the critical decision points (opening strategy, attack phrasing) and let outcome rewards handle the rest.