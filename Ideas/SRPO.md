<alphaxiv-thinking-title title="Reading paper details" />

To transition from GRPO to SPRO, you'll need to make **two key modifications**:

## 1. Add Cumulative Reward Calculation (Eq. 11)

At each timestep **t** in a trajectory, compute:

$$\tilde{R}_{i,t} = \sum_{j=0}^{t} \beta \log \frac{\pi_\theta(a_j|s_j)}{\pi_{\text{ref}}(a_j|s_j)}$$

where:
- $i$ indexes the response
- $\beta$ is the KL coefficient (they use β=0 + separate KL penalty in practice)
- This accumulates log-probability ratios from the start of the sequence up to step **t**

**Implementation tip**: You can compute this efficiently by taking the cumulative sum of per-token log-prob ratios.

## 2. Replace Advantage with MSA + Outcome (Eq. 13)

Change from GRPO's pure outcome advantage:

$$A_i^{\text{GRPO}} = \frac{r_o(y_i) - \text{mean}(\{r_o(y_i)\})}{\text{std}(\{r_o(y_i)\})}$$

to SPRO's hybrid advantage:

$$A_{i,t} = \frac{r_o(y_i) - \text{mean}(\{r_o(y_i)\})}{\text{std}(\{r_o(y_i)\})} + \left(\tilde{R}_{i,t} - \text{masked\_mean}(\{\tilde{R}_{i,t}\})\right)$$

The **masked_mean** operates vertically across responses at the **same timestep t**, excluding padding/empty positions (see Fig. 3).

---

## Practical Changes Summary

| Component | GRPO | SPRO |
|-----------|------|------|
| **Advantage granularity** | Trajectory-level (one value per response) | Token-level (one value per timestep) |
| **Process signal** | None | Cumulative log-prob ratios $\tilde{R}_{i,t}$ |
| **Baseline** | Mean across responses | Outcome mean + masked mean at each step |
| **Loss computation** | Same advantage for all tokens in trajectory | Different advantage per token |

---

## Implementation Notes

1. **Masking**: When responses have different lengths, you need to mask out invalid positions when computing `masked_mean` at each step (responses that have already terminated don't contribute to the baseline)

2. **No extra models**: Unlike PRIME, you don't need a separate PRM—the process rewards come directly from your policy vs reference model log-probs

3. **Memory**: Store per-token advantages instead of per-trajectory, but no additional model loading required

4. **No normalization on MSA**: Notice the outcome advantage is normalized by std, but the MSA term is just centered (no std division)

The elegance of SPRO is that it requires **minimal architectural changes**—just changing how you compute advantages before feeding them into the standard PPO-clipped objective.