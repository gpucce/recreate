# Study design: perceived AI-ness across iterated webpage generations

Oct 28, 2026 · @Giovanni

The core question: does perceived AI-ness increase with the number of AI-to-AI generation hops? This is an iterated-generation design, a "telephone game" for webpages, run as a rating study on Prolific.

## 1. Key variable: AI depth

AI depth is the number of consecutive AI generations since a page's last human-made ancestor. The pages form a lineage tree, and each page's parent may be human-made or AI-made.

- A page generated directly from a human page has depth 1.
- A page generated from that page has depth 2, and so on.
- Human seed pages are depth 0. Include them as stimuli: they anchor the scale and show whether depth 1 already looks "AI" or the drift is gradual.

## 2. Make depth the only thing that varies

The main risk is that depth gets confounded with other factors.

**Generation setup**

- Keep the model, prompt and settings identical across all iterations. If deep pages came from a newer model or different prompt, differences can't be attributed to iteration.
- Use 20 or more independent human seed pages across different site types.
- Run each seed to fixed depths, e.g. 0, 1, 2, 4, 8 (roughly log-spaced, since most drift tends to happen early).

**Coverage**

- Run 2–3 parallel chains per seed. Generation is stochastic, and replicates separate systematic drift from chain-specific noise.
- If the existing tree is uneven (mixed depths and lineages), generate a clean, balanced set of chains for the study. Use the existing tree as exploratory data only, since which branches got extended may be confounded with depth.

**Rendering**

- Render every page identically: same viewport, screenshot pipeline and fonts.

## 3. Two complementary tasks

| Task | What participants do | Key constraint | Strength |
| --- | --- | --- | --- |
| A. Absolute ratings (main) | Rate single pages: "How likely is it that this webpage was created by AI?" (1–7) | At most one page per chain per participant; balanced lists so every depth of every chain gets rated | Clean between-page estimates of AI-ness by depth |
| B. Paired comparisons | See two pages from the same chain at different depths (random left/right): "Which one looks more AI-generated?" | Separate participants, or a second block after Task A | Controls content and site type fully; very sensitive to small drifts |

**Secondary measures for both tasks**

- Perceived quality, and optionally originality. Iteration may simply degrade pages, so test whether the AI-ness effect holds after controlling for quality.
- Optional "why?" free text on some trials, to learn which features accumulate (generic gradients, card grids, filler copy).

## 4. Analysis

**Task A** — ordinal (cumulative link) mixed model:

```
rating ~ depth + (1 + depth | participant) + (1 + depth | seed/chain)
```

- Fit depth as continuous (or log(depth + 1)), and also as categorical to see the curve's shape.
- The random slope for seed shows whether some kinds of sites drift faster than others.

**Task B** — logistic mixed model on "chose the deeper page", with depth difference as predictor. Alternatively, a Bradley-Terry model places every page on one AI-ness scale.

**Optional objective complement** — embed all screenshots (e.g. with CLIP) and test whether pages from different seeds converge as depth increases. Rising perceived AI-ness plus convergence would suggest an "AI default aesthetic" emerging through iteration.

## 5. Practical numbers

| Parameter | Starting value |
| --- | --- |
| Seeds | 20 |
| Depths per chain | 5 (0, 1, 2, 4, 8) |
| Chains per seed | 2 |
| Total pages | 200 |
| Pages rated per participant | 20 (one per chain, mixed depths) |
| Participants | 150 |
| Ratings per page | about 15 |

Confirm these with a simulation-based power analysis after a pilot of about 20 people.

Prolific setup: desktop-only filter, attention checks, a pilot run, and preregistration before the main sample.
