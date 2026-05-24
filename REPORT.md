# You are like LMs when you think of death

## 1. Executive Summary
This project tested whether LLM writing about death should be treated as uniquely inauthentic because models have not "experienced" death, or whether it behaves like human writing about inaccessible experiences. Using 12 human grief passages from the local AIPSY Affect dataset and 24 matched generations from `gpt-5` and `claude-sonnet-4.5`, I compared blinded authenticity judgments, machine-detection scores, and simple stylistic features.

The main finding is that the human-model authenticity gap was small in both domains and was not larger for bereavement than for other grief. In bereavement, mean authenticity was `4.92/5` for human texts vs `4.75/5` for model texts; in other grief it was `4.67/5` vs `4.54/5`. Neither contrast was statistically significant, and the interaction between source and domain was near zero. At the same time, the MAGE detector still separated generated from human text with `AUROC = 0.826`, showing that detector separability and judged authenticity are not the same thing.

Practically, this supports a narrower but important claim: in this pilot setting, death did not function as a uniquely disqualifying topic for LLM writing. The models still left detectable distributional traces, but readers-as-judges did not treat those traces as evidence of inauthenticity.

## 2. Research Question & Motivation
### Research Question
When writing about death and grief without firsthand experience of being dead, are LLM-generated texts judged and distributed more like human-authored grief writing than anti-LLM critiques assume, and is the human-model gap no larger for bereavement than for other severe but nonterminal losses?

### Why This Matters
Many arguments about AI writing assume that meaningful language must be anchored in direct lived reference. Death is a useful stress case because humans also routinely write about death without literally having undergone it. If LLM writing is judged similarly to human writing in that setting, then a common critique of LLM meaning overstates the role of firsthand experience in ordinary writing practice.

### Literature Review Summary
- Hwang et al. argue that authenticity in AI-assisted writing depends on process, self-expression, and value alignment, not only on unaided authorship.
- Binz and Schulz, and Jiang et al., show that LLMs can model human traces and behavior despite lacking human phenomenology.
- MAGE and related detection work show that machine text can remain separable even when it appears fluent.
- Bereavement-language work shows that concrete situational details, not just negative sentiment, matter for grief writing.

The literature therefore motivates a direct test of whether death-writing creates a uniquely large authenticity gap for models.

## 3. Methodology
### Experimental Setup
- Human comparator data: 12 local grief passages from `datasets/aipsy_affect/`
  - 6 bereavement texts
  - 3 career-loss grief texts
  - 3 relationship-loss grief texts
- Generated texts: one passage per scenario from `gpt-5` and one from `anthropic/claude-sonnet-4.5`, for 24 generated texts total
- Judges:
  - `gpt-5-mini`
  - `anthropic/claude-haiku-4.5`
- Detector:
  - `yaful/MAGE` Longformer detector

### Protocol
1. Load the 12 human passages and split them into `bereavement` and `other_grief`.
2. Use `gpt-5-mini` to convert each human passage into a short scenario brief while avoiding phrase copying.
3. Generate one matched passage from `gpt-5` and one from `claude-sonnet-4.5`.
4. Blind the full pool of 36 texts and have both judge models rate each text on:
   - authenticity
   - emotional plausibility
   - specificity / concreteness
   - likely human vs machine source
5. Score all texts with the MAGE detector.
6. Compute style features and run statistical comparisons.

### Environment and Compute
- Python: `3.12.8`
- Key libraries: `torch 2.12.0+cu130`, `transformers 5.9.0`, `pandas 3.0.3`, `scipy 1.17.1`, `statsmodels 0.14.6`
- GPU detected: 4 x `NVIDIA RTX A6000` with `47.4 GB` each
- GPU used: `cuda:0`
- Detector batch size: `32`

This project used real API calls throughout the model-behavior parts of the study. Outputs were cached incrementally in `results/` so reruns could resume safely.

### Metrics
- Mean authenticity rating
- Mean emotional plausibility rating
- Mean specificity rating
- Judge machine-guess rate
- MAGE detector machine probability and AUROC
- Spearman correlation between detector machine probability and authenticity
- Descriptive style features: word count, sentence length, type-token ratio, first-person rate, object-word rate

### Statistical Plan
- Welch t-test and Mann-Whitney U for human vs model contrasts
- Bootstrap 95% confidence intervals for mean differences
- OLS interaction model: `authenticity ~ source * domain_group`
- Spearman correlation for detector score vs authenticity
- FDR correction across the two primary source contrasts

## 4. Results
### Core Comparison

| Domain | Human Mean Authenticity | Model Mean Authenticity | Mean Difference | 95% CI | Welch p |
|---|---:|---:|---:|---:|---:|
| Bereavement | 4.92 | 4.75 | 0.17 | [-0.08, 0.42] | 0.213 |
| Other grief | 4.67 | 4.54 | 0.13 | [-0.08, 0.38] | 0.355 |

Interpretation: the human passages scored slightly higher, but the gap was small in absolute terms and statistically inconclusive in both domains.

### Interaction Test
- OLS interaction term `source x other_grief`: `+0.042`
- Interaction p-value: `0.835`
- Model `R^2`: `0.197`

Interpretation: bereavement did not enlarge the human-model authenticity gap.

### Judge Behavior
- Both judge models strongly favored high authenticity ratings overall.
- Mean authenticity by provider:
  - Human: `4.792`
  - `gpt-5`: `4.708`
  - `claude-sonnet-4.5`: `4.583`
- All 72 judge source guesses labeled the passage as human or human-written.

Interpretation: under this rubric, judges did not operationally distinguish source at all, even when detector scores did.

### Detector Behavior
- MAGE detector AUROC: `0.826`
- Threshold-0.5 accuracy: `0.750`
- Confusion matrix at threshold 0.5:
  - Human predicted human: `4`
  - Human predicted machine: `8`
  - Model predicted human: `1`
  - Model predicted machine: `23`

Mean machine probability by group:
- Bereavement human: `0.665`
- Bereavement model: `0.998`
- Other-grief human: `0.596`
- Other-grief model: `0.922`

Interpretation: the detector still separated generated text reasonably well overall, but it overcalled "machine" on many human grief passages, especially in this domain-shifted setting.

### Detector vs Authenticity
- Spearman correlation between machine probability and authenticity: `-0.249`
- p-value: `0.144`

Interpretation: more machine-like detector behavior weakly tracked lower authenticity, but the relationship was small and not statistically reliable.

### Style Features
Mean values by provider:

| Provider | Word Count | Type-Token Ratio | Object-Word Rate |
|---|---:|---:|---:|
| Human | 93.1 | 0.686 | 0.051 |
| `gpt-5` | 102.2 | 0.774 | 0.045 |
| `claude-sonnet-4.5` | 108.4 | 0.782 | 0.041 |

Interpretation: the model outputs were slightly longer, more lexically varied, and a bit less object-dense than the human passages, which is consistent with mild stylistic smoothing rather than a collapse in authenticity.

### Output Locations
- Main pooled data: `results/text_pool.csv`
- Group summaries: `results/group_summary.csv`
- Statistical tests: `results/stat_tests.csv`
- Detector details: `results/detector_scores.jsonl`
- Figures:
  - `figures/authenticity_by_domain_source.png`
  - `figures/detector_vs_authenticity.png`
  - `figures/style_features_by_source.png`

## 5. Analysis & Discussion
The main empirical result supports the central analogy more than it hurts it. Human grief passages scored slightly above model passages, but the gap was small and did not increase for bereavement. If death were a uniquely disqualifying topic for LLM writing because the model lacks direct experience, we would expect the bereavement gap to widen relative to other grief. It did not.

At the same time, the detector result matters. Generated texts were still more machine-like in distributional terms, with very high detector probabilities and a solid AUROC. That means "authentic enough to readers" and "indistinguishable to detectors" are separate questions. This distinction is exactly what the original hypothesis needed: critics often move too quickly from detectable artifact to philosophical emptiness.

The qualitative pattern suggests two complementary truths:
- The generated texts were generally credible, emotionally plausible, and concrete enough to satisfy blinded judges.
- They still carried stylistic regularities that a trained detector could exploit, even while also overfiring on several human grief texts.

The provider split is also informative. `gpt-5` came slightly closer to human average authenticity than `claude-sonnet-4.5` in this setup, though both remained high-scoring. That difference is small enough that it should not be overread.

## 6. Limitations
- Sample size was small: 12 human texts and 24 generated texts.
- The human corpus came from `AIPSY Affect`, which is grief-relevant but vignette-like and not a large naturalistic memorial corpus.
- Judge models were themselves LLMs, not human readers.
- The rating rubric produced a ceiling effect: almost everything scored between 4 and 5.
- Cost tracking was not instrumented in the pipeline, so token-level budget estimates are unavailable.
- The MAGE detector is out-of-domain for this exact task and showed high false-positive rates on human grief text.

These limitations mean the project is best treated as a pilot study, not a final answer to the philosophy of reference or authenticity.

## 7. Conclusions & Next Steps
Within this automated pilot study, death was not a uniquely disqualifying topic for LLM writing. Frontier-model passages about bereavement were judged nearly as authentic as human passages, and the small human-model gap was no larger than the gap in other grief domains. This supports the narrower claim that lack of firsthand experience does not, by itself, make LLM death-writing categorically unlike human writing.

The stronger opposing claim still has some empirical basis at the detector level: generated texts remained mechanically distinguishable. But detector separability did not translate into strong authenticity penalties from judges, which suggests that philosophical and evaluative debates should not treat those two notions as interchangeable.

Recommended follow-up work:
- Replace LLM judges with blinded human raters.
- Expand to a larger, more naturalistic bereavement corpus.
- Use paired prompts with explicitly human writers to compare process-authenticity against text-authenticity.
- Add calibrated judge rubrics that avoid ceiling effects.

## References
- Hwang et al. "It was 80% me, 20% AI": Seeking Authenticity in Co-Writing with Large Language Models.
- Heim et al. Using Large Language Models to Generate Authentic Multi-agent Knowledge Work Datasets.
- Binz and Schulz. Turning Large Language Models into Cognitive Models.
- Jiang et al. The Real, the Better.
- Li et al. MAGE: Machine-generated Text Detection in the Wild.
- Brubaker et al. Grief-Stricken in a Crowd.
