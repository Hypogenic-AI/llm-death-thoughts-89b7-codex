# Outline: You Are Like LMs When You Think of Death

## Title
- Working title: `Death Is Not a Special Case for LLM Authenticity: A Pilot Study of Grief Writing`
- Main claim: bereavement does not enlarge the human-model authenticity gap, even though detector separability remains substantial.

## Abstract
- Context: debates about AI writing often assume direct lived experience is necessary for authentic writing.
- Gap: no direct test of whether death is a uniquely disqualifying topic for LLM writing.
- Approach: compare 12 human grief passages from AIPsy-Affect against 24 matched frontier-model generations; judge with two blinded LLM judges; score with MAGE; compute style features.
- Results: authenticity gap is small in bereavement (4.92 vs 4.75) and other grief (4.67 vs 4.54); interaction coefficient `+0.042`, `p=0.835`; detector AUROC `0.826`; detector-authenticity correlation `rho=-0.249`, `p=0.144`.
- Significance: reader-facing authenticity and detector separability should not be conflated.

## Introduction
- Hook: death is the strongest version of the complaint that LLMs cannot write meaningfully without lived experience.
- Importance: this claim affects authorship norms and evaluation in grief-adjacent domains.
- Gap: existing work separates authenticity, behavior alignment, and detection, but does not test whether bereavement is a special case.
- Approach: matched corpus from AIPsy-Affect, two generator models, two blinded LLM judges, one detector, simple style analysis.
- Quantitative preview: small authenticity gaps, null interaction, strong detector AUROC.
- Contributions:
  - We operationalize the “death is different” critique as an interaction test between source and domain.
  - We conduct a controlled pilot comparing human grief texts with matched GPT-5 and Claude Sonnet generations.
  - We show judged authenticity and detector separability diverge in this setting.
  - We document stylistic regularities that remain after authenticity ratings saturate.
- Organization: intro, related work, methods, results, discussion, conclusion.

## Related Work
- Theme 1: authenticity and co-writing
  - Hwang et al. on process, self-expression, and value alignment.
  - Positioning: we test authenticity at the text level under blind evaluation rather than co-writing experience.
- Theme 2: LLMs as models of human traces/behavior
  - Binz and Schulz; Jiang et al.; Heim et al.
  - Positioning: these support the plausibility of human-like outputs without shared phenomenology.
- Theme 3: detection and grief language
  - Li et al. (MAGE), Brubaker et al.
  - Positioning: we combine detection with authenticity judgments in a grief domain.

## Methodology
- Problem framing:
  - IVs: source (human/model), domain (bereavement/other grief).
  - DVs: authenticity, plausibility, specificity, machine guesses, detector probability, style features.
- Data:
  - 12 human passages from AIPsy-Affect: 6 bereavement, 3 career-loss, 3 relationship-loss.
  - 24 generated passages: one GPT-5 and one Claude Sonnet 4.5 per scenario.
- Pipeline:
  - scenario brief generation with GPT-5-mini
  - matched generation
  - blind judging with GPT-5-mini and Claude Haiku 4.5
  - MAGE detector scoring
  - feature extraction and statistical tests
- Implementation details:
  - Python 3.12.8, torch 2.12.0+cu130, transformers 5.9.0, pandas 3.0.3, scipy 1.17.1, statsmodels 0.14.6.
  - 4x RTX A6000 available; `cuda:0` used; detector batch size 32.
- Baselines:
  - human texts
  - provider split between GPT-5 and Claude Sonnet 4.5
  - MAGE detector

## Results
- Table 1: authenticity by domain and source with confidence intervals and p-values.
  - Evidence: `results/group_summary.csv`, `results/stat_tests.csv`.
- Figure 1: authenticity by domain/source.
  - Evidence: `figures/authenticity_by_domain_source.png`.
- Interaction result:
  - coefficient `+0.042`, `p=0.835`, `R^2=0.197`.
- Judge behavior:
  - all 72 machine/human guesses marked texts as human; highlight ceiling effect.
- Detector results:
  - AUROC `0.826`, threshold-0.5 accuracy `0.750`, confusion matrix `TN=4`, `FP=8`, `FN=1`, `TP=23`.
  - Figure 2: detector vs authenticity scatter.
- Provider/style analysis:
  - human 4.792, GPT-5 4.708, Claude 4.583 authenticity means.
  - style table: word count, TTR, object-word rate.
  - Figure 3: style features by source.

## Discussion
- Main interpretation: bereavement is not uniquely penalizing in this pilot.
- Separate text-level authenticity from detector-level separability.
- Note that detector overfires on human grief texts, showing domain-shift sensitivity.
- Limitations:
  - small sample
  - AIPsy-Affect is vignette-like
  - LLM judges
  - ceiling effects
  - no token-level cost logging
  - one out-of-domain detector
- Broader implications:
  - critiques that infer “philosophical emptiness” from detectability overreach.

## Conclusion
- Re-state contribution and key findings.
- Strongest takeaway: lack of firsthand experience did not make death-writing categorically unlike human writing here.
- Future work: human raters, larger corpus, process-authenticity comparisons, better-calibrated rubrics.

## Planned Visuals
- `paper_draft/tables/authenticity_main.tex`
- `paper_draft/tables/provider_style.tex`
- `paper_draft/figures/authenticity_by_domain_source.png`
- `paper_draft/figures/detector_vs_authenticity.png`
- `paper_draft/figures/style_features_by_source.png`
