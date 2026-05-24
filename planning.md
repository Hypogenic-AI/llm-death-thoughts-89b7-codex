# Research Plan: You are like LMs when you think of death

## Motivation & Novelty Assessment

### Why This Research Matters
Debates about whether LLM writing is "real" often assume that meaningful writing must be grounded in direct lived reference. That assumption matters because it shapes policy, pedagogy, authorship norms, and how people evaluate AI-mediated expression in sensitive domains such as grief, memorialization, and existential reflection.

### Gap in Existing Work
The literature review shows adjacent but incomplete evidence. Authenticity work studies co-writing and perceived voice; human-behavior alignment work shows that LLMs can model human traces; detection work measures separability of human and model text; bereavement-language work studies grief markers in human discourse. What is missing is a direct test of whether death-related writing is a special case where lack of firsthand experience should disqualify LLM writing, or whether human and model writing are judged similarly once both are writing about experiences they have not literally undergone.

### Our Novel Contribution
This project tests a sharper claim than generic "AI authenticity" debates. We compare human-authored grief texts and real frontier-model generations under matched scenarios, then ask whether the human-model gap is larger for bereavement than for other severe losses. If death is not uniquely penalizing for model writing, that supports the user's core analogy: humans also routinely produce meaningful writing about unexperienced phenomena.

### Experiment Justification
- Experiment 1: Build a matched corpus of grief scenarios and frontier-model generations.
  Needed because the hypothesis concerns actual writing outputs, not abstract claims.
- Experiment 2: Run blinded authenticity judgments with real LLM judges.
  Needed because authenticity is partly perceptual in the literature and cannot be reduced to raw detector scores.
- Experiment 3: Run a machine-text detector and compare its outputs to judge ratings.
  Needed because human/LLM discourse often conflates "detectable as machine-like" with "inauthentic."
- Experiment 4: Compute lightweight stylistic features and qualitative error analysis.
  Needed to see which textual cues drive any remaining gap.

## Research Question
When writing about death and grief without firsthand experience of being dead, are LLM-generated texts judged and distributed more like human-authored grief writing than anti-LLM critiques assume, and is the human-model gap no larger for bereavement than for other severe but nonterminal losses?

## Background and Motivation
Prior work suggests that authenticity is socially and rhetorically constructed rather than reducible to unaided firsthand reporting. Hwang et al. frame authenticity around self-expression, source, and value alignment. Binz and Schulz, plus Jiang et al., show that LLMs can align to human behavior traces despite lacking human phenomenology. Detection work such as MAGE and Chakraborty et al. shows that human and model text remain separable in many settings, but this does not settle whether model writing about inaccessible experiences is uniquely defective. The present study addresses that gap directly in a death-writing setting.

## Hypothesis Decomposition
- H1: Frontier-model grief texts will receive lower authenticity ratings than human grief texts overall, but the gap will be modest rather than categorical.
- H2: The human-model authenticity gap will not be larger in bereavement than in other grief domains such as abandonment or career-loss grief.
- H3: Detector-based machine-likeness and judged authenticity will be only moderately correlated, showing that "machine-like" and "inauthentic" are not equivalent.
- H4: LLM grief texts will match human texts on several surface cues linked to authenticity in grief discourse, especially concreteness and situational specificity, but will still show some stylistic regularization.

Independent variables:
- Source: human, GPT-5, second frontier model via OpenRouter
- Domain: bereavement, other grief
- Judge: GPT-5 judge, second judge model via OpenRouter

Dependent variables:
- Authenticity rating
- Emotional plausibility rating
- Specificity / concreteness rating
- Judge guess of human vs machine source
- MAGE detector human-probability / machine-probability
- Lightweight stylistic features

Alternative explanations:
- Judges may simply reward polished prose.
- Small bereavement corpus may limit power.
- AIPSY texts may be vignette-like rather than naturalistic grief testimony.
- Detector confidence may reflect training-domain artifacts rather than authenticity.

## Proposed Methodology

### Approach
Use the local `AIPSY Affect` dataset as the human comparator set and scenario seed source, because it contains explicit bereavement and grief texts. Build a small but tightly controlled matched corpus: 18 human texts across three grief domains, then use a real frontier model to derive neutral scenario prompts from those texts and generate matched responses from two frontier models. Evaluate the full blinded pool with two independent LLM judges, a pretrained MAGE detector, and simple stylistic features.

This approach is preferred over a pure detector benchmark because the research question is normative and rhetorical, not just classificatory. It is preferred over a human-subject study because the session must be fully automated and completed end-to-end in this environment.

### Experimental Steps
1. Load the local datasets and select 18 human grief texts.
   Rationale: keeps the study anchored in locally available data and balanced by domain.
2. Split domains into `bereavement` and `other_grief`.
   Rationale: directly operationalizes whether death is a special case.
3. Use a frontier model to convert each human text into a short scenario brief with target length/style constraints.
   Rationale: gives matched prompts without copying human wording verbatim into generation prompts.
4. Generate one response per scenario from GPT-5 and one second-model response via OpenRouter.
   Rationale: tests whether the effect generalizes beyond one model family.
5. Blind the pooled texts and score them with two judge models on authenticity, plausibility, specificity, and human/machine guess.
   Rationale: follows the literature's emphasis on perceived authenticity.
6. Score the same texts with the MAGE detector and extract stylistic features.
   Rationale: separates detector behavior from authenticity judgments.
7. Run statistical comparisons and qualitative error analysis.
   Rationale: tests the hypothesis and identifies where the analogy breaks down.

### Baselines
- Human-authored AIPSY grief texts
- Source comparison between GPT-5 and the second frontier model
- MAGE detector as a machine-likeness baseline
- Simple stylistic features as non-neural descriptive baselines

### Evaluation Metrics
- Mean authenticity rating on a 1-5 scale
- Mean emotional plausibility rating on a 1-5 scale
- Mean specificity/concreteness rating on a 1-5 scale
- Judge human-guess rate
- Detector machine probability and AUROC for human vs generated text
- Correlation between detector scores and authenticity ratings
- Descriptive stylistic features: length, sentence length, type-token ratio, first-person pronoun rate, lexical concreteness proxy via object-word counts

### Statistical Analysis Plan
- Alpha: 0.05, two-sided
- Primary comparison: mean authenticity gap between human and generated texts within `bereavement` and `other_grief`
- Tests:
  - Welch t-test and Mann-Whitney U for robustness
  - Bootstrap 95% confidence intervals for mean differences
  - Two-way OLS / ANOVA style analysis on authenticity with factors `source` and `domain`
  - Spearman correlation between detector score and authenticity
  - Fisher exact test on judge human/machine guesses where appropriate
- Multiple comparisons:
  - Benjamini-Hochberg correction across secondary outcomes

## Expected Outcomes
Results would support the hypothesis if:
- LLM bereavement texts are often judged as emotionally plausible and moderately authentic.
- The authenticity gap between human and model text is not larger for bereavement than for other grief.
- Detector scores and authenticity judgments diverge meaningfully.

Results would weaken the hypothesis if:
- Bereavement amplifies the human-model gap substantially relative to other grief.
- Both judges and detector cleanly separate all LLM death-writing from human texts.
- LLM texts systematically lack the concrete situational detail present in human comparators.

## Timeline and Milestones
1. Planning and design finalization: complete immediately in `planning.md`
2. Environment and dependency setup: 10-20 minutes
3. Pipeline implementation: 45-75 minutes
4. API generation and judging runs: 20-40 minutes
5. Analysis and figures: 20-30 minutes
6. Documentation and validation: 20-30 minutes

## Potential Challenges
- API model availability may differ across providers.
  Mitigation: use OpenAI for the primary model and query OpenRouter for an available second model.
- The human corpus is small.
  Mitigation: use balanced sampling, bootstrap intervals, and cautious claims.
- MAGE detector download or inference may be slow.
  Mitigation: use GPU inference and cache all scores.
- AIPSY may not represent naturalistic memorial discourse.
  Mitigation: state this explicitly and treat conclusions as a targeted pilot study.

## Success Criteria
- A reproducible pipeline exists under `src/` and outputs are saved in `results/`.
- The study runs on real frontier-model API calls, not simulated agents.
- `REPORT.md` contains actual quantitative results, figures, and limitations.
- The final interpretation directly answers whether death is a uniquely disqualifying topic for LLM writing under the tested conditions.
