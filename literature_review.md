# Literature Review: You are like LMs when you think of death

## Review Scope

### Research Question
When large language models write about death or grief without lived experience, should that writing be treated as analogous to human writing about unexperienced phenomena, especially where humans also routinely write authentically about events they have not personally undergone?

### Inclusion Criteria
- Papers on authenticity, authorship, or voice in AI-assisted writing
- Papers on LLMs as models of human behavior or cognition
- Papers or datasets for human-vs-LLM text comparison
- Papers on linguistic analysis of bereavement, grief, or distress writing

### Exclusion Criteria
- Purely technical LLM papers without relevance to human expression or authorship
- Medical grief papers without textual/linguistic data
- Detection resources with no accessible paper, dataset, or code

### Time Frame
- Core AI literature: 2023-2026
- Background bereavement language work: older foundational studies included when directly relevant

### Sources
- arXiv
- ACL Anthology
- OpenAlex
- Hugging Face datasets
- GitHub repositories linked from papers

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-05-24 | `large language models authenticity writing experience` | OpenAlex | mixed | surfaced the Hwang paper |
| 2026-05-24 | `all:"large language models" AND all:authenticity` | arXiv API | focused | surfaced Hwang, Heim et al., authorship work |
| 2026-05-24 | `all:"large language models" AND all:"human behavior"` | arXiv API | focused | surfaced Binz and RLHB-related work |
| 2026-05-24 | `all:"machine-generated text detection"` | arXiv API | broad | surfaced benchmark and detector papers |
| 2026-05-24 | `grief bereavement text dataset social media` | OpenAlex / web | sparse | led to Brubaker et al. as the best accessible language paper |

## Screening Results

| Paper | Title Screen | Abstract Screen | Full-Text | Notes |
|------|------|------|------|------|
| Hwang et al. 2024 | Include | Include | Include | Most direct authenticity paper |
| Heim et al. 2024 | Include | Include | Include | Helpful for operationalizing "authentic" synthetic text |
| Binz and Schulz 2023 | Include | Include | Include | Strong analogy to human behavior modeling |
| Jiang et al. 2024 | Include | Include | Include | Human-behavior alignment method |
| Chakraborty et al. 2023 | Include | Include | Include | Detection theory and limits |
| Li et al. 2024 | Include | Include | Include | Benchmark and dataset for human/LLM discrimination |
| Brubaker et al. 2012 | Include | Include | Include | Death/bereavement writing reference |

## Research Area Overview

The literature splits into three useful strands. First, authenticity-in-writing work shows that people evaluate authenticity through process, identity, and audience relationship rather than only through factual lived experience. Second, cognitive-modeling and behavior-alignment work shows that LLMs can approximate human behavior patterns when trained or aligned against human traces. Third, text-detection and bereavement-language work gives operational tools for testing whether model writing about death is statistically or perceptually separable from human writing.

This means the current evidence does not directly prove the hypothesis, but it does support a credible experiment design: compare human-authored death-related writing, human-authored writing about other unexperienced phenomena, and LLM-authored writing under matched prompts and evaluate them on authenticity judgments, stylistic markers, and detector behavior.

## Key Papers

### Hwang et al. 2024/2025: Seeking Authenticity in Co-Writing with LLMs
- Authors: Angel Hsing-Chi Hwang et al.
- Source: arXiv / CSCW 2025
- Key Contribution: Shows that writers define authenticity as tied to source, authentic self, content authenticity, and value alignment; authenticity is not reducible to whether all words were unaided.
- Methodology: Semi-structured interview study with 19 professional writers plus online survey with 30 avid readers; compared personalized vs non-personalized AI writing support.
- Datasets Used: Original study materials and writer-produced passages; no reusable benchmark dataset released in the paper PDF.
- Results:
  - Most writers preferred personalized AI support.
  - Personalization helped preserve voice under time pressure.
  - Writers centered authenticity in the process of expression and in self-construction.
  - Reader reactions were less uniformly hostile than writers expected.
- Code Available: Not identified in accessible sources.
- Relevance to Our Research: This is the strongest conceptual anchor for arguing that authenticity can survive mediation by tools and that lived process matters as much as direct experience.

### Heim et al. 2024: Using LLMs to Generate Authentic Multi-agent Knowledge Work Datasets
- Authors: Desiree Heim et al.
- Source: arXiv
- Key Contribution: Introduces a configurable multi-agent generator for synthetic knowledge-work documents plus a knowledge graph of the simulated context.
- Methodology: Multi-agent simulation of workplace tasks and documents, followed by human realism judgments.
- Datasets Used: Synthetic knowledge-work traces generated by the proposed system.
- Results:
  - Human raters marked 53% of generated documents and 74% of real documents as realistic.
  - The gap shows synthetic texts can appear plausible but still fall short of real-world authenticity.
- Code Available: Not located from paper metadata.
- Relevance to Our Research: Useful contrast case. It operationalizes authenticity as realism and highlights that plausibility alone is weaker than the richer notion of authenticity raised by the main hypothesis.

### Binz and Schulz 2023: Turning Large Language Models into Cognitive Models
- Authors: Marcel Binz, Eric Schulz
- Source: arXiv
- Key Contribution: Demonstrates that LLMs fine-tuned on psychological experiment data can outperform traditional cognitive models in predicting human behavior.
- Methodology: Fine-tuned LLaMA-derived models on behavioral experiment traces; evaluated on decision-making tasks and unseen-task generalization.
- Datasets Used: Psychological experiment datasets including `choices13k` and the horizon task.
- Results:
  - Fine-tuning moved model behavior substantially closer to human choice curves and regret levels.
  - Learned representations could model behavior at the individual-subject level.
  - Multi-task tuning improved transfer to unseen tasks.
- Code Available: Yes, see `code/Llama-3.1-Centaur-70B/` for the later follow-on repository.
- Relevance to Our Research: Strong evidence for the analogy that LLMs can become models of human-like behavior even when they do not share human phenomenology.

### Jiang et al. 2024: The Real, the Better
- Authors: Guanying Jiang et al.
- Source: arXiv
- Key Contribution: Proposes Reinforcement Learning with Human Behavior (RLHB), aligning LLMs directly to observed online behavior.
- Methodology: Generator-discriminator framework using triplets of query, response, and human behavior from online environments.
- Datasets Used: Real online behavior traces collected in a search setting.
- Results: Human and automatic evaluation both favored the behavior-aligned method over baselines.
- Code Available: Not identified from accessible sources.
- Relevance to Our Research: Supports the claim that human behavioral traces can be learned into LLM output distributions without requiring firsthand experience.

### Chakraborty et al. 2023: On the Possibilities of AI-Generated Text Detection
- Authors: Souradip Chakraborty et al.
- Source: arXiv
- Key Contribution: Gives a theoretical account of when human-vs-AI text detection is possible and how sample complexity grows as text distributions converge.
- Methodology: Information-theoretic analysis plus empirical tests across XSum, SQuAD, IMDb, and FakeNews.
- Datasets Used: XSum, SQuAD, IMDb, Kaggle FakeNews.
- Results:
  - Detection remains feasible unless human and machine distributions become indistinguishable over their support.
  - Longer samples substantially improve detection power.
- Code Available: Not identified from accessible sources.
- Relevance to Our Research: Important for experiment design because apparent "authenticity" may simply reflect detector weakness at short lengths; sample length must be controlled.

### Li et al. 2024: MAGE
- Authors: Yafu Li et al.
- Source: ACL 2024
- Key Contribution: Builds a large, cross-domain, cross-model machine-generated text benchmark with increasingly difficult OOD settings.
- Methodology: Aggregates human texts from 10 datasets and machine texts from 27 LLMs; evaluates multiple detectors.
- Datasets Used: 447,674 total examples across diverse writing tasks and models.
- Results:
  - Distinguishing human from machine text becomes much harder under OOD conditions.
  - The top detector still identified 86.54% of out-of-domain texts from a new LLM.
- Code Available: Yes, `code/MAGE/`
- Relevance to Our Research: Best currently available benchmark for operational human-vs-LLM text discrimination and for stress-testing claims of indistinguishability.

### Brubaker et al. 2012: Grief-Stricken in a Crowd
- Authors: Jed R. Brubaker et al.
- Source: ICWSM
- Key Contribution: Provides a linguistic analysis of bereavement-related distress in social-media memorial writing.
- Methodology: Manual coding of emotionally distressed content plus linguistic feature analysis on post-mortem social media comments.
- Datasets Used: Messages posted to deceased MySpace users' profiles.
- Results:
  - Distress is signaled by more than sentiment words alone; linguistic style contributes discriminative information.
  - The work motivates automatic detection of bereavement-related distress.
- Code Available: No public code found.
- Relevance to Our Research: Gives a concrete human-language reference for what death-related authentic distress can look like in naturally occurring text.

## Common Methodologies

- Human-subject qualitative evaluation: Hwang et al., Heim et al.
- Behavioral fitting or alignment to human traces: Binz and Schulz; Jiang et al.
- Benchmark-driven human-vs-LLM discrimination: Li et al.; Chakraborty et al.
- Linguistic feature analysis of grief discourse: Brubaker et al.

## Standard Baselines

- Longformer or RoBERTa-style supervised detectors for AI-generated text
- DetectGPT-style zero-shot detectors
- Human realism or authenticity judgments from readers/raters
- Linguistic feature baselines such as sentiment, pronouns, tense, and style markers in grief writing

## Evaluation Metrics

- Human authenticity ratings
- Human preference / likeability / perceived ownership ratings
- Detector accuracy, AUROC, and OOD robustness
- Text-length-controlled distinguishability
- Style-feature divergence between human and model texts

## Datasets in the Literature

- `MAGE`: Best accessible benchmark for human vs machine text
- Psychological experiment traces (`choices13k`, horizon task): useful precedent for behavior modeling, not for death writing
- MySpace memorial comments in Brubaker et al.: conceptually ideal but not released as an easy benchmark
- `AIPSY Affect`: useful local grief-adjacent resource, but small and not a full naturalistic benchmark

## Gaps and Opportunities

- Gap 1: No strong open benchmark directly comparing human and LLM writing specifically about death, bereavement, or existential reflection.
- Gap 2: Authenticity is usually studied either as reader perception or as authorship detection, but rarely both at once.
- Gap 3: Available grief datasets are either inaccessible, small, or not naturally matched against non-grief human controls and LLM generations.
- Gap 4: Existing detection work studies distinguishability, not whether LLM writing is analogous to human writing about unexperienced events.

## Recommendations for Our Experiment

- Recommended datasets:
  - `datasets/mage/` for the main human-vs-LLM comparison infrastructure
  - `datasets/aipsy_affect/` as a grief-language prompt and rubric seed set
- Recommended baselines:
  - MAGE Longformer baseline
  - DetectGPT zero-shot baseline
  - Human reader authenticity judgments modeled after Hwang et al.
- Recommended metrics:
  - Human authenticity ratings
  - Detector scores
  - Style divergence across lexical and discourse features
  - Agreement between readers and detectors
- Methodological considerations:
  - Match prompt length and genre tightly.
  - Separate authenticity of process from authenticity of text.
  - Include a human condition where writers describe death without direct bereavement experience.
  - Treat bereavement writing as ethically sensitive; avoid overclaiming from small grief datasets.
