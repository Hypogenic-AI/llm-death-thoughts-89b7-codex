# Resources Catalog

## Summary

This document catalogs the resources gathered for the project, including papers, datasets, and code repositories relevant to authenticity in AI writing, human-like behavior modeling, human-vs-LLM text comparison, and death-related human writing.

## Papers

Total papers downloaded: 7

| Title | Authors | Year | File | Key Info |
|------|------|------|------|------|
| "It was 80% me, 20% AI": Seeking Authenticity in Co-Writing with Large Language Models | Hwang et al. | 2024 | `papers/must_read/hwang_2024_authenticity_cowriting_llms.pdf` | Mixed-method study of writer and reader authenticity perceptions |
| Using Large Language Models to Generate Authentic Multi-agent Knowledge Work Datasets | Heim et al. | 2024 | `papers/must_read/liu_2024_authentic_multiagent_knowledge_work.pdf` | Synthetic-authenticity framing via realism ratings |
| Turning Large Language Models into Cognitive Models | Binz and Schulz | 2023 | `papers/must_read/binz_2023_turning_llms_into_cognitive_models.pdf` | LLMs fitted to human behavioral experiments |
| The Real, the Better | Jiang et al. | 2024 | `papers/should_read/real_2024_aligning_llms_online_human_behaviors.pdf` | Aligns LLMs to online human behavior traces |
| On the Possibilities of AI-Generated Text Detection | Chakraborty et al. | 2023 | `papers/should_read/gehrmann_2023_possibilities_ai_generated_text_detection.pdf` | Theory and empirical limits of detection |
| MAGE: Machine-generated Text Detection in the Wild | Li et al. | 2024 | `papers/should_read/li_2024_mage_machine_generated_text_detection.pdf` | Large benchmark and detector evaluation |
| Grief-Stricken in a Crowd | Brubaker et al. | 2012 | `papers/reference/de_choudhury_2012_grief_stricken_in_a_crowd.pdf` | Bereavement language analysis in social media |

See `papers/README.md` for detailed descriptions.

## Datasets

Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|------|------|------|------|------|
| MAGE | Hugging Face `yaful/MAGE` | 436,606 examples | Human vs AI text classification | `datasets/mage/` | Main benchmark for human/LLM comparison |
| AIPSY Affect | Hugging Face `keidolabs/aipsy-affect` | 480 examples | Affect and grief text analysis | `datasets/aipsy_affect/` | Includes 10 local bereavement-tagged examples |

See `datasets/README.md` for detailed descriptions.

## Code Repositories

Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|------|------|------|------|
| MAGE | https://github.com/yafuly/MAGE | Benchmark and detector code | `code/MAGE/` | Closest fit to current evaluation needs |
| detect-gpt | https://github.com/eric-mitchell/detect-gpt | Zero-shot AI-text detection baseline | `code/detect-gpt/` | Needs adaptation to local datasets |
| Llama-3.1-Centaur-70B | https://github.com/marcelbinz/Llama-3.1-Centaur-70B | Human cognition modeling with LLMs | `code/Llama-3.1-Centaur-70B/` | Large LFS artifacts unavailable from upstream |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy

Searches were organized around three themes:

1. Authenticity and authorship in AI-assisted writing
2. LLMs as models of human behavior or cognition
3. Human-vs-LLM text detection plus bereavement-language resources

Primary retrieval used arXiv/OpenAlex/web search for papers, Hugging Face for datasets, and GitHub for code.

### Selection Criteria

- Direct relevance to the hypothesis
- Accessible full text or open PDF
- Downloadable dataset or usable repository
- Practical value for downstream experiment design

### Challenges Encountered

- The local paper-finder service did not return results in a usable time window, so manual API/web search was used instead.
- Semantic Scholar API returned rate limits from this environment.
- Several likely mental-health datasets were gated or not straightforwardly loadable.
- The Centaur repository exceeded upstream Git LFS quota for some artifacts.

### Gaps and Workarounds

- No large, open, naturalistic bereavement benchmark was accessible.
  - Workaround: use `AIPSY Affect` for grief/bereavement prompt seeds and the Brubaker paper for linguistic guidance.
- No direct benchmark exists for "authentic writing about unexperienced death."
  - Workaround: combine MAGE-style human/LLM comparisons with custom prompting and reader studies modeled on Hwang et al.

## Recommendations for Experiment Design

1. Primary dataset(s): Use `MAGE` for the main human-vs-LLM contrast, then construct a small custom matched death-writing set using prompts informed by `AIPSY Affect`.
2. Baseline methods: Start with MAGE Longformer and DetectGPT; compare them with human readers rather than relying on detectors alone.
3. Evaluation metrics: Use authenticity ratings, ownership/process ratings, detector scores, and feature-based stylistic analyses.
4. Code to adapt/reuse: Reuse `code/MAGE/` for benchmark formatting and OOD evaluation logic; adapt `code/detect-gpt/` for zero-shot scoring; use `code/Llama-3.1-Centaur-70B/` mainly as methodological inspiration rather than as a runnable baseline in this environment.
