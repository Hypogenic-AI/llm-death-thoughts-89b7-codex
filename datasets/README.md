# Downloaded Datasets

This directory contains locally downloaded datasets for the project. Large data artifacts are intentionally excluded from git by `datasets/.gitignore`.

## Dataset 1: MAGE

### Overview
- Source: `yaful/MAGE` on Hugging Face
- Local path: `datasets/mage/`
- Size: 436,606 examples total
- Format: Hugging Face dataset saved with `save_to_disk`
- Task: Human-written vs machine-generated text classification
- Splits: train (319,071), validation (56,792), test (60,743)
- Features: `text`, `label`, `src`
- License: See upstream dataset card / repository

### Why it matters here
- Supports the core comparison between human and LLM writing.
- Covers multiple writing tasks including story generation, question answering, summaries, reviews, and scientific writing.
- Useful both as a baseline detector benchmark and as a source of matched human/LLM text for style analysis.

### Download Instructions

Using Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("yaful/MAGE")
dataset.save_to_disk("datasets/mage")
```

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/mage")
```

### Sample Data
- Saved at `datasets/mage/samples/examples.json`

### Notes
- Training labels are imbalanced in the local copy: more machine-generated than human-written examples in `train`.
- `src` identifies both domain and generator, which is useful for controlled subgroup analyses.

## Dataset 2: AIPSY Affect

### Overview
- Source: `keidolabs/aipsy-affect` on Hugging Face
- Local path: `datasets/aipsy_affect/`
- Size: 480 examples total
- Format: Hugging Face dataset saved with `save_to_disk`
- Task: Emotion-conditioned text analysis with grief, terror, rage, admiration, and related affect labels
- Splits: clinical (192), neutral (192), moderate (48), complex_neutral (48)
- Features: `id`, `emotion`, `intensity`, `domain`, `domain_label`, `matched_control_id`, `word_count`, `text`
- License: See upstream dataset card

### Why it matters here
- Contains explicit `grief` examples and a small bereavement subset with `domain_label = "Bereavement (death of loved one)"`.
- Gives a death-adjacent corpus for prompt construction, qualitative comparison, and rubric design around grief writing.

### Download Instructions

Using Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("keidolabs/aipsy-affect")
dataset.save_to_disk("datasets/aipsy_affect")
```

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/aipsy_affect")
```

### Sample Data
- Clinical examples: `datasets/aipsy_affect/samples/clinical_examples.json`
- Bereavement-focused examples: `datasets/aipsy_affect/samples/bereavement_examples.json`

### Notes
- This is small and partly vignette-like rather than a large naturalistic bereavement benchmark.
- The local copy contains 10 bereavement-tagged examples across splits.

## Dataset Recommendation

For the first experiment pass:

1. Use `MAGE` as the main human-vs-LLM comparison benchmark.
2. Use `AIPSY Affect` only as a grief/death prompt and rubric resource, not as a standalone benchmark.
3. If a larger death-focused corpus is needed later, plan a secondary collection step from public memorial or support-forum data with ethics review.
