# Cloned Repositories

## Repo 1: MAGE
- URL: https://github.com/yafuly/MAGE
- Purpose: Benchmark and detector resources for machine-generated text detection in the wild
- Location: `code/MAGE/`
- Key files:
  - `deployment/prepare_testbeds.py`
  - `deployment/utils.py`
  - `training/longformer/main.py`
  - `README.md`
- Notes:
  - The repository links directly to the `yaful/MAGE` dataset already downloaded into `datasets/mage/`.
  - Best use here is as baseline code and as documentation of realistic OOD evaluation settings.

## Repo 2: detect-gpt
- URL: https://github.com/eric-mitchell/detect-gpt
- Purpose: Official DetectGPT implementation for zero-shot machine-generated text detection
- Location: `code/detect-gpt/`
- Key files:
  - `run.py`
  - `custom_datasets.py`
  - `paper_scripts/main.sh`
  - `paper_scripts/cross.sh`
- Notes:
  - Useful as a baseline detector against human/LLM writing contrasts.
  - README expects external datasets such as WritingPrompts, so this is not plug-and-play with the current local datasets without adaptation.

## Repo 3: Llama-3.1-Centaur-70B
- URL: https://github.com/marcelbinz/Llama-3.1-Centaur-70B
- Purpose: Model and analysis code for using LLMs as models of human cognition
- Location: `code/Llama-3.1-Centaur-70B/`
- Key files:
  - `run_minimal.py`
  - `test.py`
  - `test_adapter.py`
  - `finetune.py`
  - `metabench/metabench.py`
- Notes:
  - The original clone hit an upstream Git LFS quota failure; the repository was recloned with `GIT_LFS_SKIP_SMUDGE=1`.
  - Source code and README are present, but some large artifacts are unavailable unless the upstream LFS budget is restored.
  - The README reports very high hardware requirements: about 80 GB GPU for the adapter route and about 160 GB GPU for full model loading.
