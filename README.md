# llm-death-thoughts-89b7-codex

Pilot AI-authorship study on whether LLM writing about death is uniquely inauthentic, or whether it behaves like human writing about experiences the author has not directly undergone. The project uses local grief texts, real frontier-model generations, blinded LLM judges, and a pretrained detector.

## Key Findings
- Human and model passages both received very high authenticity scores; the human-model gap was small in both bereavement and other-grief conditions.
- The bereavement gap (`0.17` points on a 5-point scale) was not larger than the other-grief gap (`0.13`), which weakens the claim that death is a uniquely disqualifying topic for LLM writing.
- The MAGE detector still separated generated text overall (`AUROC = 0.826`) but produced many false positives on human grief texts.
- Judge-model source guesses labeled every passage as human or human-written, highlighting a sharp divergence between authenticity judgments and detector behavior.

## Reproduce
```bash
uv venv
source .venv/bin/activate
uv sync
python -m research_workspace.run_study
```

If API outputs already exist and you only want to recompute detector scores, statistics, and figures:

```bash
source .venv/bin/activate
python -m research_workspace.run_study --skip-api
```

## File Structure
- `planning.md`: study design and motivation
- `REPORT.md`: full write-up with results and interpretation
- `src/research_workspace/run_study.py`: end-to-end pipeline
- `results/`: cached prompts, generations, judgments, pooled data, and statistics
- `figures/`: exported plots used in the report

Full details are in [REPORT.md](REPORT.md).
