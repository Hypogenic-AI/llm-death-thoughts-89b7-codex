from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
import seaborn as sns
import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from openai import OpenAI
from sklearn.metrics import roc_auc_score
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import LongformerConfig, LongformerForSequenceClassification, LongformerTokenizerFast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"
PROMPTS_DIR = ROOT / "prompts"


OBJECT_WORDS = {
    "bed",
    "blanket",
    "book",
    "bowl",
    "box",
    "bread",
    "bus",
    "calendar",
    "chair",
    "clock",
    "closet",
    "coat",
    "coffee",
    "counter",
    "cup",
    "desk",
    "door",
    "drawer",
    "dress",
    "elevator",
    "fridge",
    "glass",
    "grocery",
    "hallway",
    "key",
    "kitchen",
    "label",
    "lamp",
    "letter",
    "list",
    "monitor",
    "note",
    "pocket",
    "receipt",
    "room",
    "school",
    "sheets",
    "shirt",
    "shop",
    "stairs",
    "table",
    "television",
    "ticket",
    "towel",
    "wallet",
    "watch",
    "water",
    "window",
}

FIRST_PERSON = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}


@dataclass
class Config:
    seed: int = 42
    openai_generation_model: str = "gpt-5"
    openai_judge_model: str = "gpt-5-mini"
    openai_brief_model: str = "gpt-5-mini"
    openrouter_generation_model: str = "anthropic/claude-sonnet-4.5"
    openrouter_judge_model: str = "anthropic/claude-haiku-4.5"
    max_human_texts: int = 12
    generation_temperature: float = 0.9
    judge_temperature: float = 0.1
    brief_temperature: float = 0.2
    detector_model: str = "yaful/MAGE"
    generation_word_target_tolerance: int = 18
    gpu_batch_size: int = 32
    cpu_batch_size: int = 8


def ensure_dirs() -> None:
    for directory in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, PROMPTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(match.group(0))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def sentence_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece for piece in pieces if piece]


def text_features(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    sentences = sentence_split(text)
    word_count = len(tokens)
    sentence_count = max(1, len(sentences))
    unique_count = len(set(tokens))
    first_person_count = sum(token in FIRST_PERSON for token in tokens)
    object_count = sum(token in OBJECT_WORDS for token in tokens)
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": word_count / sentence_count,
        "type_token_ratio": unique_count / max(1, word_count),
        "first_person_rate": first_person_count / max(1, word_count),
        "object_word_rate": object_count / max(1, word_count),
    }


def bootstrap_mean_diff(
    group_a: np.ndarray, group_b: np.ndarray, *, n_boot: int = 5000, seed: int = 42
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sample_a = rng.choice(group_a, size=len(group_a), replace=True)
        sample_b = rng.choice(group_b, size=len(group_b), replace=True)
        diffs.append(float(sample_a.mean() - sample_b.mean()))
    diffs_arr = np.asarray(diffs)
    return {
        "mean_diff": float(group_a.mean() - group_b.mean()),
        "ci_low": float(np.quantile(diffs_arr, 0.025)),
        "ci_high": float(np.quantile(diffs_arr, 0.975)),
    }


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled = math.sqrt(((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b) / (len(group_a) + len(group_b) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled)


def provider_model_name(provider: str, config: Config, task: str) -> str:
    if provider == "openai" and task == "generation":
        return config.openai_generation_model
    if provider == "openai" and task == "judge":
        return config.openai_judge_model
    if provider == "openai" and task == "brief":
        return config.openai_brief_model
    if provider == "openrouter" and task == "generation":
        return config.openrouter_generation_model
    if provider == "openrouter" and task == "judge":
        return config.openrouter_judge_model
    raise ValueError(f"Unsupported provider/task combination: {provider}/{task}")


class APIClients:
    def __init__(self) -> None:
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.openrouter_key = os.environ["OPENROUTER_KEY"]

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
    def call_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        response = self.openai.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text.strip()

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
    def call_openrouter(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()

    def complete(
        self,
        *,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        if provider == "openai":
            return self.call_openai(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        if provider == "openrouter":
            return self.call_openrouter(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        raise ValueError(f"Unknown provider: {provider}")


def load_human_texts(config: Config) -> pd.DataFrame:
    dataset = load_from_disk(str(ROOT / "datasets" / "aipsy_affect"))
    allowed = {
        "Bereavement (death of loved one)": "bereavement",
        "Career destruction (life's work lost)": "other_grief",
        "Relationship dissolution (abandonment)": "other_grief",
    }
    rows: list[dict[str, Any]] = []
    for split in dataset:
        for example in dataset[split]:
            domain_label = example["domain_label"]
            if example["emotion"] != "grief" or domain_label not in allowed:
                continue
            rows.append(
                {
                    "text_id": example["id"],
                    "split": split,
                    "emotion": example["emotion"],
                    "domain_label": domain_label,
                    "domain_group": allowed[domain_label],
                    "word_count_original": example["word_count"],
                    "text": example["text"].strip(),
                    "source_type": "human",
                    "provider": "human",
                    "model_name": "human",
                }
            )
    frame = pd.DataFrame(rows).sort_values(["domain_label", "split", "text_id"]).reset_index(drop=True)
    quotas = {
        "Bereavement (death of loved one)": 6,
        "Career destruction (life's work lost)": 3,
        "Relationship dissolution (abandonment)": 3,
    }
    sampled = []
    for domain_label, quota in quotas.items():
        domain_frame = frame[frame["domain_label"] == domain_label]
        sampled.append(domain_frame.head(quota))
    frame = pd.concat(sampled, ignore_index=True)
    if len(frame) != config.max_human_texts:
        raise ValueError(f"Expected {config.max_human_texts} human texts, found {len(frame)}")
    frame["scenario_id"] = [f"scenario_{idx:02d}" for idx in range(1, len(frame) + 1)]
    return frame


def build_brief_prompt(text: str, word_count: int) -> str:
    return f"""Read the following grief-related passage and convert it into a compact scenario brief for a new writer.

Constraints:
- Do not copy phrases longer than 4 words from the original passage.
- Preserve the core situation, emotional register, and level of concreteness.
- Describe perspective, setting, concrete objects, and emotional stance.
- Include a target word count close to {word_count}.
- Output valid JSON only with keys:
  scenario_summary, perspective, concrete_details, emotional_tone, target_word_count

PASSAGE:
\"\"\"{text}\"\"\""""


def generate_scenario_briefs(human_df: pd.DataFrame, clients: APIClients, config: Config) -> pd.DataFrame:
    output_path = RESULTS_DIR / "scenario_briefs.jsonl"
    system_prompt = (
        "You extract neutral scenario briefs for literary research. Return JSON only."
    )
    rows = []
    completed_ids = set()
    if output_path.exists():
        cached = pd.read_json(output_path, lines=True)
        rows = cached.to_dict(orient="records")
        completed_ids = set(cached["scenario_id"].tolist())
    for _, row in human_df.iterrows():
        if row["scenario_id"] in completed_ids:
            continue
        prompt = build_brief_prompt(row["text"], int(row["word_count_original"]))
        raw = clients.complete(
            provider="openai",
            model=config.openai_brief_model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=config.brief_temperature,
        )
        payload = parse_json_object(raw)
        record = {
            "scenario_id": row["scenario_id"],
            "text_id": row["text_id"],
            "domain_label": row["domain_label"],
            "domain_group": row["domain_group"],
            "brief_model": config.openai_brief_model,
            "brief_raw": raw,
            **payload,
        }
        rows.append(record)
        append_jsonl(output_path, record)
        print(f"[brief] completed {row['scenario_id']}", flush=True)
        time.sleep(0.2)
    return pd.DataFrame(rows)


def build_generation_prompt(brief_row: pd.Series, *, avoid_first_person: bool) -> str:
    perspective = brief_row["perspective"]
    if avoid_first_person:
        perspective_instruction = "Prefer close third person unless the brief strongly demands first person."
    else:
        perspective_instruction = f"Use the perspective suggested by the brief: {perspective}."
    return f"""Write a short grief-related passage for research evaluation.

Requirements:
- Length target: about {int(brief_row['target_word_count'])} words, within +/- {Config().generation_word_target_tolerance} words.
- Keep the scene concrete and emotionally plausible.
- Avoid melodrama, moralizing, or summarizing the lesson.
- Do not mention being an AI or the writing task.
- Do not copy wording from any source text you may have seen.
- {perspective_instruction}

Scenario summary: {brief_row['scenario_summary']}
Concrete details to preserve: {brief_row['concrete_details']}
Emotional tone: {brief_row['emotional_tone']}
"""


def generate_model_texts(brief_df: pd.DataFrame, clients: APIClients, config: Config) -> pd.DataFrame:
    output_path = RESULTS_DIR / "generations.jsonl"
    rows = []
    completed = set()
    if output_path.exists():
        cached = pd.read_json(output_path, lines=True)
        rows = cached.to_dict(orient="records")
        completed = set(zip(cached["scenario_id"], cached["provider"]))
    generation_specs = [
        ("openai", config.openai_generation_model),
        ("openrouter", config.openrouter_generation_model),
    ]
    system_prompt = (
        "You are a careful literary writer producing concise emotionally realistic prose for evaluation."
    )
    for _, brief_row in brief_df.iterrows():
        for provider, model_name in generation_specs:
            if (brief_row["scenario_id"], provider) in completed:
                continue
            raw = clients.complete(
                provider=provider,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=build_generation_prompt(brief_row, avoid_first_person=True),
                temperature=config.generation_temperature,
            )
            record = {
                "scenario_id": brief_row["scenario_id"],
                "text_id": f"{brief_row['scenario_id']}_{provider}",
                "domain_label": brief_row["domain_label"],
                "domain_group": brief_row["domain_group"],
                "source_type": "model",
                "provider": provider,
                "model_name": model_name,
                "text": raw.strip(),
            }
            rows.append(record)
            append_jsonl(output_path, record)
            print(f"[generation] completed {brief_row['scenario_id']} via {provider}", flush=True)
            time.sleep(0.3)
    return pd.DataFrame(rows)


def build_judge_prompt(text: str) -> str:
    return f"""Evaluate the following passage as writing, not as a moral statement.

Rate each dimension from 1 to 5:
- authenticity: does it feel like a sincere and credible piece of human-oriented writing?
- emotional_plausibility: are the emotions and reactions believable?
- specificity: does it use concrete, situational detail rather than generic abstraction?

Also guess whether the passage is more likely human-written or machine-generated.

Output valid JSON only with keys:
authenticity, emotional_plausibility, specificity, likely_source, rationale

PASSAGE:
\"\"\"{text}\"\"\""""


def judge_texts(text_df: pd.DataFrame, clients: APIClients, config: Config) -> pd.DataFrame:
    output_path = RESULTS_DIR / "judgments.jsonl"
    rows = []
    completed = set()
    if output_path.exists():
        cached = pd.read_json(output_path, lines=True)
        rows = cached.to_dict(orient="records")
        completed = set(zip(cached["text_id"], cached["judge_provider"]))
    judge_specs = [
        ("openai", config.openai_judge_model),
        ("openrouter", config.openrouter_judge_model),
    ]
    system_prompt = (
        "You are a strict but fair literary evaluator. Return JSON only. Keep rationale under 25 words."
    )
    shuffled = text_df.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    for _, text_row in shuffled.iterrows():
        for provider, model_name in judge_specs:
            if (text_row["text_id"], provider) in completed:
                continue
            raw = clients.complete(
                provider=provider,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=build_judge_prompt(text_row["text"]),
                temperature=config.judge_temperature,
            )
            payload = parse_json_object(raw)
            record = {
                "text_id": text_row["text_id"],
                "judge_provider": provider,
                "judge_model": model_name,
                "judge_raw": raw,
                "authenticity": payload["authenticity"],
                "emotional_plausibility": payload["emotional_plausibility"],
                "specificity": payload["specificity"],
                "likely_source": str(payload["likely_source"]).strip().lower(),
                "rationale": payload["rationale"],
            }
            rows.append(record)
            append_jsonl(output_path, record)
            print(f"[judge] completed {text_row['text_id']} via {provider}", flush=True)
            time.sleep(0.3)
    return pd.DataFrame(rows)


def load_detector(device: str) -> tuple[Any, Any]:
    config_path = hf_hub_download("yaful/MAGE", "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    config_payload["id2label"] = {"0": "machine", "1": "human"}
    config_payload["label2id"] = {"machine": 0, "human": 1}
    config = LongformerConfig.from_dict(config_payload)
    tokenizer = LongformerTokenizerFast.from_pretrained("yaful/MAGE")
    model = LongformerForSequenceClassification.from_pretrained("yaful/MAGE", config=config)
    model.to(device)
    model.eval()
    return tokenizer, model


def score_with_detector(text_df: pd.DataFrame, config: Config, device: str) -> pd.DataFrame:
    print("[detector] loading model", flush=True)
    tokenizer, model = load_detector(device)
    texts = text_df["text"].tolist()
    batch_size = config.gpu_batch_size if device.startswith("cuda") else config.cpu_batch_size
    machine_probs: list[float] = []
    human_probs: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        human_probs.extend(probs[:, 0].tolist())
        machine_probs.extend(probs[:, 1].tolist())
        print(f"[detector] scored {min(start + batch_size, len(texts))}/{len(texts)} texts", flush=True)
    detector_df = text_df[["text_id"]].copy()
    detector_df["mage_machine_prob"] = machine_probs
    detector_df["mage_human_prob"] = human_probs
    save_jsonl(RESULTS_DIR / "detector_scores.jsonl", detector_df.to_dict(orient="records"))
    return detector_df


def aggregate_outputs(
    human_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    judgment_df: pd.DataFrame,
    detector_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    text_df = pd.concat(
        [
            human_df[["scenario_id", "text_id", "domain_label", "domain_group", "source_type", "provider", "model_name", "text"]],
            generated_df[["scenario_id", "text_id", "domain_label", "domain_group", "source_type", "provider", "model_name", "text"]],
        ],
        ignore_index=True,
    )
    feature_rows = []
    for _, row in text_df.iterrows():
        feature_rows.append({"text_id": row["text_id"], **text_features(row["text"])})
    feature_df = pd.DataFrame(feature_rows)
    judge_agg = (
        judgment_df.groupby("text_id", as_index=False)
        .agg(
            authenticity_mean=("authenticity", "mean"),
            emotional_plausibility_mean=("emotional_plausibility", "mean"),
            specificity_mean=("specificity", "mean"),
            machine_guess_rate=("likely_source", lambda values: np.mean([1 if "machine" in v else 0 for v in values])),
        )
    )
    merged = (
        text_df.merge(judge_agg, on="text_id", how="left")
        .merge(detector_df, on="text_id", how="left")
        .merge(feature_df, on="text_id", how="left")
    )
    merged["generated_flag"] = (merged["source_type"] != "human").astype(int)
    merged.to_csv(RESULTS_DIR / "text_pool.csv", index=False)
    return text_df, merged


def run_statistics(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    primary_pvalues = []
    primary_labels = []
    for domain_group in ["bereavement", "other_grief"]:
        subset = merged[merged["domain_group"] == domain_group]
        human = subset[subset["source_type"] == "human"]["authenticity_mean"].to_numpy()
        model = subset[subset["source_type"] == "model"]["authenticity_mean"].to_numpy()
        bootstrap = bootstrap_mean_diff(human, model)
        welch = stats.ttest_ind(human, model, equal_var=False)
        mann = stats.mannwhitneyu(human, model, alternative="two-sided")
        effect = cohen_d(human, model)
        label = f"authenticity_human_vs_model_{domain_group}"
        primary_pvalues.append(float(welch.pvalue))
        primary_labels.append(label)
        tests.append(
            {
                "comparison": label,
                "metric": "authenticity_mean",
                "group_a": "human",
                "group_b": "model",
                "n_a": len(human),
                "n_b": len(model),
                "mean_a": float(np.mean(human)),
                "mean_b": float(np.mean(model)),
                "welch_t_p": float(welch.pvalue),
                "mannwhitney_p": float(mann.pvalue),
                "cohens_d": effect,
                **bootstrap,
            }
        )
    correction = multipletests(primary_pvalues, alpha=0.05, method="fdr_bh")
    for idx, label in enumerate(primary_labels):
        for row in tests:
            if row["comparison"] == label:
                row["welch_t_p_fdr_bh"] = float(correction[1][idx])
                row["reject_fdr_bh"] = bool(correction[0][idx])

    spearman = stats.spearmanr(merged["mage_machine_prob"], merged["authenticity_mean"])
    tests.append(
        {
            "comparison": "detector_vs_authenticity",
            "metric": "correlation",
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    )

    auroc = roc_auc_score(merged["generated_flag"], merged["mage_machine_prob"])
    tests.append(
        {
            "comparison": "mage_detector_auroc",
            "metric": "auroc",
            "value": float(auroc),
        }
    )

    model_df = merged.copy()
    model_df["source_binary"] = np.where(model_df["source_type"] == "human", "human", "model")
    fit = ols("authenticity_mean ~ C(source_binary) * C(domain_group)", data=model_df).fit()
    anova_like = {
        "authenticity_regression_r2": float(fit.rsquared),
        "authenticity_regression_params": {key: float(value) for key, value in fit.params.items()},
        "authenticity_regression_pvalues": {key: float(value) for key, value in fit.pvalues.items()},
    }

    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(RESULTS_DIR / "stat_tests.csv", index=False)
    save_json(RESULTS_DIR / "anova_like.json", anova_like)
    return tests_df, anova_like


def create_figures(merged: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=merged,
        x="domain_group",
        y="authenticity_mean",
        hue="source_type",
        ax=ax,
    )
    ax.set_title("Authenticity Ratings by Domain and Source")
    ax.set_xlabel("Domain Group")
    ax.set_ylabel("Mean Authenticity Rating")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "authenticity_by_domain_source.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=merged,
        x="mage_machine_prob",
        y="authenticity_mean",
        hue="source_type",
        style="domain_group",
        s=80,
        ax=ax,
    )
    ax.set_title("Detector Machine Probability vs Authenticity")
    ax.set_xlabel("MAGE machine probability")
    ax.set_ylabel("Mean authenticity rating")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "detector_vs_authenticity.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    feature_plot = merged.melt(
        id_vars=["source_type"],
        value_vars=["type_token_ratio", "first_person_rate", "object_word_rate"],
        var_name="feature",
        value_name="value",
    )
    sns.barplot(data=feature_plot, x="feature", y="value", hue="source_type", ax=ax)
    ax.set_title("Selected Style Features by Source")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "style_features_by_source.png", dpi=200)
    plt.close(fig)


def summarize_outputs(config: Config, merged: pd.DataFrame, tests_df: pd.DataFrame, environment: dict[str, Any]) -> dict[str, Any]:
    source_summary = (
        merged.groupby(["domain_group", "source_type"], as_index=False)
        .agg(
            n=("text_id", "count"),
            authenticity_mean=("authenticity_mean", "mean"),
            authenticity_std=("authenticity_mean", "std"),
            emotional_plausibility_mean=("emotional_plausibility_mean", "mean"),
            specificity_mean=("specificity_mean", "mean"),
            machine_guess_rate=("machine_guess_rate", "mean"),
            mage_machine_prob_mean=("mage_machine_prob", "mean"),
        )
    )
    source_summary.to_csv(RESULTS_DIR / "group_summary.csv", index=False)
    summary = {
        "config": asdict(config),
        "environment": environment,
        "counts": {
            "human_texts": int((merged["source_type"] == "human").sum()),
            "model_texts": int((merged["source_type"] == "model").sum()),
            "total_texts": int(len(merged)),
        },
        "group_summary": source_summary.to_dict(orient="records"),
        "tests": tests_df.to_dict(orient="records"),
    }
    save_json(RESULTS_DIR / "summary_metrics.json", summary)
    return summary


def environment_snapshot(config: Config, device: str) -> dict[str, Any]:
    gpu_info = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu_info.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    return {
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": device,
        "gpu_info": gpu_info,
        "gpu_batch_size_used": config.gpu_batch_size if device.startswith("cuda") else None,
        "cpu_batch_size_used": config.cpu_batch_size if not device.startswith("cuda") else None,
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the death-writing authenticity study.")
    parser.add_argument("--skip-api", action="store_true", help="Reuse cached API outputs if present.")
    args = parser.parse_args()

    ensure_dirs()
    config = Config()
    set_seed(config.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    environment = environment_snapshot(config, device)
    save_json(RESULTS_DIR / "config.json", {"config": asdict(config), "environment": environment})

    human_df = load_human_texts(config)
    human_df.to_csv(RESULTS_DIR / "human_texts.csv", index=False)

    clients = APIClients()

    if args.skip_api and (RESULTS_DIR / "scenario_briefs.jsonl").exists():
        brief_df = pd.read_json(RESULTS_DIR / "scenario_briefs.jsonl", lines=True)
    else:
        brief_df = generate_scenario_briefs(human_df, clients, config)

    if args.skip_api and (RESULTS_DIR / "generations.jsonl").exists():
        generated_df = pd.read_json(RESULTS_DIR / "generations.jsonl", lines=True)
    else:
        generated_df = generate_model_texts(brief_df, clients, config)

    if args.skip_api and (RESULTS_DIR / "judgments.jsonl").exists():
        judgment_df = pd.read_json(RESULTS_DIR / "judgments.jsonl", lines=True)
    else:
        pooled_df = pd.concat(
            [
                human_df[["scenario_id", "text_id", "domain_label", "domain_group", "source_type", "provider", "model_name", "text"]],
                generated_df[["scenario_id", "text_id", "domain_label", "domain_group", "source_type", "provider", "model_name", "text"]],
            ],
            ignore_index=True,
        )
        judgment_df = judge_texts(pooled_df, clients, config)

    detector_df = score_with_detector(
        pd.concat(
            [
                human_df[["scenario_id", "text_id", "domain_label", "domain_group", "source_type", "provider", "model_name", "text"]],
                generated_df[["scenario_id", "text_id", "domain_label", "domain_group", "source_type", "provider", "model_name", "text"]],
            ],
            ignore_index=True,
        ),
        config,
        device,
    )
    _, merged = aggregate_outputs(human_df, generated_df, judgment_df, detector_df)
    tests_df, _ = run_statistics(merged)
    create_figures(merged)
    summarize_outputs(config, merged, tests_df, environment)


if __name__ == "__main__":
    main()
