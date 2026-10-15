#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_llm_prefs.py
------------------
LLM ordering preferences for the 458 novel binomials used in the human study.

For each item the model scores the two orderings the participants chose between:

    alpha     = prompt + "word1 and word2"      (alphabetical order)
    nonalpha  = prompt + "word2 and word1"

and records

    preference = alpha_logprob - nonalpha_logprob

so preference > 0 means the model prefers the alphabetical order, matching
`alpha_answer == 1` on the human side. p(alpha) is plogis(preference).

The prompt is identical in both strings, so its contribution cancels in the
difference; scoring the whole sequence (rather than the binomial alone) is kept
for consistency with the earlier LLM-Ordering-Preferences and
hand-trained-model-prefs pipelines. Sequence log-probabilities are NOT length
normalised, also for consistency with that work.

Items come from Experiment/novel_binoms_conditions.csv with is_filler == 0, and
the two orderings are taken from its `Alpha`/`Nonalpha` columns, so the strings
scored here are character-for-character the ones shown on screen. Attention-check
fillers are excluded. There is no attested-binomial anchor set.

Base (non-instruct) models only: instruction tuning shifts sequence probabilities
in ways that would confound a scale comparison.

Output: Data/llm-prefs/{model_slug}.csv, one row per (prompt, item)
  model, model_family, params, item_id, word1, word2, Item,
  prompt_value, prompt_text, alpha_logprob, nonalpha_logprob, preference

Constraint codings are deliberately NOT duplicated here; Analysis/load_llm_data.R
joins them back from the conditions file, which stays the single source of truth.

Resume-safe: a model whose CSV already holds every (prompt, item) pair is skipped,
and a partial CSV is topped up rather than recomputed.

Usage:
  python Scripts/score_llm_prefs.py                      # every model in MODEL_CONFIGS
  python Scripts/score_llm_prefs.py --models gpt2 pythia-160m
  python Scripts/score_llm_prefs.py --dry-run            # list the work, touch nothing
  python Scripts/score_llm_prefs.py --verify             # check repo ids resolve on HF
  python Scripts/score_llm_prefs.py --load-4bit          # NF4 for models too big for VRAM
  python Scripts/score_llm_prefs.py --workers-per-gpu 1  # single desktop GPU
"""

import os
import argparse
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The conditions file lives in Experiment/, which is its own repo synced to
# Pavlovia and therefore not part of this one. Data/ holds a mirror so a fresh
# clone (the H100 box, say) can score without the experiment checked out.
# Experiment/ wins when present, since that is the copy the participants saw.
_ITEMS_CANDIDATES = [
    PROJECT_ROOT / "Experiment" / "novel_binoms_conditions.csv",
    PROJECT_ROOT / "Data" / "novel_binoms_conditions.csv",
]
ITEMS_CSV = str(next((p for p in _ITEMS_CANDIDATES if p.exists()),
                     _ITEMS_CANDIDATES[0]))
OUT_DIR   = str(PROJECT_ROOT / "Data" / "llm-prefs")

# Where model weights are downloaded. C: is nearly full on the Windows box, so
# the default points at D:; anywhere else (the H100 server) needs an override.
HF_CACHE_DIR = os.environ.get(
    "BINOM_HF_CACHE",
    "D:/hf_model_cache" if os.name == "nt" else os.path.expanduser("~/hf_model_cache"),
)

# torch.compile is a loss here: the sequences are short and every model is loaded
# once, so compilation costs more than it saves. Off by default, unlike the
# checkpoint-sweep scripts where one model is reused across many checkpoints.
USE_TORCH_COMPILE = False
COMPILE_MODE      = "reduce-overhead"

WORKERS_PER_GPU = 1

# ── Prompts ──────────────────────────────────────────────────────────────────
# The 51 prefixes from hand-trained-model-prefs/model-prefs-all-ckpts.py, copied
# verbatim. (ALL_MODELS_ALL_PREFIXES_NOVEL_AND_ATTESTED.csv shows 50 distinct
# values because prompt 0 is a bare space, which does not survive a CSV round
# trip. Rows here are keyed on the integer `prompt_value`, not the text, so
# resuming and grouping are unaffected.)
# Prompt 0 is the closest match to how the binomials were shown to people; the
# rest are there because preference is prefix-sensitive, so every estimate is
# averaged over contexts rather than resting on one frame.
LIST_OF_PROMPTS = [
    " ",
    "Well, ",
    "So, ",
    "Then ",
    "Possibly ",
    "Or even ",
    "Maybe a ",
    "Perhaps a ",
    "At times ",
    "Suddenly, the ",
    "Honestly just ",
    "Especially the ",
    "For instance ",
    "In some cases ",
    "Every now and then ",
    "Occasionally you’ll find ",
    "There can be examples like ",
    "You might notice things like ",
    "People sometimes mention ",
    "Sometimes just ",
    "Nothing specific comes to mind except the ",
    "It reminded me loosely of the ",
    "There was a vague reference to the ",
    "Unexpectedly the ",
    "It’s easy to overlook the ",
    "There used to be talk of ",
    "Out in the distance was the ",
    "What puzzled everyone was the ",
    "At some point I overheard ",
    "Without warning came ",
    "A friend once described the ",
    "The scene shifted toward ",
    "Nobody expected to hear about ",
    "Things eventually turned toward ",
    "The conversation eventually returned to ",
    "I only remember a hint of the ",
    "I couldn’t quite place the ",
    "It somehow led back to the ",
    "What stood out most was the ",
    "The oddest part involved the ",
    "Later on, people were discussing ",
    "There was this fleeting idea about ",
    "I once heard someone bring up the ",
    "There was a moment involving the ",
    "It all started when we noticed the ",
    "Another example floated around concerning the ",
    "I came across something about the ",
    "A situation arose involving the ",
    "The conversation drifted toward the ",
    "At one point we ended up discussing ",
    "Out of nowhere came a mention of the ",
]

# =============================================================================
# MODELS
# =============================================================================
#
# All base/pretrained checkpoints, final revision only.
#
# `params`      parameter count in millions, used for the scale axis in the analysis
# `gated`       repo needs a HF token with an accepted licence
# `large_model` too big for 8 GB in bf16; loaded with device_map="auto" so it
#               spills to CPU, or with --load-4bit
#
# Repo ids marked VERIFIED are ones already run in this lab's other pipelines.
# The rest are from the public model cards; run --verify before a long job so a
# typo surfaces in seconds rather than after the first 20 models.

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ── GPT-2 (OpenAI, WebText) ──────────────────────────────────────────────
    "gpt2":        {"family": "gpt2", "params":  124},   # VERIFIED
    "gpt2-medium": {"family": "gpt2", "params":  355},
    "gpt2-large":  {"family": "gpt2", "params":  774},
    "gpt2-xl":     {"family": "gpt2", "params": 1558},   # VERIFIED

    # ── Pythia (EleutherAI, the Pile) ────────────────────────────────────────
    # The scale ladder: one architecture, one corpus, one data order.
    "EleutherAI/pythia-70m":   {"family": "pythia", "params":   70},
    "EleutherAI/pythia-160m":  {"family": "pythia", "params":  160},  # VERIFIED
    "EleutherAI/pythia-410m":  {"family": "pythia", "params":  410},  # VERIFIED
    "EleutherAI/pythia-1b":    {"family": "pythia", "params": 1000},  # VERIFIED
    "EleutherAI/pythia-1.4b":  {"family": "pythia", "params": 1400},  # VERIFIED
    "EleutherAI/pythia-2.8b":  {"family": "pythia", "params": 2800},  # VERIFIED
    "EleutherAI/pythia-6.9b":  {"family": "pythia", "params": 6900, "large_model": True},  # VERIFIED

    # ── OPT (Meta) ───────────────────────────────────────────────────────────
    # Same architecture as the hand-trained BabyLM models below, so the
    # pretrained-vs-hand-trained comparison holds architecture constant.
    "facebook/opt-125m": {"family": "opt", "params":  125},
    "facebook/opt-350m": {"family": "opt", "params":  350},
    "facebook/opt-1.3b": {"family": "opt", "params": 1300},
    "facebook/opt-2.7b": {"family": "opt", "params": 2700},

    # ── OLMo (AI2, open data) ────────────────────────────────────────────────
    "allenai/OLMo-2-0425-1B":  {"family": "olmo", "params": 1000},  # VERIFIED
    "allenai/OLMo-7B-0424-hf": {"family": "olmo", "params": 7000, "large_model": True},  # VERIFIED
    "allenai/Olmo-3-1025-7B":  {"family": "olmo", "params": 7000, "large_model": True},  # VERIFIED

    # ── Hand-trained BabyLM models (znhoughton) ──────────────────────────────
    # The 20-epoch seed-964 runs, which is what binom-prefs-novel-ablation scored.
    # model-prefs-all-ckpts.py names a 64eps set instead, but those repos do not
    # exist on the Hub (checked with --verify); only these three do.
    "znhoughton/opt-babylm-125m-20eps-seed964": {"family": "babylm", "params":  125},
    "znhoughton/opt-babylm-350m-20eps-seed964": {"family": "babylm", "params":  350},
    "znhoughton/opt-babylm-1.3b-20eps-seed964": {"family": "babylm", "params": 1300},

    # ── Modern families at 2 to 8B ───────────────────────────────────────────
    "Qwen/Qwen3-1.7B-Base":      {"family": "qwen3",   "params": 1700},
    "Qwen/Qwen3-4B-Base":        {"family": "qwen3",   "params": 4000},
    "Qwen/Qwen3-8B-Base":        {"family": "qwen3",   "params": 8000, "large_model": True},
    # `gated` reflects what the Hub actually enforces, checked anonymously by
    # HEADing a weight file rather than by resolving the repo id: a gated repo's
    # metadata resolves for anyone, so model_info() alone says nothing about
    # whether the weights will download. Mistral-7B-v0.3 is ungated now whatever
    # it was when this list was first written.
    "meta-llama/Llama-3.2-3B":   {"family": "llama3",  "params": 3000, "gated": True},
    "google/gemma-3-4b-pt":      {"family": "gemma3",  "params": 4000, "gated": True},
    "mistralai/Mistral-7B-v0.3": {"family": "mistral", "params": 7000,
                                  "large_model": True},

    # ── Off by default, uncomment to add ─────────────────────────────────────
    # A dense 1-epoch BabyLM run at 125m, useful as an epochs-held-constant
    # contrast against the 20-epoch model of the same size.
    # "znhoughton/opt-babylm-125m-1ep-dense-seed964": {"family": "babylm-1ep", "params": 125},
    #
    # The matched C4 controls would isolate the training corpus, but
    # znhoughton/opt-c4-* is not on the Hub (model-prefs-all-ckpts.py refers to
    # repos that were never pushed, or were pushed under another account). Add
    # the ids here once they exist.

    # Beyond one 8 GB card even at 4-bit; add these on the H100 box.
    # "allenai/Olmo-3-1125-32B": {"family": "olmo",  "params": 32000, "large_model": True},
    # "Qwen/Qwen3-32B-Base":     {"family": "qwen3", "params": 32000, "large_model": True},
}


def model_slug(model_name: str) -> str:
    """Filename-safe name: the repo's last path segment."""
    return model_name.split("/")[-1]


# =============================================================================
# DATA LOADING
# =============================================================================

_ITEMS_DF: Optional[pd.DataFrame] = None


def load_items() -> pd.DataFrame:
    """
    The 458 novel binomials, exactly as presented: `Alpha` and `Nonalpha` are the
    strings from the conditions file rather than rebuilt from word1/word2, so the
    scored text cannot drift from the experiment. The 10 attention-check fillers
    are dropped.
    """
    df = pd.read_csv(ITEMS_CSV)
    df = df[df["is_filler"] == 0].copy()

    # Guard against a conditions file where Alpha stops matching its own parts.
    rebuilt_alpha    = df["word1"] + " and " + df["word2"]
    rebuilt_nonalpha = df["word2"] + " and " + df["word1"]
    bad = df[(df["Alpha"] != rebuilt_alpha) | (df["Nonalpha"] != rebuilt_nonalpha)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} item(s) where Alpha/Nonalpha disagree with word1/word2, "
            f"first: {bad.iloc[0]['item_id']}"
        )

    keep = ["item_id", "word1", "word2", "Alpha", "Nonalpha"]
    return df[keep].reset_index(drop=True)


# =============================================================================
# RESUME SAFETY
# =============================================================================

def missing_prompts(out_path: str, items_df: pd.DataFrame) -> Optional[List[int]]:
    """
    Prompt indices that still need running for an existing output file.

    Returns [] when the file is complete, every index when it is unreadable, and
    the incomplete ones otherwise. A prompt counts as done only when all items
    are present, so a run killed mid-prompt redoes that prompt and nothing else.
    """
    expected_items = set(items_df["item_id"])
    try:
        df = pd.read_csv(out_path)
    except Exception as e:
        print(f"    could not read {out_path} ({e}); rerunning in full")
        return list(range(len(LIST_OF_PROMPTS)))

    if not {"prompt_value", "item_id"}.issubset(df.columns):
        print(f"    {out_path} has unexpected columns; rerunning in full")
        return list(range(len(LIST_OF_PROMPTS)))

    todo = []
    for idx in range(len(LIST_OF_PROMPTS)):
        seen = set(df.loc[df["prompt_value"] == idx, "item_id"])
        if expected_items - seen:
            todo.append(idx)
    return todo


def atomic_write_csv(df: pd.DataFrame, out_path: str) -> None:
    tmp = out_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


# =============================================================================
# SCORING
# =============================================================================

@torch.inference_mode()
def to_tokens_and_logprobs(
    model,
    tokenizer,
    input_texts: List[str],
    batch_size: int = 512,
    desc: Optional[str] = None,
) -> List[float]:
    """Total sequence log-probability for each text. Halves the batch on OOM."""
    all_logprobs: List[float] = []
    n = len(input_texts)

    i = 0
    pbar = tqdm(total=n, desc=desc, leave=False, unit="seq")
    while i < n:
        batch = input_texts[i:i + batch_size]
        try:
            enc = tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
            first_device   = next(model.parameters()).device
            input_ids      = enc["input_ids"].to(first_device)
            attention_mask = enc["attention_mask"].to(first_device)

            outputs  = model(input_ids, attention_mask=attention_mask)
            logprobs = torch.log_softmax(outputs.logits.float(), dim=-1)

            # Token t is predicted from position t-1, so drop the last position
            # and the first target. Padding is masked out of the sum.
            logprobs    = logprobs[:, :-1, :]
            target_ids  = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:]

            token_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            token_logprobs = token_logprobs * target_mask

            all_logprobs.extend(token_logprobs.sum(dim=-1).tolist())
            i += batch_size
            pbar.update(len(batch))

        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            pbar.set_description(f"{desc} (OOM, bs={batch_size})")
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if batch_size == 1:
                raise RuntimeError("OOM at batch_size=1") from e
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            pbar.set_description(f"{desc} (OOM, bs={batch_size})")

    pbar.close()
    return all_logprobs


# Peak activation memory is the float32 log_softmax over [batch, tokens, vocab],
# which dwarfs the weights for a small model: gpt2 at batch 1024 wanted 7.7 GB of
# an 8 GB card. Budgeting that tensor directly keeps a 151k-vocab model (Qwen3)
# from starting every batch in the OOM-halving loop.
LOGPROB_BUDGET_BYTES = 1.5e9
MAX_SEQ_TOKENS       = 32  # longest prompt + binomial, with headroom


def pick_start_batch_size(params_m: int, vocab_size: int) -> int:
    """Sequences here are ~10 to 20 tokens, so batches can be generous."""
    if params_m >= 6000:
        ceiling = 64
    elif params_m >= 2500:
        ceiling = 256
    elif params_m >= 1000:
        ceiling = 512
    else:
        ceiling = 1024

    by_vocab = int(LOGPROB_BUDGET_BYTES // (MAX_SEQ_TOKENS * vocab_size * 4))
    return max(1, min(ceiling, by_vocab))


def score_prompt(
    prompt_idx: int,
    prompt: str,
    model_name: str,
    config: Dict[str, Any],
    tokenizer,
    model,
) -> pd.DataFrame:
    df = _ITEMS_DF

    alpha_texts    = (prompt + df["Alpha"]).tolist()
    nonalpha_texts = (prompt + df["Nonalpha"]).tolist()

    # Both orderings in one pass: same padding bucket, one kernel launch sequence.
    scores = to_tokens_and_logprobs(
        model=model,
        tokenizer=tokenizer,
        input_texts=alpha_texts + nonalpha_texts,
        batch_size=pick_start_batch_size(config["params"],
                                         getattr(model.config, "vocab_size", 50257)),
        desc=f"      prompt {prompt_idx}",
    )

    n = len(df)
    alpha_scores    = scores[:n]
    nonalpha_scores = scores[n:]

    # .to_numpy() throughout: the score lists align positionally, and mixing
    # indexed Series with bare lists in a DataFrame constructor would align the
    # Series on their index instead.
    return pd.DataFrame({
        "model":            model_name,
        "model_family":     config["family"],
        "params":           config["params"],
        "item_id":          df["item_id"].to_numpy(),
        "word1":            df["word1"].to_numpy(),
        "word2":            df["word2"].to_numpy(),
        "Item":             df["Alpha"].to_numpy(),
        "prompt_value":     prompt_idx,
        "prompt_text":      prompt,
        "alpha_logprob":    alpha_scores,
        "nonalpha_logprob": nonalpha_scores,
        "preference":       np.array(alpha_scores) - np.array(nonalpha_scores),
    })


# =============================================================================
# WORKER
# =============================================================================

@dataclass
class WorkItem:
    model_name: str
    config: Dict[str, Any]
    load_4bit: bool = False


def load_model(item: WorkItem, device: str, cache_dir: str):
    load_kwargs: Dict[str, Any] = dict(low_cpu_mem_usage=True, cache_dir=cache_dir)

    if item.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = "auto"
    elif device == "cpu":
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"]  = "cpu"
    elif item.config.get("large_model", False):
        # Spills to CPU rather than failing outright. Slow but it finishes.
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"]  = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"]  = device

    return AutoModelForCausalLM.from_pretrained(item.model_name, **load_kwargs).eval()


def run_work_item(item: WorkItem, device: str, out_dir: str) -> None:
    global _ITEMS_DF
    if _ITEMS_DF is None:
        _ITEMS_DF = load_items()

    model_name = item.model_name
    out_path   = os.path.join(out_dir, f"{model_slug(model_name)}.csv")

    existing_df = None
    if os.path.exists(out_path):
        todo = missing_prompts(out_path, _ITEMS_DF)
        if not todo:
            print(f"  {model_name}: complete, skipping")
            return
        if len(todo) < len(LIST_OF_PROMPTS):
            existing_df = pd.read_csv(out_path)
            existing_df = existing_df[~existing_df["prompt_value"].isin(todo)]
            print(f"  {model_name}: resuming, {len(todo)} prompt(s) left")
    else:
        todo = list(range(len(LIST_OF_PROMPTS)))

    tokenizer = AutoTokenizer.from_pretrained(item.config.get("tokenizer", model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding keeps the real tokens at the front, where the mask above
    # expects them; left padding would shift the target alignment.
    tokenizer.padding_side = "right"

    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    tmp_cache = tempfile.mkdtemp(prefix="hf_", dir=HF_CACHE_DIR)
    model     = None
    try:
        print(f"  {model_name}: loading ({item.config['params']}M params)")
        model = load_model(item, device, tmp_cache)

        if USE_TORCH_COMPILE:
            try:
                model = torch.compile(model, mode=COMPILE_MODE)
            except Exception as e:
                print(f"    torch.compile failed, continuing uncompiled: {e}")

        dfs = [
            score_prompt(idx, LIST_OF_PROMPTS[idx], model_name, item.config,
                         tokenizer, model)
            for idx in tqdm(todo, desc=f"  {model_slug(model_name)}", unit="prompt")
        ]

        new_df   = pd.concat(dfs, ignore_index=True)
        final_df = (pd.concat([existing_df, new_df], ignore_index=True)
                    if existing_df is not None else new_df)
        final_df = final_df.sort_values(["prompt_value", "item_id"]).reset_index(drop=True)

        atomic_write_csv(final_df, out_path)
        print(f"  {model_name}: wrote {len(final_df)} rows -> {out_path}")

    finally:
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_cache, ignore_errors=True)


def worker_main(rank: int, gpu_id: int, items: List[WorkItem], out_dir: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    os.makedirs(out_dir, exist_ok=True)
    failures = []

    print(f"\n=== worker {rank} on GPU {gpu_id}: {len(items)} model(s) ===")
    for item in items:
        try:
            run_work_item(item, device="cuda:0", out_dir=out_dir)
        except Exception as e:
            failures.append((item.model_name, str(e)))
            print(f"  FAILED {item.model_name}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    if failures:
        print(f"\nworker {rank}: {len(failures)} failure(s)")
        for m, err in failures:
            print(f"  - {m}: {err}")


# =============================================================================
# DRIVER
# =============================================================================

def detect_num_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip():
        return torch.cuda.device_count()
    return len([v for v in visible.split(",") if v.strip()])


def select_models(requested: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """--models matches on the full repo id or on its last segment."""
    if not requested:
        return MODEL_CONFIGS

    chosen, unknown = {}, []
    for want in requested:
        hits = [m for m in MODEL_CONFIGS if m == want or model_slug(m) == want]
        if hits:
            chosen.update({m: MODEL_CONFIGS[m] for m in hits})
        else:
            unknown.append(want)
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}")
    return chosen


def verify_models(configs: Dict[str, Dict[str, Any]]) -> None:
    """Resolve every repo id against the Hub without downloading weights."""
    api = HfApi()
    bad = []
    for name, cfg in configs.items():
        try:
            api.model_info(name)
            print(f"  ok      {name}")
        except Exception as e:
            note = " (gated: needs an accepted licence on your HF account)" if cfg.get("gated") else ""
            print(f"  FAILED  {name}: {type(e).__name__}{note}")
            bad.append(name)
    print(f"\n{len(configs) - len(bad)}/{len(configs)} resolved")
    if bad:
        print("unresolved: " + ", ".join(bad))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of MODEL_CONFIGS (repo id or last segment).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without loading anything.")
    parser.add_argument("--verify", action="store_true",
                        help="Check that every repo id resolves on the Hub, then exit.")
    parser.add_argument("--load-4bit", action="store_true",
                        help="NF4 quantisation. Lets 7B+ fit in 8 GB, at the cost of "
                             "quantisation noise in the log-probabilities.")
    parser.add_argument("--out-dir", default=None,
                        help="Write CSVs here instead of Data/llm-prefs (used by the smoke test).")
    parser.add_argument("--workers-per-gpu", type=int, default=WORKERS_PER_GPU)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (very slow).")
    args = parser.parse_args()

    configs = select_models(args.models)
    out_dir = args.out_dir or OUT_DIR

    if args.verify:
        print(f"Verifying {len(configs)} repo id(s) on the Hub\n")
        verify_models(configs)
        return

    items_df = load_items()
    n_seq = len(items_df) * 2 * len(LIST_OF_PROMPTS)
    print(f"{len(items_df)} novel binomials x {len(LIST_OF_PROMPTS)} prompts x 2 orders "
          f"= {n_seq:,} sequences per model")
    print(f"{len(configs)} model(s) -> {out_dir}/\n")

    work = [WorkItem(name, cfg, load_4bit=args.load_4bit) for name, cfg in configs.items()]
    # Smallest first: the cheap models land early, so stopping a long run partway
    # still leaves a usable set rather than three half-downloaded 7B models.
    work.sort(key=lambda w: w.config["params"])

    if args.dry_run:
        for w in work:
            out_path = os.path.join(out_dir, f"{model_slug(w.model_name)}.csv")
            if os.path.exists(out_path):
                todo = missing_prompts(out_path, items_df)
                state = "complete" if not todo else f"{len(todo)} prompt(s) left"
            else:
                state = "not started"
            flags = []
            if w.config.get("gated"):      flags.append("gated")
            if w.config.get("large_model"): flags.append("large")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {w.model_name:<45} {w.config['params']:>6}M  {state}{suffix}")
        return

    os.makedirs(out_dir, exist_ok=True)

    num_gpus = 0 if args.cpu else detect_num_gpus()
    if num_gpus == 0:
        print("No GPU in use, running on CPU")
        for w in work:
            try:
                run_work_item(w, device="cpu", out_dir=out_dir)
            except Exception as e:
                print(f"  FAILED {w.model_name}: {e}")
                traceback.print_exc()
    elif num_gpus == 1 and args.workers_per_gpu == 1:
        # One GPU, one worker: stay in-process so errors and progress bars are
        # not buried in a child process.
        print("1 GPU, 1 worker")
        for w in work:
            try:
                run_work_item(w, device="cuda:0", out_dir=out_dir)
            except Exception as e:
                print(f"  FAILED {w.model_name}: {e}")
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache()
    else:
        num_workers = num_gpus * args.workers_per_gpu
        print(f"Sharding {len(work)} model(s) across {num_workers} worker(s) "
              f"on {num_gpus} GPU(s)")
        shards = [work[i::num_workers] for i in range(num_workers)]
        ctx    = mp.get_context("spawn")
        procs  = []
        for rank in range(num_workers):
            if not shards[rank]:
                continue
            p = ctx.Process(target=worker_main, args=(rank, rank % num_gpus, shards[rank], out_dir))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print("\nDONE")


if __name__ == "__main__":
    main()
