#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_llm_prefs_checkpoints.py
------------------------------
Experiment 3: ordering preferences across training, not just at the end.

Same measure and same items as score_llm_prefs.py, scored at a series of
training checkpoints so the trajectory of a preference can be read rather than
only its final value. Both OLMo and Pythia publish checkpoints, which lets the
question be asked of a scale ladder rather than a single model.

Checkpoints are discovered from the Hub, not hardcoded, so a repo that adds
revisions is picked up without editing this file:

    step_branches   git branches named "stepN"        (Pythia)
    stage1_branches git branches named "stage1-stepN" (OLMo 3)
    step_tags       git tags named "step-N"           (the hand-trained models)

`tokens_per_step` converts a step index into tokens seen, which is the x-axis
worth plotting: a step means different amounts of data for different models,
and the comparison of interest is at matched tokens rather than matched steps.

Output: Data/llm-prefs-checkpoints/{model_slug}__{checkpoint}.csv
  the columns score_llm_prefs.py writes, plus checkpoint, step, tokens

One file per checkpoint, written only once that checkpoint has finished all
prompts, so an interrupted run resumes at checkpoint granularity.

Usage:
  python Scripts/score_llm_prefs_checkpoints.py --dry-run
  python Scripts/score_llm_prefs_checkpoints.py --models pythia-1.4b
  python Scripts/score_llm_prefs_checkpoints.py --n-checkpoints 12
"""

import argparse
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the measure, the prompts and the item loading rather than restating them.
import importlib.util

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("score_llm_prefs",
                                               _HERE / "score_llm_prefs.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

PROJECT_ROOT = _HERE.parent
OUT_DIR      = str(PROJECT_ROOT / "Data" / "llm-prefs-checkpoints")

# ── Which models expose checkpoints, and how ─────────────────────────────────
#
# tokens_per_step is seq_len x global batch, taken from each model's published
# training configuration. Pythia is uniform across sizes (2048 x 1024).
#
# Scope: the hand-trained BabyLM models, the Pythia sizes that match them
# (160m/410m/1.4b against 125m/350m/1.3b), and one OLMo. The point of the
# comparison is BabyLM against a same-size model trained on far more data, so
# Pythia sizes with no BabyLM counterpart are not needed; 2.8b and 6.9b are
# therefore left out rather than scored and discarded.
CHECKPOINT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "EleutherAI/pythia-160m": {"family": "pythia", "params": 160,
                               "source": "step_branches", "tokens_per_step": 2_097_152},
    "EleutherAI/pythia-410m": {"family": "pythia", "params": 410,
                               "source": "step_branches", "tokens_per_step": 2_097_152},
    "EleutherAI/pythia-1.4b": {"family": "pythia", "params": 1400,
                               "source": "step_branches", "tokens_per_step": 2_097_152},

    "allenai/Olmo-3-1025-7B": {"family": "olmo", "params": 7000,
                               "source": "stage1_branches", "tokens_per_step": 8_388_608,
                               "large_model": True},

    "znhoughton/opt-babylm-125m-20eps-seed964": {"family": "babylm", "params": 125,
                                                 "source": "step_tags",
                                                 "tokens_per_step": 1_638_400},
    "znhoughton/opt-babylm-350m-20eps-seed964": {"family": "babylm", "params": 350,
                                                 "source": "step_tags",
                                                 "tokens_per_step": 819_200},
    "znhoughton/opt-babylm-1.3b-20eps-seed964": {"family": "babylm", "params": 1300,
                                                 "source": "step_tags",
                                                 "tokens_per_step": 1_024_000},
}


def discover_checkpoints(repo_id: str, source: str, tokens_per_step: int) -> List[Dict[str, Any]]:
    """Checkpoints exposed by a repo, oldest first."""
    refs = HfApi().list_repo_refs(repo_id)
    out = []

    if source == "step_branches":
        for b in refs.branches:
            if b.name.startswith("step") and b.name[4:].isdigit():
                out.append({"checkpoint": b.name, "revision": b.name,
                            "step": int(b.name[4:])})
    elif source == "stage1_branches":
        for b in refs.branches:
            if b.name.startswith("stage1-step"):
                tail = b.name[len("stage1-step"):].split("-")[0]
                if tail.isdigit():
                    out.append({"checkpoint": b.name, "revision": b.name,
                                "step": int(tail)})
    elif source == "step_tags":
        for t in refs.tags:
            if t.name.startswith("step-") and t.name[5:].isdigit():
                out.append({"checkpoint": t.name, "revision": t.name,
                            "step": int(t.name[5:])})
    else:
        raise ValueError(f"unknown checkpoint source: {source}")

    for c in out:
        c["tokens"] = c["step"] * tokens_per_step
    out.sort(key=lambda c: c["step"])
    return out


def log_sample(checkpoints: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """
    Sample on a log scale. Learning curves move fastest early, so evenly spaced
    steps would spend most of the budget where nothing is changing. The first
    and last checkpoints are always kept.
    """
    total = len(checkpoints)
    if total <= n:
        return checkpoints
    idx = sorted({min(int(round(np.exp(x))) - 1, total - 1)
                  for x in np.linspace(np.log(1), np.log(total), n)})
    return [checkpoints[i] for i in idx]


def score_checkpoint(model_name: str, cfg: Dict[str, Any], ckpt: Dict[str, Any],
                     device: str, out_dir: str) -> None:
    slug     = _base.model_slug(model_name)
    out_path = os.path.join(out_dir, f"{slug}__{ckpt['checkpoint']}.csv")

    if os.path.exists(out_path):
        todo = _base.missing_prompts(out_path, _base._ITEMS_DF)
        if not todo:
            print(f"  {slug} @ {ckpt['checkpoint']}: complete, skipping")
            return

    # The tokenizer comes from the default branch, not the checkpoint revision:
    # a model's vocabulary does not change during training, so pinning it to the
    # revision buys nothing and exposes the run to incomplete per-branch configs.
    # pythia-2.8b is the case in point. At step0 and step1000 its tokenizer
    # reports eos_token = None, which made the pad fallback below a silent no-op
    # and then failed with "Asking to pad but the tokenizer does not have a
    # padding token". The same model's default branch and later steps are fine,
    # and pythia-1.4b has eos at every revision, so this is a per-branch repo
    # inconsistency rather than anything about the model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        # Padded positions are masked out of the summed log-probability, so the
        # identity of the pad token cannot affect a score. Any in-vocabulary id
        # is therefore safe as a last resort.
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    os.makedirs(_base.HF_CACHE_DIR, exist_ok=True)
    tmp_cache = tempfile.mkdtemp(prefix="hf_ckpt_", dir=_base.HF_CACHE_DIR)
    model = None
    try:
        load_kwargs: Dict[str, Any] = dict(low_cpu_mem_usage=True, cache_dir=tmp_cache,
                                           revision=ckpt["revision"],
                                           torch_dtype=torch.bfloat16)
        load_kwargs["device_map"] = "auto" if cfg.get("large_model") else device
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).eval()

        dfs = []
        for idx, prompt in enumerate(_base.LIST_OF_PROMPTS):
            dfs.append(_base.score_prompt(idx, prompt, model_name, cfg, tokenizer, model))
        df = pd.concat(dfs, ignore_index=True)
        df["checkpoint"] = ckpt["checkpoint"]
        df["step"]       = ckpt["step"]
        df["tokens"]     = ckpt["tokens"]

        _base.atomic_write_csv(df.sort_values(["prompt_value", "item_id"]), out_path)
        print(f"  {slug} @ {ckpt['checkpoint']} ({ckpt['tokens']:,} tokens): "
              f"{len(df)} rows")
    finally:
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_cache, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-checkpoints", type=int, default=15,
                    help="log-spaced checkpoints per model (default 15)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    configs = CHECKPOINT_CONFIGS
    if args.models:
        chosen, unknown = {}, []
        for want in args.models:
            hits = [m for m in configs if m == want or _base.model_slug(m) == want]
            if hits:
                chosen.update({m: configs[m] for m in hits})
            else:
                unknown.append(want)
        if unknown:
            raise SystemExit(f"unknown model(s): {', '.join(unknown)}")
        configs = chosen

    out_dir = args.out_dir or OUT_DIR
    _base._ITEMS_DF = _base.load_items()
    n_items = len(_base._ITEMS_DF)

    plan = []
    for model_name, cfg in configs.items():
        try:
            ckpts = discover_checkpoints(model_name, cfg["source"], cfg["tokens_per_step"])
        except Exception as e:
            print(f"  {model_name}: could not list checkpoints ({type(e).__name__}), skipping")
            continue
        if not ckpts:
            print(f"  {model_name}: no checkpoints found, skipping")
            continue
        sampled = log_sample(ckpts, args.n_checkpoints)
        print(f"  {_base.model_slug(model_name):<34} {len(ckpts):>4} checkpoints "
              f"-> {len(sampled):>3} sampled, "
              f"{sampled[0]['tokens']:,} to {sampled[-1]['tokens']:,} tokens")
        plan.extend((model_name, cfg, c) for c in sampled)

    n_seq = n_items * 2 * len(_base.LIST_OF_PROMPTS)
    print(f"\n{len(plan)} checkpoint(s) x {n_seq:,} sequences each")

    if args.dry_run:
        return

    os.makedirs(out_dir, exist_ok=True)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    for model_name, cfg, ckpt in plan:
        try:
            score_checkpoint(model_name, cfg, ckpt, device, out_dir)
        except Exception as e:
            print(f"  FAILED {model_name} @ {ckpt['checkpoint']}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
    print("\nDONE")


if __name__ == "__main__":
    main()
