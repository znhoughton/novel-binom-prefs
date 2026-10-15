#!/usr/bin/env python3
"""
annotate_constraints.py
------------------------
Annotates curated_1000_plural.csv with phonological/lexical constraint columns
and empty slots for semantic constraints.

Coding convention (positive = word1 first preferred by that constraint):
  Len      = syl(word2) - syl(word1)
  Lapse    = [trailing_unstressed(word2) + leading_unstressed(word1)]
             - [trailing_unstressed(word1) + leading_unstressed(word2)]
  *BStress = +1 if word1 ends stressed and word2 does not; -1 if reverse; 0 otherwise
  Freq     = ln(dolma_count(word1) / dolma_count(word2))

Blank columns (for manual annotation):
  Form, Percept, Culture, Power, Intense, Icon, OverallFreq, RelFreq

Usage:
    python Scripts/annotate_constraints.py [--input ...] [--output ...]
"""

import argparse
import csv
import json
import math
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import nltk
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "Data" / "curated_1000_plural.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "Data" / "curated_1000_plural_coded.csv"
DEFAULT_CKPT   = PROJECT_ROOT / "Data" / "checkpoint_dolma_unigrams.json"

API_URL         = "https://api.infini-gram.io/"
INDEX           = "v4_dolma-v1_7_llama"
REQUEST_DELAY   = 2.0
RATE_LIMIT_WAIT = 60.0
MAX_RETRIES     = 8
BACKOFF_BASE    = 2.0

SEMANTIC_COLS = ["Form", "Percept", "Culture", "Power", "Intense", "Icon",
                 "OverallFreq", "RelFreq"]


# ── CMU dict setup ────────────────────────────────────────────────────────────

def _ensure_nltk(resource, pkg):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(pkg, quiet=True)

_ensure_nltk("corpora/cmudict", "cmudict")
from nltk.corpus import cmudict
CMU = cmudict.dict()

DEFAULT_SUPPLEMENT = PROJECT_ROOT / "Data" / "supplement_phones.json"
_SUPPLEMENT: dict = {}
if DEFAULT_SUPPLEMENT.exists():
    with open(DEFAULT_SUPPLEMENT, encoding="utf-8") as _f:
        _SUPPLEMENT = {k: v for k, v in json.load(_f).items() if not k.startswith("_")}


def get_phones(word: str):
    """Return ARPAbet phone list for word, falling back to supplement_phones.json."""
    result = CMU.get(word.lower(), [None])[0]
    if result is not None:
        return result
    return _SUPPLEMENT.get(word.lower())

def stress_pattern(phones) -> list:
    return [int(p[-1]) for p in phones if p[-1].isdigit()]

def trailing_unstressed(phones) -> int:
    stresses = stress_pattern(phones)
    for i in range(len(stresses) - 1, -1, -1):
        if stresses[i] > 0:
            return len(stresses) - 1 - i
    return len(stresses)

def leading_unstressed(phones) -> int:
    stresses = stress_pattern(phones)
    for i, s in enumerate(stresses):
        if s > 0:
            return i
    return len(stresses)

def final_stressed(phones) -> bool:
    stresses = stress_pattern(phones)
    return bool(stresses) and stresses[-1] > 0

def compute_len(pa, pb) -> int:
    syl_a = sum(1 for p in pa if p[-1].isdigit())
    syl_b = sum(1 for p in pb if p[-1].isdigit())
    return syl_b - syl_a

def compute_lapse(pa, pb) -> int:
    return (trailing_unstressed(pb) + leading_unstressed(pa)
            - trailing_unstressed(pa) - leading_unstressed(pb))

def compute_bstress(pa, pb) -> int:
    a_stressed = final_stressed(pa)
    b_stressed = final_stressed(pb)
    if a_stressed and not b_stressed:
        return 1
    if b_stressed and not a_stressed:
        return -1
    return 0


# ── Dolma unigram count ───────────────────────────────────────────────────────

def query_unigram_count(word: str) -> int:
    """Query Dolma for the unigram count of a single word. Returns -1 on failure."""
    payload = json.dumps({
        "index": INDEX,
        "query_type": "count",
        "query": word,
    }).encode("utf-8")

    transient_attempts = 0
    while transient_attempts < MAX_RETRIES:
        time.sleep(REQUEST_DELAY)
        try:
            req = urllib.request.Request(
                API_URL, data=payload,
                headers={"Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0 (research; dolma-freq-query)"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "error" in data:
                    msg = str(data["error"]).lower()
                    if any(k in msg for k in ("rate", "limit", "lock", "quota")):
                        tqdm.write(f"  [rate-limit] '{word}' — waiting {RATE_LIMIT_WAIT:.0f}s",
                                   file=sys.stderr)
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    raise ValueError(f"API error: {data['error']}")
                return int(data.get("count", 0))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503):
                tqdm.write(f"  [rate-limit HTTP {e.code}] '{word}' — waiting {RATE_LIMIT_WAIT:.0f}s",
                           file=sys.stderr)
                time.sleep(RATE_LIMIT_WAIT)
                continue
            wait = BACKOFF_BASE ** transient_attempts
            tqdm.write(f"  [retry {transient_attempts+1}] '{word}': HTTP {e.code} — {wait:.0f}s",
                       file=sys.stderr)
            time.sleep(wait)
            transient_attempts += 1
        except Exception as e:
            wait = BACKOFF_BASE ** transient_attempts
            tqdm.write(f"  [retry {transient_attempts+1}] '{word}': {e} — {wait:.0f}s",
                       file=sys.stderr)
            time.sleep(wait)
            transient_attempts += 1

    tqdm.write(f"  [ERROR] Giving up on '{word}'.", file=sys.stderr)
    return -1


def fetch_unigram_counts(words: list, ckpt_path: Path) -> dict:
    """Fetch Dolma unigram counts for each word, with checkpointing."""
    counts = {}
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            counts = json.load(f)
        print(f"Loaded {len(counts):,} cached unigram counts.")

    todo = [w for w in words if w not in counts]
    print(f"Already cached: {len(counts) - len([w for w in words if w not in counts]):,} "
          f"| Remaining: {len(todo):,}")

    if todo:
        with tqdm(total=len(todo), desc="Unigram counts", unit="word") as bar:
            for word in todo:
                counts[word] = query_unigram_count(word)
                bar.update(1)
                if len(counts) % 50 == 0:
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump(counts, f)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(counts, f)
        print(f"Saved unigram counts to {ckpt_path}")

    return counts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    args = parser.parse_args()

    # Read pairs
    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows):,} pairs from {args.input}")

    # Collect unique words and fetch Dolma counts
    all_words = sorted({r["word1"] for r in rows} | {r["word2"] for r in rows})
    print(f"Unique words: {len(all_words):,}")
    counts = fetch_unigram_counts(all_words, Path(args.checkpoint))

    # Annotate each pair
    missing_cmu = set()
    out_rows = []
    for r in rows:
        w1, w2 = r["word1"], r["word2"]
        p1 = get_phones(w1)
        p2 = get_phones(w2)

        if p1 is None:
            missing_cmu.add(w1)
        if p2 is None:
            missing_cmu.add(w2)

        if p1 is not None and p2 is not None:
            length   = compute_len(p1, p2)
            lapse    = compute_lapse(p1, p2)
            bstress  = compute_bstress(p1, p2)
        else:
            length = lapse = bstress = ""

        c1 = counts.get(w1, -1)
        c2 = counts.get(w2, -1)
        if c1 > 0 and c2 > 0:
            freq = round(math.log(c1 / c2), 4)
        else:
            freq = ""

        out_row = {
            "word1":       w1,
            "word2":       w2,
            "Form":        "",
            "Percept":     "",
            "Culture":     "",
            "Power":       "",
            "Intense":     "",
            "Icon":        "",
            "Freq":        freq,
            "Len":         length,
            "Lapse":       lapse,
            "*BStress":    bstress,
            "OverallFreq": 0,
            "RelFreq":     0,
        }
        # Carry over any existing columns (shared_domain, avg_zipf, etc.)
        for k, v in r.items():
            if k not in out_row:
                out_row[k] = v

        out_rows.append(out_row)

    # Build fieldnames: standard constraint cols first, then any extras
    standard = ["word1", "word2", "Form", "Percept", "Culture", "Power",
                "Intense", "Icon", "Freq", "Len", "Lapse", "*BStress",
                "OverallFreq", "RelFreq"]
    extra = [k for k in rows[0].keys() if k not in standard]
    fieldnames = standard + extra

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows):,} annotated pairs to {out_path}")
    n_freq_missing = sum(1 for r in out_rows if r["Freq"] == "")
    n_phon_missing = sum(1 for r in out_rows if r["Len"] == "")
    print(f"  Freq missing (zero/error count): {n_freq_missing}")
    print(f"  Phonological missing (not in CMU): {n_phon_missing}")
    if missing_cmu:
        print(f"\nWords not in CMU dict ({len(missing_cmu)}):")
        for w in sorted(missing_cmu):
            print(f"  {w}")


if __name__ == "__main__":
    main()
