#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_bigram_logodds.py
-------------------------
Corpus bigram log-odds for each novel binomial, as a surface-statistics control.

The trigram "word1 and word2" is zero for every item by construction: that is
the novelty criterion check_dolma.py enforces. So the phrase itself carries no
information, and the surface baseline has to be built from the component
bigrams, which are not zero:

    C(w1 and), C(and w2), C(w2 and), C(and w1)

Under a first-order Markov model, log P(w1 and w2) - log P(w2 and w1) reduces to

    log C(w1 and) + log C(and w2) - log C(w2 and) - log C(and w1)

because the unigram terms C(w1), C(w2) and the shared C(and) cancel between the
two orderings. Multiplying (adding in logs) rather than summing the counts is
what the factorisation requires: both transitions are needed to build the
phrase, so a zero in either position should collapse the estimate, which a sum
would not do.

Positive = corpus bigram statistics favour the alphabetical order, matching the
sign convention of `preference` and `alpha_answer`.

The counts are per word, not per pair, so the 458 items need only
2 x (distinct words) queries rather than 4 x 458.

Raw counts are written alongside the derived value so the smoothing choice
stays revisable without re-querying.

Output: Data/bigram_logodds.csv
  item_id, word1, word2, corpus,
  c_w1_and, c_and_w2, c_w2_and, c_and_w1, bigram_logodds

Usage:
  python Scripts/compute_bigram_logodds.py
  python Scripts/compute_bigram_logodds.py --index v4_piletrain_llama --corpus pile
"""

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ITEMS_CANDIDATES = [
    PROJECT_ROOT / "Experiment" / "novel_binoms_conditions.csv",
    PROJECT_ROOT / "Data" / "novel_binoms_conditions.csv",
]
ITEMS_CSV = next((p for p in _ITEMS_CANDIDATES if p.exists()), _ITEMS_CANDIDATES[0])
OUT_CSV   = PROJECT_ROOT / "Data" / "bigram_logodds.csv"
CACHE     = PROJECT_ROOT / "Data" / ".bigram_count_cache.json"

API_URL         = "https://api.infini-gram.io/"
MAX_RETRIES     = 8
BACKOFF_BASE    = 2.0
RATE_LIMIT_WAIT = 60.0
REQUEST_DELAY   = 1.0

# Added to each raw count before the log. Load-bearing rather than cosmetic: a
# genuine zero would otherwise send the log to -inf. 0.5 follows
# hand-trained-model-prefs/analysis-scripts/compute_bigram_logodds.py.
PSEUDOCOUNT = 0.5


def query_count(query: str, index: str) -> int:
    """Count occurrences of `query` in `index`. Returns -1 if retries run out."""
    payload = json.dumps({
        "index": index, "query_type": "count", "query": query,
    }).encode("utf-8")

    attempts = 0
    while attempts < MAX_RETRIES:
        time.sleep(REQUEST_DELAY)
        try:
            req = urllib.request.Request(
                API_URL, data=payload,
                headers={"Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0 (research; bigram-logodds)"},
                method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and data.get("error"):
                msg = str(data["error"]).lower()
                if "rate" in msg or "lockout" in msg:
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                raise RuntimeError(data["error"])
            return int(data["count"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(RATE_LIMIT_WAIT)
                continue
            attempts += 1
            time.sleep(BACKOFF_BASE * (2 ** attempts))
        except Exception:
            attempts += 1
            time.sleep(BACKOFF_BASE * (2 ** attempts))
    return -1


def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    tmp = str(CACHE) + ".tmp"
    Path(tmp).write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp, CACHE)


def counts_from_trigram_table(path: str, words: set):
    """
    Derive the same two per-word counts from a local trigram table, for corpora
    infini-gram does not index (the BabyLM models' training corpus).

    Every occurrence of the bigram "w and" sits inside some trigram "w and Y",
    so summing over Y recovers C(w and) exactly -- but only if the table is
    unpruned. babylm_eng_trigrams.csv is: its minimum count is 1 and 79% of rows
    are singletons. A frequency-thresholded table would undercount, and would do
    so worst for rare words, which is precisely where it would bias a frequency
    comparison.
    """
    w_and = {w: 0 for w in words}
    and_w = {w: 0 for w in words}
    n = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            i = line.rfind(",")
            if i == -1:
                continue
            toks = line[:i].split()
            if len(toks) != 3 or toks[1] != "and":
                continue
            try:
                c = int(line[i + 1:])
            except ValueError:
                continue
            if toks[0] in w_and:
                w_and[toks[0]] += c
            if toks[2] in and_w:
                and_w[toks[2]] += c
            n += 1
    print(f"  scanned {n:,} 'X and Y' trigrams")
    return w_and, and_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="v4_dolma-v1_7_llama",
                    help="infini-gram index (default: Dolma v1.7, the corpus the "
                         "novelty check used)")
    ap.add_argument("--corpus", default="dolma",
                    help="label written to the corpus column")
    ap.add_argument("--trigram-table", default=None,
                    help="count from a local unpruned trigram CSV (trigram,count) "
                         "instead of the API, for corpora infini-gram lacks")
    ap.add_argument("--out", default=None, help="output path (default Data/bigram_logodds.csv)")
    args = ap.parse_args()

    df = pd.read_csv(ITEMS_CSV)
    df = df[df["is_filler"] == 0].copy()
    df["word1"] = df["word1"].str.lower()
    df["word2"] = df["word2"].str.lower()

    words = sorted(set(df["word1"]) | set(df["word2"]))
    print(f"{len(df)} items, {len(words)} distinct words "
          f"-> {2 * len(words)} queries against {args.index}")

    if args.trigram_table:
        w_and, and_w = counts_from_trigram_table(args.trigram_table, set(words))
    else:
        cache = load_cache()
        def cached(query: str) -> int:
            key = f"{args.index}|{query}"
            if key not in cache:
                cache[key] = query_count(query, args.index)
                if len(cache) % 20 == 0:
                    save_cache(cache)
            return cache[key]

        w_and, and_w = {}, {}
        for i, w in enumerate(words, 1):
            w_and[w] = cached(f"{w} and")
            and_w[w] = cached(f"and {w}")
            if i % 25 == 0 or i == len(words):
                print(f"  {i}/{len(words)} words", flush=True)
        save_cache(cache)

        failed = [w for w in words if w_and[w] < 0 or and_w[w] < 0]
        if failed:
            raise SystemExit(f"{len(failed)} word(s) failed to retrieve, first: {failed[:5]}. "
                             "Rerun to resume from the cache.")

    P = PSEUDOCOUNT
    rows = []
    for r in df.itertuples(index=False):
        w1, w2 = r.word1, r.word2
        a, b = w_and[w1] + P, and_w[w2] + P      # alphabetical ordering
        c, d = w_and[w2] + P, and_w[w1] + P      # reverse
        rows.append({
            "item_id": r.item_id, "word1": w1, "word2": w2, "corpus": args.corpus,
            "c_w1_and": w_and[w1], "c_and_w2": and_w[w2],
            "c_w2_and": w_and[w2], "c_and_w1": and_w[w1],
            "bigram_logodds": math.log(a) + math.log(b) - math.log(c) - math.log(d),
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else OUT_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    zero_both = sum(1 for w in words if w_and[w] == 0 and and_w[w] == 0)
    print(f"\nwrote {len(out)} rows -> {out_path}")
    print(f"bigram_logodds: mean {out.bigram_logodds.mean():+.3f}  "
          f"sd {out.bigram_logodds.std():.3f}  "
          f"range {out.bigram_logodds.min():+.2f}..{out.bigram_logodds.max():+.2f}")
    print(f"words with zero counts in both positions: {zero_both}/{len(words)}")


if __name__ == "__main__":
    main()
