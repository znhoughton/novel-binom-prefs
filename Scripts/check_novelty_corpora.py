#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_novelty_corpora.py
------------------------
Verify that the binomial trigram is absent from a corpus, for corpora other than
the one the item set was screened against.

check_dolma.py established novelty in Dolma, and that is the corpus the items
were selected on. But the models are trained on different data: Pythia on the
Pile, GPT-2 on WebText, and so on. A pair unattested in Dolma may well be
attested elsewhere, in which case it is not novel for that model and the whole
premise fails for it.

Same criterion as check_dolma.py: four surface forms, novel only if all are 0.

    "word1 and word2"   "Word1 and word2"
    "word2 and word1"   "Word2 and word1"

Output: Data/novelty_<corpus>.csv
  item_id, word1, word2, corpus, lower_fwd, cap_fwd, lower_rev, cap_rev,
  total_count, novel

Usage:
  python Scripts/check_novelty_corpora.py --index v4_piletrain_llama --corpus pile
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ITEMS = [
    PROJECT_ROOT / "Experiment" / "novel_binoms_conditions.csv",
    PROJECT_ROOT / "Data" / "novel_binoms_conditions.csv",
]
ITEMS_CSV = next((p for p in _ITEMS if p.exists()), _ITEMS[0])

API_URL         = "https://api.infini-gram.io/"
MAX_RETRIES     = 8
BACKOFF_BASE    = 2.0
RATE_LIMIT_WAIT = 60.0
REQUEST_DELAY   = 1.0


def query_count(query: str, index: str) -> int:
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
                         "User-Agent": "Mozilla/5.0 (research; novelty-check)"},
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()

    cache_path = PROJECT_ROOT / "Data" / f".novelty_cache_{args.corpus}.json"
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    def cached(q: str) -> int:
        if q not in cache:
            cache[q] = query_count(q, args.index)
            if len(cache) % 25 == 0:
                tmp = str(cache_path) + ".tmp"
                Path(tmp).write_text(json.dumps(cache), encoding="utf-8")
                os.replace(tmp, cache_path)
        return cache[q]

    df = pd.read_csv(ITEMS_CSV)
    df = df[df["is_filler"] == 0].copy()
    df["word1"] = df["word1"].str.lower()
    df["word2"] = df["word2"].str.lower()
    print(f"{len(df)} items x 4 surface forms = {4 * len(df)} queries "
          f"against {args.index}")

    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        w1, w2 = r.word1, r.word2
        counts = {
            "lower_fwd": cached(f"{w1} and {w2}"),
            "cap_fwd":   cached(f"{w1.capitalize()} and {w2}"),
            "lower_rev": cached(f"{w2} and {w1}"),
            "cap_rev":   cached(f"{w2.capitalize()} and {w1}"),
        }
        total = sum(max(c, 0) for c in counts.values())
        rows.append({"item_id": r.item_id, "word1": w1, "word2": w2,
                     "corpus": args.corpus, **counts,
                     "total_count": total, "novel": total == 0})
        if i % 50 == 0 or i == len(df):
            print(f"  {i}/{len(df)} items", flush=True)

    tmp = str(cache_path) + ".tmp"
    Path(tmp).write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp, cache_path)

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "Data" / f"novelty_{args.corpus}.csv"
    out.to_csv(out_path, index=False)

    n_bad = int((~out.novel).sum())
    print(f"\nwrote {len(out)} rows -> {out_path}")
    print(f"novel in {args.corpus}: {int(out.novel.sum())}/{len(out)}")
    if n_bad:
        print(f"\nNOT novel ({n_bad}), highest counts:")
        for _, r in out[~out.novel].nlargest(15, "total_count").iterrows():
            print(f"   {r.word1} and {r.word2}: {int(r.total_count)}")


if __name__ == "__main__":
    main()
