#!/usr/bin/env python3
"""
check_dolma_singular.py
------------------------
For each pair in curated_1000_plural.csv, singularize both words and check
whether the singular pair appears anywhere in Dolma (v4_dolma-v1_7_llama).

Cross-references the existing checkpoint_dolma.json (from the singular
pipeline) first so we don't re-query pairs already verified.

Checks four surface forms per pair:
    1. "word1 and word2"    — lowercase
    2. "Word1 and word2"    — capitalized
    3. "word2 and word1"    — reversed, lowercase
    4. "Word2 and word1"    — reversed, capitalized

A pair is NOVEL (as singulars) if the total count across all four is 0.

Outputs:
    Data/singular_novelty_check.csv — every pair with counts + novel flag

Usage:
    python Scripts/check_dolma_singular.py
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:
    import inflect
    _inflect = inflect.engine()
    HAS_INFLECT = True
except ImportError:
    HAS_INFLECT = False
    print("Warning: inflect not installed. Run: pip install inflect", file=sys.stderr)

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "Data" / "curated_1000_plural.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "Data" / "singular_novelty_check.csv"
DEFAULT_CKPT   = PROJECT_ROOT / "Data" / "checkpoint_dolma_singular.json"
EXISTING_CKPT  = PROJECT_ROOT / "Data" / "checkpoint_dolma.json"

API_URL         = "https://api.infini-gram.io/"
INDEX           = "v4_dolma-v1_7_llama"
MAX_RETRIES     = 8
BACKOFF_BASE    = 2.0
RATE_LIMIT_WAIT = 60.0
REQUEST_DELAY   = 2.0


def singularize(word: str) -> str:
    if HAS_INFLECT:
        s = _inflect.singular_noun(word)
        return s if s else word
    return word


def query_count(query: str) -> int:
    payload = json.dumps({
        "index": INDEX,
        "query_type": "count",
        "query": query,
    }).encode("utf-8")

    transient_attempts = 0
    while transient_attempts < MAX_RETRIES:
        time.sleep(REQUEST_DELAY)
        try:
            req = urllib.request.Request(
                API_URL, data=payload,
                headers={"Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0 (research; dolma-singular-check)"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "error" in data:
                    msg = str(data["error"]).lower()
                    if any(k in msg for k in ("rate", "limit", "lock", "quota")):
                        tqdm.write(f"  [rate-limit] '{query}' — waiting {RATE_LIMIT_WAIT:.0f}s",
                                   file=sys.stderr)
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    raise ValueError(f"API error: {data['error']}")
                return int(data.get("count", 0))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503):
                tqdm.write(f"  [rate-limit HTTP {e.code}] '{query}' — waiting {RATE_LIMIT_WAIT:.0f}s",
                           file=sys.stderr)
                time.sleep(RATE_LIMIT_WAIT)
                continue
            wait = BACKOFF_BASE ** transient_attempts
            tqdm.write(f"  [retry {transient_attempts+1}] '{query}': HTTP {e.code} — {wait:.0f}s",
                       file=sys.stderr)
            time.sleep(wait)
            transient_attempts += 1
        except Exception as e:
            wait = BACKOFF_BASE ** transient_attempts
            tqdm.write(f"  [retry {transient_attempts+1}] '{query}': {e} — {wait:.0f}s",
                       file=sys.stderr)
            time.sleep(wait)
            transient_attempts += 1

    tqdm.write(f"  [ERROR] Giving up on '{query}'.", file=sys.stderr)
    return -1


def check_pair(s1: str, s2: str) -> dict:
    forms = {
        "order1_lower":   f"{s1} and {s2}",
        "order1_capital": f"{s1.capitalize()} and {s2}",
        "order2_lower":   f"{s2} and {s1}",
        "order2_capital": f"{s2.capitalize()} and {s1}",
    }
    counts = {}
    error = False
    for key, query in forms.items():
        c = query_count(query)
        if c < 0:
            error = True
        counts[key] = c

    total = sum(v for v in counts.values() if v >= 0)
    novel = (total == 0) and not error
    return {**counts, "total_count": total, "novel": novel, "error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=str(DEFAULT_INPUT))
    parser.add_argument("--output",     default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--workers",    type=int, default=5)
    parser.add_argument("--delay",      type=float, default=None)
    args = parser.parse_args()

    global REQUEST_DELAY
    if args.delay is not None:
        REQUEST_DELAY = args.delay

    # Read plural pairs
    plural_pairs = []
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            plural_pairs.append((row["word1"].strip(), row["word2"].strip()))
    print(f"Loaded {len(plural_pairs):,} plural pairs.")

    # Convert to singular pairs (norm alphabetical order)
    def norm(a, b):
        return (a, b) if a <= b else (b, a)

    singular_pairs = []
    for w1, w2 in plural_pairs:
        s1, s2 = singularize(w1), singularize(w2)
        singular_pairs.append(norm(s1, s2))
    print(f"Singular pairs (unique): {len(set(singular_pairs)):,}")

    # Load existing singular checkpoint
    ckpt_path = Path(args.checkpoint)
    results = {}
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results):,} results from {ckpt_path.name}")

    # Cross-reference the original singular pipeline checkpoint
    if EXISTING_CKPT.exists():
        with open(EXISTING_CKPT, encoding="utf-8") as f:
            existing = json.load(f)
        n_found = 0
        for s1, s2 in set(singular_pairs):
            key = f"{s1}|{s2}"
            if key not in results and key in existing:
                results[key] = existing[key]
                n_found += 1
        print(f"Found {n_found:,} pairs in existing singular checkpoint.")

    done_keys = set(results.keys())
    todo = [(s1, s2) for s1, s2 in set(singular_pairs) if f"{s1}|{s2}" not in done_keys]
    print(f"Already done: {len(done_keys):,}  |  Remaining to query: {len(todo):,}\n")

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(check_pair, s1, s2): (s1, s2) for s1, s2 in todo}
            with tqdm(total=len(todo), desc="Singular pairs", unit="pair") as bar:
                for fut in as_completed(futures):
                    s1, s2 = futures[fut]
                    key = f"{s1}|{s2}"
                    try:
                        result = fut.result()
                    except Exception as e:
                        tqdm.write(f"  [ERROR] ({s1}, {s2}): {e}", file=sys.stderr)
                        result = {
                            "order1_lower": 0, "order1_capital": 0,
                            "order2_lower": 0, "order2_capital": 0,
                            "total_count": -1, "novel": False, "error": True,
                        }
                    results[key] = result
                    bar.update(1)
                    if len(results) % 50 == 0:
                        with open(ckpt_path, "w", encoding="utf-8") as f:
                            json.dump(results, f)

        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(results, f)

    # Write output
    fields = ["plural_word1", "plural_word2", "singular_word1", "singular_word2",
              "order1_lower", "order1_capital", "order2_lower", "order2_capital",
              "total_count", "novel", "error"]
    out_rows = []
    for (pw1, pw2), (sw1, sw2) in zip(plural_pairs, singular_pairs):
        key = f"{sw1}|{sw2}"
        r = results.get(key, {})
        out_rows.append({
            "plural_word1":  pw1,
            "plural_word2":  pw2,
            "singular_word1": sw1,
            "singular_word2": sw2,
            **{k: r.get(k, "") for k in ["order1_lower", "order1_capital",
                                          "order2_lower", "order2_capital",
                                          "total_count", "novel", "error"]},
        })

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    n_novel  = sum(1 for r in out_rows if str(r.get("novel", "")).lower() == "true")
    n_errors = sum(1 for r in out_rows if str(r.get("error", "")).lower() == "true")
    n_not_novel = len(out_rows) - n_novel - n_errors
    print(f"\nDone.")
    print(f"  Total pairs      : {len(out_rows):,}")
    print(f"  Novel as singular: {n_novel:,}")
    print(f"  NOT novel        : {n_not_novel:,}")
    print(f"  Errors           : {n_errors:,}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
