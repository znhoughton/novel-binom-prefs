#!/usr/bin/env python3
"""
check_dolma.py
--------------
Query the infini-gram API (index: v4_dolma-v1_7_llama, full Dolma v1.7,
2.6 T tokens) to find which candidate binomial pairs are truly novel.

For each pair (word1, word2) we check four surface forms:
    1. "word1 and word2"   — mid-sentence lowercase
    2. "Word1 and word2"   — sentence-initial (first word capitalised)
    3. "word2 and word1"   — reverse, mid-sentence
    4. "Word2 and word1"   — reverse, sentence-initial

A pair is NOVEL if the total count across all four forms is 0.

Checkpointing: results are saved incrementally to Data/checkpoint_dolma.json
so the run can be resumed after interruption.

Outputs:
    Data/all_results_dolma.csv    — every candidate with counts
    Data/novel_binomials.csv      — novel pairs only (word1, word2)

Rate limiting:
    Each worker sleeps 0.5 s between requests. On HTTP 429 / 503 or a
    rate-limit body the worker backs off for 60 s before retrying.
    Keep --workers low (default 5) to avoid triggering the lockout.

Usage:
    python Scripts/check_dolma.py [--input Data/candidates.csv]
                                  [--workers 5]
"""

import csv
import json
import sys
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "Data" / "candidates.csv"
DEFAULT_ALL    = PROJECT_ROOT / "Data" / "all_results_dolma.csv"
DEFAULT_NOVEL  = PROJECT_ROOT / "Data" / "novel_binomials.csv"
DEFAULT_CKPT   = PROJECT_ROOT / "Data" / "checkpoint_dolma.json"

API_URL          = "https://api.infini-gram.io/"
INDEX            = "v4_dolma-v1_7_llama"
MAX_RETRIES      = 8
BACKOFF_BASE     = 2.0    # seconds; doubles on each transient retry
RATE_LIMIT_WAIT  = 60.0   # seconds to wait when the API signals rate-limiting
REQUEST_DELAY    = 2.0    # seconds between every request within a worker (overridden by --delay)


# ── API helper ────────────────────────────────────────────────────────────────

def query_count(query: str) -> int:
    """
    POST a count query to the infini-gram API and return the integer count.

    Rate-limit handling (HTTP 429 or 503 / 'lockout' body):
        Wait RATE_LIMIT_WAIT seconds before retrying — does not consume a
        retry slot so the worker stays blocked until the lockout clears.

    Transient errors (timeout, connection reset, etc.):
        Exponential backoff starting at BACKOFF_BASE seconds, up to MAX_RETRIES.

    Returns -1 if all retries are exhausted (pair is NOT marked novel).
    """
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
                API_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (research; dolma-novelty-checker)",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # Some APIs return an error field instead of raising HTTP error
                if "error" in data:
                    msg = str(data["error"]).lower()
                    if any(kw in msg for kw in ("rate", "limit", "lock", "quota")):
                        tqdm.write(f"  [rate-limit] '{query}' — waiting {RATE_LIMIT_WAIT:.0f}s",
                                   file=sys.stderr)
                        time.sleep(RATE_LIMIT_WAIT)
                        continue   # retry without consuming transient slot
                    raise ValueError(f"API error: {data['error']}")
                return int(data.get("count", 0))

        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503):
                # Treat all of these as rate-limit / lockout signals.
                # 403 is returned by the infini-gram API under heavy load.
                tqdm.write(f"  [rate-limit HTTP {e.code}] '{query}' — waiting {RATE_LIMIT_WAIT:.0f}s",
                           file=sys.stderr)
                time.sleep(RATE_LIMIT_WAIT)
                continue   # retry without consuming transient slot
            # Other HTTP errors — treat as transient
            wait = BACKOFF_BASE ** transient_attempts
            tqdm.write(f"  [retry {transient_attempts+1}/{MAX_RETRIES}] '{query}': HTTP {e.code} — "
                       f"waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            transient_attempts += 1

        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            wait = BACKOFF_BASE ** transient_attempts
            tqdm.write(f"  [retry {transient_attempts+1}/{MAX_RETRIES}] '{query}': {e} — "
                       f"waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            transient_attempts += 1

    tqdm.write(f"  [ERROR] Giving up on '{query}' after {MAX_RETRIES} retries.", file=sys.stderr)
    return -1   # unknown; pair will not be marked novel


# ── per-pair worker ───────────────────────────────────────────────────────────

def check_pair(w1: str, w2: str) -> dict:
    """
    Check all four surface forms of a pair.
    Returns a result dict with counts for each form and the novel flag.
    """
    forms = {
        "order1_lower":   f"{w1} and {w2}",
        "order1_capital": f"{w1.capitalize()} and {w2}",
        "order2_lower":   f"{w2} and {w1}",
        "order2_capital": f"{w2.capitalize()} and {w1}",
    }
    counts = {}
    error  = False
    for key, query in forms.items():
        c = query_count(query)
        if c < 0:
            error = True
        counts[key] = c

    total = sum(v for v in counts.values() if v >= 0)
    novel = (total == 0) and not error

    return {
        "word1": w1,
        "word2": w2,
        **counts,
        "total_count": total,
        "novel": novel,
        "error": error,
    }


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_candidates(path: Path) -> list:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w1 = row.get("word1", "").strip().lower()
            w2 = row.get("word2", "").strip().lower()
            if w1 and w2 and w1 != w2:
                pairs.append((w1, w2))
    return pairs


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, results: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f)


def write_outputs(results: dict, all_path: Path, novel_path: Path):
    all_fields = [
        "word1", "word2",
        "order1_lower", "order1_capital",
        "order2_lower", "order2_capital",
        "total_count", "novel", "error",
    ]
    rows = sorted(results.values(), key=lambda r: (r["word1"], r["word2"]))

    with open(all_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(rows)

    novel_rows = [r for r in rows if r["novel"]]
    with open(novel_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word1", "word2"])
        w.writeheader()
        for r in novel_rows:
            w.writerow({"word1": r["word1"], "word2": r["word2"]})

    return novel_rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check candidate binomial pairs against Dolma via infini-gram."
    )
    parser.add_argument("--input",   default=str(DEFAULT_INPUT))
    parser.add_argument("--output",  default=str(DEFAULT_NOVEL))
    parser.add_argument("--all",     default=str(DEFAULT_ALL))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel worker threads (default 5). Keep low to avoid rate-limiting.")
    parser.add_argument("--delay", type=float, default=None,
                        help="Per-request sleep in seconds (default: 2.0).")
    args = parser.parse_args()

    global REQUEST_DELAY
    if args.delay is not None:
        REQUEST_DELAY = args.delay
    print(f"Request delay: {REQUEST_DELAY}s | Workers: {args.workers}")

    pairs = read_candidates(Path(args.input))
    print(f"Loaded {len(pairs):,} candidate pairs.")

    ckpt_path = Path(args.checkpoint)
    results   = load_checkpoint(ckpt_path)
    done_keys = set(results.keys())

    todo = [(w1, w2) for w1, w2 in pairs if f"{w1}|{w2}" not in done_keys]
    print(f"Already done: {len(done_keys):,}  |  Remaining: {len(todo):,}\n")

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(check_pair, w1, w2): (w1, w2) for w1, w2 in todo}
            with tqdm(total=len(todo), desc="Pairs", unit="pair", dynamic_ncols=True) as bar:
                for fut in as_completed(futures):
                    w1, w2 = futures[fut]
                    key = f"{w1}|{w2}"
                    try:
                        result = fut.result()
                    except Exception as e:
                        tqdm.write(f"  [ERROR] ({w1}, {w2}): {e}", file=sys.stderr)
                        result = {
                            "word1": w1, "word2": w2,
                            "order1_lower": 0, "order1_capital": 0,
                            "order2_lower": 0, "order2_capital": 0,
                            "total_count": -1, "novel": False, "error": True,
                        }
                    results[key] = result
                    bar.update(1)

                    # Save checkpoint every 50 completions
                    if len(results) % 50 == 0:
                        save_checkpoint(ckpt_path, results)

        save_checkpoint(ckpt_path, results)

    novel_rows = write_outputs(results, Path(args.all), Path(args.output))

    n_novel  = len(novel_rows)
    n_errors = sum(1 for r in results.values() if r.get("error"))
    print(f"\nDone.")
    print(f"  Total checked : {len(results):,}")
    print(f"  Novel pairs   : {n_novel:,}")
    print(f"  Errors        : {n_errors:,}")
    print(f"  All results   -> {args.all}")
    print(f"  Novel pairs   -> {args.output}")


if __name__ == "__main__":
    main()
