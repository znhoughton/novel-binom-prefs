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


def open_cache(corpus):
    """Shared with the surface mode: a query answered once is never re-asked,
    so running lemma mode after surface mode only pays for the new strings."""
    path = PROJECT_ROOT / "Data" / f".novelty_cache_{corpus}.json"
    cache = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    return path, cache


def save_cache(path, cache):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp, str(path))


def main_lemma(args):
    """Novelty against number, case and order variants rather than the bare
    surface string. Writes Data/novelty_lemma_<corpus>.csv."""
    import inflect

    engine = inflect.engine()
    cache_path, cache = open_cache(args.corpus)

    df = pd.read_csv(ITEMS_CSV)
    df = df[df["is_filler"] == 0].copy()
    df["word1"] = df["word1"].str.lower()
    df["word2"] = df["word2"].str.lower()

    plan = {r.item_id: item_variants(r.word1, r.word2, engine)
            for r in df.itertuples(index=False)}
    all_q = {q for v in plan.values() for q in v}
    todo = [q for q in all_q if q not in cache]
    print(f"{len(df)} items, {len(all_q):,} distinct strings, "
          f"{len(cache):,} already cached, {len(todo):,} to fetch "
          f"(~{len(todo) / 3600:.1f} h) against {args.index}", flush=True)

    done = 0
    for q in todo:
        cache[q] = query_count(q, args.index)
        done += 1
        if done % 100 == 0:
            save_cache(cache_path, cache)
            print(f"  {done}/{len(todo)} queries", flush=True)
    save_cache(cache_path, cache)

    rows = []
    for r in df.itertuples(index=False):
        v = plan[r.item_id]
        trusted = [q for q in v if case_pattern(q) in TRUSTED_PATTERNS]
        hits = {q: cache.get(q, 0) for q in trusted if cache.get(q, 0) > 0}
        untrusted = {q: cache.get(q, 0) for q in v
                     if case_pattern(q) not in TRUSTED_PATTERNS and cache.get(q, 0) > 0}
        surf = max(cache.get(f"{r.word1} and {r.word2}", 0), 0) + \
               max(cache.get(f"{r.word2} and {r.word1}", 0), 0)
        rows.append({
            "item_id": r.item_id, "word1": r.word1, "word2": r.word2,
            "corpus": args.corpus,
            "n_variants": len(trusted), "n_attested": len(hits),
            "surface_count": surf,
            "total_count": sum(hits.values()),
            "max_count": max(hits.values(), default=0),
            "novel_surface": surf == 0,
            "novel_lemma": len(hits) == 0,
            "attested_forms": "; ".join(f"{q}={c}" for q, c in sorted(hits.items())),
            ## Carried for transparency only; excluded from the verdict.
            "untrusted_hits": len(untrusted),
        })

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "Data" / f"novelty_lemma_{args.corpus}.csv"
    out.to_csv(out_path, index=False)
    lost = out[out.novel_surface & ~out.novel_lemma]
    print(f"\nnovel on surface criterion : {int(out.novel_surface.sum())}/{len(out)}")
    print(f"novel on lemma criterion   : {int(out.novel_lemma.sum())}/{len(out)}")
    print(f"newly excluded by lemma    : {len(lost)}")
    for r in lost.sort_values('total_count', ascending=False).head(20).itertuples(index=False):
        print(f"  {r.item_id}  {r.word1}/{r.word2}  n={r.total_count}  {r.attested_forms[:80]}")
    print(f"\nwrote {out_path}")
    return 0


def number_variants(word, engine):
    """The word as it appears in the item set, plus its singular.

    Every word in the set is a plural noun, so this is a two-element list except
    where singularisation is a no-op. `inflect` handles the irregulars that
    matter here (geese -> goose, fireflies -> firefly).
    """
    sg = engine.singular_noun(word)
    return [word] if not sg or sg == word else [word, sg]


## Title-case and all-caps counts from infini-gram are not usable, and the
## evidence is unambiguous: every all-caps forward/reverse pair in our data
## returns an IDENTICAL count (283 of 283 in Dolma, 141 of 141 in the Pile),
## which a real phrase count cannot do, and title case shows the same pathology
## less often (8 of 28 in Dolma, including 6-vs-6 and 3-vs-3 ties rather than
## only 1-vs-1). Hit rates give the same picture: 27.2% of all-caps queries
## return a nonzero count against 1.5% of lowercase ones. The cause is almost
## certainly that capitalised words fragment into rare token sequences under the
## Llama tokenizer and the index falls back to something order-insensitive.
##
## They are still queried, so that the artifact stays documented in the cache
## and in the output, but they do not count toward the novelty verdict.
TRUSTED_PATTERNS = ("lower", "cap_first")


def case_pattern(query):
    if " AND " in query:
        return "all_caps"
    a, b = query.split(" and ")
    if a.islower() and b.islower():
        return "lower"
    if a[0].isupper() and b[0].isupper():
        return "title"
    return "cap_first"


def case_variants(w1, w2):
    """The four orthographic patterns that occur in running text.

    Mixed forms like "cats and DOGS" are not queried: they do not occur, and
    each one costs a request. Only the first two count toward novelty; see
    TRUSTED_PATTERNS above for why.
    """
    return [
        f"{w1} and {w2}",
        f"{w1.capitalize()} and {w2}",
        f"{w1.capitalize()} and {w2.capitalize()}",
        f"{w1.upper()} AND {w2.upper()}",
    ]


def item_variants(w1, w2, engine):
    """All number x case x order forms for one pair, deduplicated."""
    out = {}
    for a in number_variants(w1, engine):
        for b in number_variants(w2, engine):
            for first, second in ((a, b), (b, a)):
                for q in case_variants(first, second):
                    out[q] = (first == a)      # True when alphabetical order
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--variants", choices=("surface", "lemma"), default="surface",
                    help="surface: the original 4 forms. lemma: also singular/plural "
                         "and title/all-caps, 32 forms per item.")
    args = ap.parse_args()

    if args.variants == "lemma":
        return main_lemma(args)

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
