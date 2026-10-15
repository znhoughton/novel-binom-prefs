#!/usr/bin/env python3
"""
test_experiment.py
------------------
Checks on the novel-binomial experiment. Run after regenerating the conditions
file or editing the row-selection code.

    python Scripts/test_experiment.py

The central requirement is that no participant ever sees the same binomial
twice. That rests on three things, each tested below:

  1. The conditions file contains no binomial twice -- in either word order.
  2. The row-selection code emits distinct indices.
  3. PsychoJS shows each selected row exactly once.

(3) was verified by reading the PsychoJS source rather than assumed:
  - TrialHandler.importConditions does `fields = sheet.shift()` BEFORE applying
    the selection, so index 0 is the first data row, not the header.
  - selectFromArray(array, "3,1,2") does split(",").map(parseInt).map(i =>
    array[i]) -- order-preserving, one element per index, and no de-duplication,
    so distinctness has to come from the indices we generate.
  - _prepareSequence for Method.RANDOM does shuffle(indices.slice()) per rep,
    a permutation; with nReps=1 every selected row is visited exactly once.

This file re-implements the row-selection code in Python. It is a port, not the
code that ships, so test_code_matches_csv() cross-checks the constants actually
written in the .psyexp and the two compiled .js builds against the CSV.
"""

import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "Experiment"
CONDITIONS = EXP / "novel_binoms_conditions.csv"
PSYEXP = EXP / "novel-binom-judgements.psyexp"
JS = EXP / "novel-binom-judgements.js"
JS_LEGACY = EXP / "novel-binom-judgements-legacy-browsers.js"

N_SESSIONS = 2000
SEED = 964

failures = []
checks_run = 0


def check(condition, label, detail=""):
    global checks_run
    checks_run += 1
    if not condition:
        failures.append(label + (("\n      " + detail) if detail else ""))
        print("  FAIL  %s" % label)
        if detail:
            print("        %s" % detail)
    else:
        print("  ok    %s" % label)


def load_rows():
    with open(CONDITIONS, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# 1. conditions file
# ---------------------------------------------------------------------------

def test_conditions_file(rows):
    print("\n[conditions file]")

    raw = CONDITIONS.read_bytes()
    check(not raw.startswith(b"\xef\xbb\xbf"), "no UTF-8 BOM",
          "a BOM corrupts the first column name when PsychoJS decodes the file")
    check(all(ord(c) < 128 for c in raw.decode("utf-8")), "pure ASCII",
          "smart quotes or accents can decode inconsistently in the browser")

    lines = CONDITIONS.read_text(encoding="utf-8").splitlines()
    blank = [i for i, ln in enumerate(lines) if not ln.strip()]
    check(not blank, "no blank lines",
          "sheet_to_json uses blankrows:false, so a blank row would shift every "
          "index after it; found at lines %s" % blank)

    novel = [r for r in rows if r["is_filler"] == "0"]
    checks = [r for r in rows if r["is_filler"] == "1"]
    check(len(rows) == 468, "468 data rows", "got %d" % len(rows))
    check(len(novel) == 458, "458 novel items", "got %d" % len(novel))
    check(len(checks) == 10, "10 attention checks", "got %d" % len(checks))

    # Layout: the row-selection code indexes novel items as 0..n_novel-1 and
    # checks as the block right after, so ordering here is load-bearing.
    check(all(r["is_filler"] == "0" for r in rows[:458]), "rows 0-457 are novel")
    check(all(r["is_filler"] == "1" for r in rows[458:]), "rows 458-467 are checks")

    ids = [r["item_id"] for r in rows]
    dup_ids = [k for k, v in Counter(ids).items() if v > 1]
    check(not dup_ids, "item_id unique", "duplicates: %s" % dup_ids[:5])

    # The requirement, at the file level. Unordered, so a pair that appears once
    # as (a,b) and once as (b,a) is caught too -- those are the same binomial.
    unordered = [frozenset((r["word1"], r["word2"])) for r in rows]
    dup_pairs = [k for k, v in Counter(unordered).items() if v > 1]
    check(not dup_pairs, "no binomial appears twice (either order)",
          "duplicates: %s" % [sorted(p) for p in dup_pairs[:5]])

    bad_self = [r["item_id"] for r in rows if r["word1"] == r["word2"]]
    check(not bad_self, "word1 != word2", "degenerate: %s" % bad_self[:5])

    # Novel items and attention checks must not collide either.
    nov_pairs = {frozenset((r["word1"], r["word2"])) for r in novel}
    chk_pairs = {frozenset((r["word1"], r["word2"])) for r in checks}
    check(not (nov_pairs & chk_pairs), "no overlap between novel items and checks",
          "shared: %s" % [sorted(p) for p in (nov_pairs & chk_pairs)])

    bad_alpha = [r["item_id"] for r in rows
                 if r["Alpha"] != "%s and %s" % (r["word1"], r["word2"])]
    check(not bad_alpha, "Alpha == 'word1 and word2'", "offenders: %s" % bad_alpha[:5])

    bad_non = [r["item_id"] for r in rows
               if r["Nonalpha"] != "%s and %s" % (r["word2"], r["word1"])]
    check(not bad_non, "Nonalpha == 'word2 and word1'", "offenders: %s" % bad_non[:5])

    check(all(r["Alpha"] != r["Nonalpha"] for r in rows),
          "the two orders always differ")

    # word1 alphabetically first keeps the Alpha/Nonalpha convention meaningful
    # and makes a reversed-duplicate impossible by construction.
    bad_sort = [r["item_id"] for r in rows if not r["word1"] < r["word2"]]
    check(not bad_sort, "word1 sorts before word2", "offenders: %s" % bad_sort[:5])

    ws = [r["item_id"] for r in rows
          for f in ("word1", "word2", "Alpha", "Nonalpha")
          if r[f] != r[f].strip() or not r[f]]
    check(not ws, "no empty or untrimmed text fields", "offenders: %s" % ws[:5])

    bad_exp = [r["item_id"] for r in checks if r["expected"] not in ("Alpha", "Nonalpha")]
    check(not bad_exp, "every check names an expected order", "offenders: %s" % bad_exp[:5])

    check(all(r["expected"] == "" for r in novel),
          "novel items have no expected order")

    # Checks are only usable for exclusion if they do not all point the same way;
    # otherwise always clicking one side passes every one of them. Count both
    # directions explicitly: min() over a Counter skips absent keys entirely, so
    # an all-Alpha set would otherwise look perfectly balanced.
    n_alpha = sum(1 for r in checks if r["expected"] == "Alpha")
    n_nonalpha = sum(1 for r in checks if r["expected"] == "Nonalpha")
    check(min(n_alpha, n_nonalpha) >= 4, "checks balanced by direction",
          "Alpha=%d Nonalpha=%d -- a side-biased participant would pass too many"
          % (n_alpha, n_nonalpha))

    sem = ("Form", "Percept", "Culture", "Power", "Intense", "Icon")
    bad_sem = [(r["item_id"], c) for r in novel for c in sem
               if r[c] not in ("-1", "0", "1")]
    check(not bad_sem, "semantic codings are all in {-1,0,1}",
          "offenders: %s" % bad_sem[:5])

    bad_num = [(r["item_id"], c) for r in novel for c in ("Freq", "Len", "Lapse", "BStress")
               if not _is_number(r[c])]
    check(not bad_num, "numeric constraint columns parse", "offenders: %s" % bad_num[:5])

    # Column names become JS variables, so they must be legal identifiers.
    bad_cols = [c for c in rows[0].keys() if not re.fullmatch(r"[A-Za-z_]\w*", c)]
    check(not bad_cols, "column names are legal identifiers",
          "illegal: %s (e.g. '*BStress' would break PsychoPy)" % bad_cols)


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# 2. row selection -- a port of the shipped Begin JS Experiment code
# ---------------------------------------------------------------------------

def select_rows(n_novel=458, n_checks=10, rng=random):
    """Mirror of the Begin JS Experiment block."""
    n_per_participant = round(n_novel / 5)
    all_indices = list(range(n_novel))
    rng.shuffle(all_indices)
    chosen = all_indices[:n_per_participant]
    for i in range(n_checks):
        chosen.append(n_novel + i)
    return chosen


def side_assignments(subset_size, rng=random):
    """Mirror of the side-counterbalancing block."""
    sides = [i % 2 == 0 for i in range(subset_size)]
    rng.shuffle(sides)
    return sides


def test_selection(rows):
    print("\n[row selection, %d simulated sessions]" % N_SESSIONS)
    rng = random.Random(SEED)

    sizes, seen_counts = set(), Counter()
    dup_index_sessions = 0
    dup_binom_sessions = 0
    out_of_range = 0
    wrong_check_count = 0

    for _ in range(N_SESSIONS):
        idx = select_rows(rng=rng)
        sizes.add(len(idx))

        if len(set(idx)) != len(idx):
            dup_index_sessions += 1
        if any(i < 0 or i >= len(rows) for i in idx):
            out_of_range += 1

        trial_rows = [rows[i] for i in idx]
        pairs = [frozenset((r["word1"], r["word2"])) for r in trial_rows]
        if len(set(pairs)) != len(pairs):
            dup_binom_sessions += 1
        if sum(1 for r in trial_rows if r["is_filler"] == "1") != 10:
            wrong_check_count += 1
        for r in trial_rows:
            seen_counts[r["item_id"]] += 1

    check(sizes == {102}, "every session is exactly 102 trials", "sizes seen: %s" % sorted(sizes))
    check(dup_index_sessions == 0, "no session selects a row index twice",
          "%d sessions affected" % dup_index_sessions)
    check(dup_binom_sessions == 0,
          "NO PARTICIPANT SEES A BINOMIAL MORE THAN ONCE",
          "%d sessions affected" % dup_binom_sessions)
    check(out_of_range == 0, "every index lands inside the conditions file",
          "%d sessions affected" % out_of_range)
    check(wrong_check_count == 0, "every session contains all 10 checks",
          "%d sessions affected" % wrong_check_count)

    novel_ids = [r["item_id"] for r in rows if r["is_filler"] == "0"]
    check(all(seen_counts[i] > 0 for i in novel_ids),
          "every novel item is reachable",
          "never shown: %s" % [i for i in novel_ids if not seen_counts[i]][:5])

    check(all(seen_counts[r["item_id"]] == N_SESSIONS for r in rows if r["is_filler"] == "1"),
          "every check appears in every session")

    counts = [seen_counts[i] for i in novel_ids]
    exp = N_SESSIONS / 5
    print("        novel item exposure over %d sessions: min %d, mean %.1f, max %d "
          "(expected %.0f)" % (N_SESSIONS, min(counts), sum(counts) / len(counts),
                               max(counts), exp))

    # selectFromArray takes the comma branch only when the string has a comma;
    # a single-index selection would silently return one trial.
    sel = ",".join(str(i) for i in select_rows(rng=rng))
    check("," in sel, "selection string contains commas",
          "without a comma PsychoJS would take a different branch")
    check(all(p.isdigit() for p in sel.split(",")),
          "selection string is plain integers (parseInt-safe)")


def test_side_counterbalancing():
    print("\n[side counterbalancing, %d simulated sessions]" % N_SESSIONS)
    rng = random.Random(SEED)
    total = 102

    per_session_left = []
    for _ in range(N_SESSIONS):
        sides = side_assignments(total, rng)
        per_session_left.append(sum(sides))

    check(len(sides) == total, "one side assignment per trial")
    check(set(per_session_left) == {total // 2},
          "every participant gets an exactly even split (%d/%d)" % (total // 2, total // 2),
          "splits seen: %s" % sorted(set(per_session_left))[:6])

    # Independent coin flips would NOT give this; that is the point of the change.
    rng2 = random.Random(SEED)
    coin = [sum(rng2.random() < 0.5 for _ in range(total)) for _ in range(N_SESSIONS)]
    check(len(set(coin)) > 1,
          "(reference) independent coin flips would drift, range %d-%d" % (min(coin), max(coin)))


# ---------------------------------------------------------------------------
# 3. the shipped code agrees with the conditions file
# ---------------------------------------------------------------------------

def _grab(text, pattern, label, found):
    m = re.search(pattern, text)
    if m:
        found.append((label, int(m.group(1))))
    return m


def test_code_matches_csv(rows):
    print("\n[shipped code vs conditions file]")
    n_novel = sum(1 for r in rows if r["is_filler"] == "0")
    n_checks = sum(1 for r in rows if r["is_filler"] == "1")

    found = []
    psy = PSYEXP.read_text(encoding="utf-8")
    _grab(psy, r"n_novel = (\d+);", "psyexp/js n_novel", found)
    _grab(psy, r"n_checks = (\d+);", "psyexp/js n_checks", found)
    _grab(psy, r"n_novel = (\d+)&amp;#10;", "psyexp/py n_novel", found)
    _grab(psy, r"n_checks = (\d+)&amp;#10;", "psyexp/py n_checks", found)
    for path in (JS, JS_LEGACY):
        t = path.read_text(encoding="utf-8")
        _grab(t, r"n_novel = (\d+);", "%s n_novel" % path.name, found)
        _grab(t, r"n_checks = (\d+);", "%s n_checks" % path.name, found)

    check(len(found) == 8, "found all 8 count declarations",
          "found %d: %s" % (len(found), [f[0] for f in found]))
    for label, value in found:
        expected = n_novel if "n_novel" in label else n_checks
        check(value == expected, "%s == %d" % (label, expected), "got %d" % value)

    # Everything the page loads must actually be on disk.
    js = JS.read_text(encoding="utf-8")
    for name in re.findall(r"\{'name': '([^']+)', 'path': '[^']+'\}", js):
        check((EXP / name).exists(), "resource present: %s" % name)

    cond_name = CONDITIONS.name
    check(("val=\"%s\"" % cond_name) in psy, "psyexp points at %s" % cond_name)
    check(cond_name in js, "compiled js points at %s" % cond_name)

    # The two builds must not drift apart.
    for label, pat in [("n_novel", r"n_novel = (\d+);"), ("n_checks", r"n_checks = (\d+);")]:
        a = re.search(pat, JS.read_text(encoding="utf-8")).group(1)
        b = re.search(pat, JS_LEGACY.read_text(encoding="utf-8")).group(1)
        check(a == b, "both js builds agree on %s" % label, "%s vs %s" % (a, b))


def main():
    if not CONDITIONS.exists():
        sys.exit("missing %s -- run Scripts/build_experiment_conditions.py first" % CONDITIONS)
    rows = load_rows()
    test_conditions_file(rows)
    test_selection(rows)
    test_side_counterbalancing()
    test_code_matches_csv(rows)

    print("\n" + "=" * 60)
    if failures:
        print("FAILED: %d of %d checks" % (len(failures), checks_run))
        for f in failures:
            print("  - %s" % f)
        sys.exit(1)
    print("PASSED: all %d checks" % checks_run)


if __name__ == "__main__":
    main()
