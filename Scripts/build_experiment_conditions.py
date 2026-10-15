#!/usr/bin/env python3
"""
build_experiment_conditions.py
------------------------------
Builds the PsychoPy conditions file for the novel-binomial preference experiment.

Joins the audited semantic constraint codings (novel_binoms_coded.csv) with the
phonological/frequency constraints (curated_500_binoms.csv), adds the two
orderings as Alpha/Nonalpha text, and appends a small set of frozen attested
binomials to serve as attention checks.

Row order matters: the experiment's "Selected rows" code assumes the novel items
occupy rows 0..N_NOVEL-1 and the attention checks occupy the rows after them.
The script asserts this before writing.

Column notes:
  Alpha      = "word1 and word2"  (word1 is alphabetically first in every row)
  Nonalpha   = "word2 and word1"
  BStress    = renamed from curated_500_binoms.csv's "*BStress"; the asterisk is
               not a legal PsychoPy/JS variable name. Sign convention is
               unchanged (see annotate_constraints.py): +1 if word1 ends stressed
               and word2 does not, -1 if the reverse.
  is_filler  = 1 for attention checks, 0 for the novel items of interest.
  expected   = for attention checks, which column holds the strongly preferred
               order ("Alpha" or "Nonalpha"). Empty for novel items.

Constraint columns are deliberately left empty for the attention checks: they are
coded in attested_binoms.csv under a coding pass that has not been reconciled
with this project's, and they are excluded from the constraint analysis anyway.

Usage:
    python Scripts/build_experiment_conditions.py
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "Data"
CODED = DATA / "novel_binoms_coded.csv"
CURATED = DATA / "curated_500_binoms.csv"
OUT = PROJECT_ROOT / "Experiment" / "novel_binoms_conditions.csv"

# Frozen attested binomials used as attention checks. RelFreq is the corpus
# proportion of the alphabetical order, taken from the previous study's
# attested_binoms.csv; all of these exceed 0.98 in their preferred direction
# with OverallFreq > 36,000, i.e. they are effectively fixed in English.
# Balanced 5/5 so that always clicking the alphabetical option fails half of them.
ATTENTION_CHECKS = [
    # (word1, word2, expected, relfreq_alpha)
    ("rock", "roll", "Alpha", 0.997205),
    ("arts", "crafts", "Alpha", 0.995250),
    ("crime", "punishment", "Alpha", 0.988594),
    ("bread", "butter", "Alpha", 0.987170),
    ("law", "order", "Alpha", 0.985217),
    ("file", "rank", "Nonalpha", 0.000357),
    ("error", "trial", "Nonalpha", 0.000487),
    ("crannies", "nooks", "Nonalpha", 0.002649),
    ("cents", "dollars", "Nonalpha", 0.003060),
    ("ends", "odds", "Nonalpha", 0.003767),
]

CONSTRAINT_COLS = [
    "Form", "Percept", "Culture", "Power", "Intense", "Icon",
    "Freq", "Len", "Lapse", "BStress",
]
OUT_COLS = (
    ["item_id", "word1", "word2", "Alpha", "Nonalpha", "is_filler", "expected"]
    + CONSTRAINT_COLS
    + ["word1_freq", "word2_freq", "RelFreq"]
)


def read_csv(path):
    # curated_500_binoms.csv carries a UTF-8 BOM
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    coded = read_csv(CODED)
    curated = read_csv(CURATED)

    if len(coded) != len(curated):
        raise SystemExit(
            f"row count mismatch: {CODED.name} has {len(coded)}, "
            f"{CURATED.name} has {len(curated)}"
        )

    rows = []
    for i, (c, u) in enumerate(zip(coded, curated), start=1):
        if (c["word1"], c["word2"]) != (u["word1"], u["word2"]):
            raise SystemExit(
                f"pair mismatch at row {i}: {c['word1']}/{c['word2']} vs "
                f"{u['word1']}/{u['word2']} -- the two files are no longer aligned"
            )
        w1, w2 = c["word1"], c["word2"]
        if w1 >= w2:
            raise SystemExit(f"row {i}: word1 ({w1}) is not alphabetically first")

        row = {
            "item_id": f"nov_{i:03d}",
            "word1": w1,
            "word2": w2,
            "Alpha": f"{w1} and {w2}",
            "Nonalpha": f"{w2} and {w1}",
            "is_filler": 0,
            "expected": "",
            # semantic constraints from the audited file
            "Form": c["Form"], "Percept": c["Percept"], "Culture": c["Culture"],
            "Power": c["Power"], "Intense": c["Intense"], "Icon": c["Icon"],
            # phonological/frequency constraints from the curated file
            "Freq": u["Freq"], "Len": u["Len"], "Lapse": u["Lapse"],
            "BStress": u["*BStress"],
            "word1_freq": u["word1_freq"], "word2_freq": u["word2_freq"],
            "RelFreq": "",  # novel items are zero-frequency in both orders
        }
        rows.append(row)

    n_novel = len(rows)

    for j, (w1, w2, expected, relfreq) in enumerate(ATTENTION_CHECKS, start=1):
        if w1 >= w2:
            raise SystemExit(f"attention check {w1}/{w2}: word1 not alphabetically first")
        row = {col: "" for col in OUT_COLS}
        row.update({
            "item_id": f"check_{j:02d}",
            "word1": w1,
            "word2": w2,
            "Alpha": f"{w1} and {w2}",
            "Nonalpha": f"{w2} and {w1}",
            "is_filler": 1,
            "expected": expected,
            "RelFreq": relfreq,
        })
        rows.append(row)

    # The experiment's row-selection code depends on this layout.
    assert all(r["is_filler"] == 0 for r in rows[:n_novel])
    assert all(r["is_filler"] == 1 for r in rows[n_novel:])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}")
    print(f"  novel items     : rows 0-{n_novel - 1}  (n={n_novel})")
    print(f"  attention checks: rows {n_novel}-{len(rows) - 1}  (n={len(rows) - n_novel})")
    print()
    print(f"  set n_novel = {n_novel} and n_checks = {len(rows) - n_novel} "
          "in the experiment's Begin Experiment code")


if __name__ == "__main__":
    main()
