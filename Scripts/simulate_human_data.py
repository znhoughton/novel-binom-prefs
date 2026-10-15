#!/usr/bin/env python3
"""
simulate_human_data.py
----------------------
Generates fake Pavlovia output for the novel-binomial study, so the analysis
pipeline can be exercised before any real participant has run.

    python Scripts/simulate_human_data.py --n 60 --out Data/human-responses-sim

Responses are drawn from a known generative model:

    logit(P(alphabetical)) = beta * (AbsPref - 0.5) + item_effect + participant_effect

so Analysis/human_prefs_analysis.Rmd should recover roughly `--beta` for the
AbsPref coefficient. That is the point: it turns "the Rmd runs" into "the Rmd
recovers the right answer".

A share of simulated participants (--inattentive) answer at random, so the
attention-check exclusion has something to catch.

The column layout mirrors a real Pavlovia file: every conditions-file column is
carried through, `mouse.clicked_name` is a list-as-string accumulating each
click, and instruction rows carry empty item cells.
"""

import argparse
import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ROOT / "Experiment" / "novel_binoms_conditions.csv"

ML_WEIGHTS = {
    "Intercept": 0.02191943, "Form": 0.23925834, "Percept": 0.24889543,
    "Culture": 0.41836997, "Power": 0.25967334, "Intense": 0.01867604,
    "Icon": 1.30365980, "Freq": 0.08553552, "Len": 0.15241566,
    "Lapse": -0.19381657, "BStress": 0.36019221,
}

N_NOVEL, N_CHECKS = 458, 10
N_PER_PARTICIPANT = round(N_NOVEL / 5)   # 92, matching the experiment


def abs_pref(row):
    z = ML_WEIGHTS["Intercept"]
    for k, w in ML_WEIGHTS.items():
        if k == "Intercept":
            continue
        z += w * float(row[k])
    return 1.0 / (1.0 + math.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="participants to simulate")
    ap.add_argument("--beta", type=float, default=3.0,
                    help="true AbsPref coefficient the analysis should recover")
    ap.add_argument("--item-sd", type=float, default=0.5)
    ap.add_argument("--participant-sd", type=float, default=0.3)
    ap.add_argument("--inattentive", type=float, default=0.1,
                    help="share of participants answering at random")
    ap.add_argument("--seed", type=int, default=964)
    ap.add_argument("--out", default="Data/human-responses-sim")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(CONDITIONS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    novel = [r for r in rows if r["is_filler"] == "0"]
    checks = [r for r in rows if r["is_filler"] == "1"]
    assert len(novel) == N_NOVEL and len(checks) == N_CHECKS

    item_effect = {r["item_id"]: rng.gauss(0, args.item_sd) for r in novel}
    fieldnames = list(rows[0].keys()) + [
        "alpha_side", "mouse.clicked_name", "mouse.time", "mouse.x",
        "trials.thisN", "participant", "session", "date", "expName",
    ]

    n_bad = 0
    for p in range(args.n):
        pid = "%06d" % rng.randrange(1000000)
        p_effect = rng.gauss(0, args.participant_sd)
        inattentive = rng.random() < args.inattentive
        n_bad += inattentive

        trial_rows = rng.sample(novel, N_PER_PARTICIPANT) + list(checks)
        rng.shuffle(trial_rows)

        # balanced side assignment, as the experiment does
        sides = [i % 2 == 0 for i in range(len(trial_rows))]
        rng.shuffle(sides)

        out_rows = []
        for i, (r, alpha_left) in enumerate(zip(trial_rows, sides)):
            if inattentive:
                p_alpha = 0.5
            elif r["is_filler"] == "1":
                # attentive participants know the frozen binomials
                p_alpha = 0.95 if r["expected"] == "Alpha" else 0.05
            else:
                z = (args.beta * (abs_pref(r) - 0.5)
                     + item_effect[r["item_id"]] + p_effect)
                p_alpha = 1.0 / (1.0 + math.exp(-z))

            chose_alpha = rng.random() < p_alpha
            clicked = "text_10" if chose_alpha else "text_11"

            # mimic a mind-change on a few trials: the LAST entry is the answer
            if rng.random() < 0.05:
                other = "text_11" if chose_alpha else "text_10"
                seq = [other, clicked, "text_9"]
            else:
                seq = [clicked, "text_9"]

            row = dict(r)
            row.update({
                "alpha_side": "left" if alpha_left else "right",
                "mouse.clicked_name": "[" + ", ".join("'%s'" % s for s in seq) + "]",
                "mouse.time": "[%.3f]" % rng.uniform(0.6, 4.0),
                "mouse.x": "%.3f" % (-0.4 if alpha_left == chose_alpha else 0.4),
                "trials.thisN": str(i),
                "participant": pid,
                "session": "001",
                "date": "2026-01-01_00h00.00.000",
                "expName": "novel-binom-judgements",
            })
            out_rows.append(row)

        # a leading instruction row with no item, as Pavlovia writes
        blank = {k: "" for k in fieldnames}
        blank.update({"participant": pid, "session": "001",
                      "expName": "novel-binom-judgements"})

        path = out_dir / ("%s_novel-binom-judgements_sim.csv" % pid)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerow(blank)
            w.writerows(out_rows)

    print("wrote %d participant files to %s" % (args.n, out_dir))
    print("  %d trials each (%d novel + %d checks)"
          % (N_PER_PARTICIPANT + N_CHECKS, N_PER_PARTICIPANT, N_CHECKS))
    print("  %d simulated as inattentive (should be excluded by the analysis)" % n_bad)
    print()
    print("  true AbsPref coefficient = %.2f" % args.beta)
    print("  point DATA_DIR in Analysis/human_prefs_analysis.Rmd at '../%s'"
          % args.out)


if __name__ == "__main__":
    main()
