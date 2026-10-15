#!/usr/bin/env python3
"""
test_shipped_js.py
------------------
Runs the experiment's ACTUAL shipped JavaScript, rather than a Python port of it.

    python Scripts/test_shipped_js.py

test_experiment.py re-implements the logic in Python, which would not catch a bug
introduced by editing the .js by hand. This script instead extracts two blocks
verbatim out of novel-binom-judgements.js and executes them on Windows' JScript
engine (cscript //E:JScript). Both are plain ES5, so they run unmodified:

  - the Begin Experiment block (row selection + side counterbalancing)
  - the Begin Routine block  (per-trial positioning + alpha_side logging)

Around them it reconstructs the rest of the PsychoJS pipeline:

  isInt / selectFromArray   transliterated to ES5 from psychojs-2024.2.4.iife.js
                            (the library uses arrow functions and let, which
                            JScript cannot parse, so control flow is preserved
                            but the syntax is rewritten)
  shuffle                   likewise; what Method.RANDOM applies to the trial
                            list, as a permutation
  text_10 / text_11         stubs recording setPos calls
  psychoJS.experiment       stub recording addData calls

So the two blocks are the real shipped code; the scaffolding is faithful ES5.
"""

import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "Experiment"
JS = EXP / "novel-binom-judgements.js"
CONDITIONS = EXP / "novel_binoms_conditions.csv"
N_SESSIONS = 500

EXPECTED_TRIALS = 102

BLOCK_RE = re.compile(
    r"// --- Row selection -+\n.*?console\.log\('trials this session: ' \+ subset_size\);",
    re.DOTALL,
)
ROUTINE_RE = re.compile(
    r"// Which side the alphabetical order takes this trial.*?"
    r"progress_val \+= 1\.0 / subset_size;",
    re.DOTALL,
)

HARNESS = r"""
// ---- ES5 stand-ins for the browser bits the blocks touch ---------------
var lastLog = "";
var console = { log: function (s) { lastLog = String(s); } };

var sideLog = [];
var psychoJS = { experiment: { addData: function (k, v) {
    if (k === "alpha_side") { sideLog.push(v); }
} } };

function makeText() {
    return { pos: null, setPos: function (p) { this.pos = p; } };
}
var text_10 = makeText(), text_11 = makeText();
var progress_val = 0;

// ---- transliterated from psychojs-2024.2.4.iife.js ---------------------
function isArray(a) { return Object.prototype.toString.call(a) === "[object Array]"; }

function isInt(obj) {
    if (isNaN(obj)) { return false; }
    var x = parseFloat(obj);
    return (x | 0) === x;
}

function selectFromArray(array, selection) {
    if (isInt(selection)) { return [array[parseInt(selection, 10)]]; }
    if (isArray(selection)) {
        var out = [];
        for (var k = 0; k < selection.length; k++) { out.push(array[selection[k]]); }
        return out;
    }
    if (typeof selection === "string") {
        if (selection.indexOf(",") > -1) {
            var parts = selection.split(","), asArray = [];
            for (var m = 0; m < parts.length; m++) { asArray.push(parseInt(parts[m], 10)); }
            return selectFromArray(array, asArray);
        }
    }
    throw new Error("unhandled selection type");
}

function shuffle(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var t = array[i]; array[i] = array[j]; array[j] = t;
    }
    return array;
}

// ---- globals the blocks assign, as PsychoPy declares them --------------
var n_novel, n_checks, n_per_participant, allIndices, randomIndices;
var new_rows, subset_size, alphaOnLeft, trialCounter, alphaLeftThisTrial;

function runSelectionBlock() {
__BLOCK__
}

function runRoutineBlock() {
__ROUTINE__
}

// ---- the real conditions-file rows, keyed by binomial ------------------
var PAIR_KEYS = __PAIRS__;
var IS_CHECK  = __CHECKS__;

// ---- drive it ----------------------------------------------------------
var nSessions = __N__, expectedTrials = __EXPECTED__;
var badSize = 0, dupIndex = 0, dupBinom = 0, outOfRange = 0, badCheckCount = 0;
var sideSplitBad = 0, posBad = 0, sideMismatch = 0, progressBad = 0, sideUndef = 0;
var sizes = {}, everSeen = {}, totalLeft = 0, totalTrials = 0;

for (var s = 0; s < nSessions; s++) {
    runSelectionBlock();

    // what PsychoJS does: split the selection string and map it over the rows
    var selected = selectFromArray(PAIR_KEYS, new_rows);
    // what Method.RANDOM + nReps 1 does: permute, so each selected row shows once
    var order = [];
    for (var i = 0; i < selected.length; i++) { order.push(i); }
    shuffle(order);

    var shown = [];
    for (var i = 0; i < order.length; i++) { shown.push(selected[order[i]]); }

    sizes[shown.length] = 1;
    if (shown.length !== expectedTrials) { badSize++; }

    var seenIdx = {}, seenPair = {}, nChecks = 0;
    var idxParts = new_rows.split(",");
    for (var i = 0; i < idxParts.length; i++) {
        var ix = parseInt(idxParts[i], 10);
        if (ix < 0 || ix >= PAIR_KEYS.length || isNaN(ix)) { outOfRange++; }
        if (seenIdx[ix]) { dupIndex++; } else { seenIdx[ix] = 1; }
        if (IS_CHECK[ix]) { nChecks++; }
    }
    if (nChecks !== 10) { badCheckCount++; }

    for (var i = 0; i < shown.length; i++) {
        var key = shown[i];
        if (key === undefined) { outOfRange++; continue; }
        if (seenPair[key]) { dupBinom++; } else { seenPair[key] = 1; }
        everSeen[key] = 1;
    }

    // --- per-trial code: run the real Begin Routine block once per trial ---
    sideLog = [];
    progress_val = 0;
    var nLeft = 0;
    for (var i = 0; i < subset_size; i++) {
        runRoutineBlock();
        if (alphaLeftThisTrial === undefined) { sideUndef++; continue; }

        var p10 = text_10.pos, p11 = text_11.pos;
        // exactly one on each side, at the expected offsets
        if (!(p10[0] === -p11[0] && Math.abs(p10[0]) === 0.4 &&
              p10[1] === 0 && p11[1] === 0)) { posBad++; }
        // the logged side must match where text_10 (the alphabetical order) went
        var loggedLeft = (sideLog[sideLog.length - 1] === "left");
        if (loggedLeft !== (p10[0] < 0)) { sideMismatch++; }
        if (loggedLeft) { nLeft++; totalLeft++; }
        totalTrials++;
    }
    if (nLeft !== Math.floor(expectedTrials / 2)) { sideSplitBad++; }
    // the progress bar must land exactly on 1.0 after the last trial
    if (Math.abs(progress_val - 1.0) > 1e-9) { progressBad++; }
}

var nSeen = 0;
for (var k in everSeen) { if (everSeen.hasOwnProperty(k)) { nSeen++; } }
var sizeList = [];
for (var k in sizes) { if (sizes.hasOwnProperty(k)) { sizeList.push(k); } }

WScript.Echo("sessions="      + nSessions);
WScript.Echo("sizes="         + sizeList.join("/"));
WScript.Echo("badSize="       + badSize);
WScript.Echo("dupIndex="      + dupIndex);
WScript.Echo("dupBinom="      + dupBinom);
WScript.Echo("outOfRange="    + outOfRange);
WScript.Echo("badCheckCount=" + badCheckCount);
WScript.Echo("distinctSeen="  + nSeen);
WScript.Echo("sideUndef="     + sideUndef);
WScript.Echo("posBad="        + posBad);
WScript.Echo("sideMismatch="  + sideMismatch);
WScript.Echo("sideSplitBad="  + sideSplitBad);
WScript.Echo("progressBad="   + progressBad);
WScript.Echo("pctAlphaLeft="  + (100 * totalLeft / totalTrials).toFixed(2));
WScript.Echo("lastLog="       + lastLog);
WScript.Echo("subset_size="   + subset_size);
"""


def main():
    js_text = JS.read_text(encoding="utf-8")
    block = BLOCK_RE.search(js_text)
    routine = ROUTINE_RE.search(js_text)
    if not block:
        sys.exit("could not find the row-selection block in %s" % JS.name)
    if not routine:
        sys.exit("could not find the per-trial routine block in %s" % JS.name)
    print("extracted from %s: selection block %d lines, routine block %d lines"
          % (JS.name, len(block.group(0).splitlines()),
             len(routine.group(0).splitlines())))

    with open(CONDITIONS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    pairs = ['"%s"' % "|".join(sorted((r["word1"], r["word2"]))) for r in rows]
    checks = ["1" if r["is_filler"] == "1" else "0" for r in rows]

    script = (HARNESS
              .replace("__BLOCK__", block.group(0))
              .replace("__ROUTINE__", routine.group(0))
              .replace("__PAIRS__", "[" + ",".join(pairs) + "]")
              .replace("__CHECKS__", "[" + ",".join(checks) + "]")
              .replace("__EXPECTED__", str(EXPECTED_TRIALS))
              .replace("__N__", str(N_SESSIONS)))

    tmp = Path(tempfile.gettempdir()) / "novel_binom_selection_test.js"
    tmp.write_text(script, encoding="ascii")

    proc = subprocess.run(["cscript", "//Nologo", "//E:JScript", str(tmp)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        sys.exit("cscript failed:\n" + proc.stderr)

    out = dict(ln.split("=", 1) for ln in proc.stdout.strip().splitlines() if "=" in ln)
    print()
    for k, v in out.items():
        print("  %-14s %s" % (k, v))

    half = EXPECTED_TRIALS // 2
    expectations = [
        ("sizes", str(EXPECTED_TRIALS), "every session is exactly %d trials" % EXPECTED_TRIALS),
        ("badSize", "0", "no session has the wrong trial count"),
        ("dupIndex", "0", "no session selects a row index twice"),
        ("dupBinom", "0", "NO PARTICIPANT SEES A BINOMIAL MORE THAN ONCE"),
        ("outOfRange", "0", "every index resolves to a real row"),
        ("badCheckCount", "0", "every session contains all 10 attention checks"),
        ("distinctSeen", str(len(rows)), "all %d items reachable across sessions" % len(rows)),
        ("sideUndef", "0", "no trial reads past the end of the side list"),
        ("posBad", "0", "the two orders always sit on opposite sides at +/-0.4"),
        ("sideMismatch", "0", "logged alpha_side always matches the actual position"),
        ("sideSplitBad", "0",
         "every participant gets exactly %d alphabetical-left trials" % half),
        ("progressBad", "0", "progress bar lands exactly on 1.0 after the last trial"),
        ("subset_size", str(EXPECTED_TRIALS), "subset_size matches the trial count"),
        ("lastLog", "trials this session: %d" % EXPECTED_TRIALS,
         "the block logs the right count"),
    ]

    print()
    failed = []
    for key, want, label in expectations:
        got = out.get(key, "<missing>")
        ok = got == want
        print("  %s  %s" % ("ok  " if ok else "FAIL", label))
        if not ok:
            print("        expected %r, got %r" % (want, got))
            failed.append(label)

    pct = float(out.get("pctAlphaLeft", "0"))
    ok = abs(pct - 50.0) < 1.0
    print("  %s  alphabetical order appears left %.2f%% of the time"
          % ("ok  " if ok else "FAIL", pct))
    if not ok:
        failed.append("side balance")

    print("\n" + "=" * 60)
    if failed:
        sys.exit("FAILED %d check(s) running the shipped JS" % len(failed))
    n_novel = sum(1 for r in rows if r["is_filler"] == "0")
    print("PASSED: shipped JavaScript behaves correctly over %d sessions" % N_SESSIONS)
    print("        (%d novel items + 10 checks; %d shown per participant)"
          % (n_novel, EXPECTED_TRIALS))


if __name__ == "__main__":
    main()
