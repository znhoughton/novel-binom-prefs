# novel-binom-prefs

Ordering preferences for 458 novel (nonce) binomials, measured in people and in
language models over the same items.

Humans choose between the two orderings in a 2AFC task run on Pavlovia. Models
are scored by the log-odds of one ordering against the other. Both sides are then
fit with the same specification, so the constraint weights are comparable.

## Layout

```
Analysis/     R: loaders, constraint definitions, the two analysis notebooks
Data/         conditions file, human responses, per-model LLM scores
Scripts/      Python: LLM scoring, experiment build, data simulation
Experiment/   the PsychoPy experiment (NOT in this repo, see below)
```

`Experiment/` is its own repository, synced to Pavlovia at
`gitlab.pavlovia.org/znhoughton/novel-binom-judgements`, so it is gitignored
here. `Data/novel_binoms_conditions.csv` is a mirror of its conditions file,
which lets this repo stand alone; both the Python and R loaders prefer
`Experiment/` when it is checked out and fall back to the mirror otherwise.
If you edit the conditions, re-copy the mirror.

## Scoring LLM ordering preferences

For each item the model scores both orderings behind each of 51 prompt prefixes:

```
preference = logP(prompt + "word1 and word2") - logP(prompt + "word2 and word1")
```

so `preference > 0` means the model prefers the alphabetical order, matching
`alpha_answer == 1` on the human side. The prefix is identical in both strings
and cancels in the difference. Sequence log-probabilities are not length
normalised, for consistency with the earlier LLM-Ordering-Preferences work.

```bash
python Scripts/score_llm_prefs.py --dry-run          # what is done, what is left
python Scripts/score_llm_prefs.py --verify           # do the repo ids resolve
python Scripts/score_llm_prefs.py                    # everything outstanding
python Scripts/score_llm_prefs.py --models gpt2 pythia-160m
```

Base (non-instruct) models only: instruction tuning shifts sequence
probabilities in ways that confound a scale comparison. One CSV per model in
`Data/llm-prefs/`, written only when a model finishes all 51 prompts. Reruns
skip completed models and top up partial ones.

On a multi-GPU box, `--workers-per-gpu 1` shards models across cards. Set
`BINOM_HF_CACHE` to choose where weights download. Avoid `--load-4bit` unless a
model will not otherwise fit: quantisation noise lands directly in the log-odds
difference being measured.

## Analysis

```
Analysis/constraints.R          Morgan & Levy weights and AbsPref, shared
Analysis/load_human_data.R      Pavlovia responses -> scored trials
Analysis/load_llm_data.R        per-model CSVs -> scored items
Analysis/human_prefs_analysis.Rmd
Analysis/llm_prefs_analysis.Rmd
```

Run from `Analysis/`; the loaders use relative paths. Both notebooks set
`options(contrasts = c("contr.sum", "contr.sum"))` and seed 964.

Two details worth knowing before touching the human loader:

- Pavlovia's database export writes a BOM at the **start of every line**, not
  just the file. The first column holds a per-click list, so the BOM sits before
  its opening quote, the quoting is void, and every later column shifts right by
  (clicks - 1) with no parser error. `read_pavlovia_csv()` strips the BOMs; the
  check that alignment is restored is `Alpha == word1 + " and " + word2`.
- `alpha_side` is NA for anything collected before counterbalancing was added to
  the experiment, so it enters the models only when it was logged and varies.
  Otherwise every one of those rows would be dropped silently.

## Scripts

| script | what it does |
| --- | --- |
| `score_llm_prefs.py` | LLM ordering preferences, one CSV per model |
| `build_experiment_conditions.py` | builds the conditions file from the item set |
| `simulate_human_data.py` | fake Pavlovia output with a known AbsPref effect |
| `test_experiment.py`, `test_shipped_js.py` | checks on the built experiment |
| `annotate_constraints.py`, `make_power_list.py` | item coding helpers |
| `check_dolma*.py` | novelty checks against Dolma |

`simulate_human_data.py` exists so the analysis can be exercised before real
participants: responses are drawn from a known `--beta`, so the notebook should
recover roughly that value.
