# novel-binom-prefs

Ordering preferences for 444 novel (nonce) binomials, measured in people and in
language models over the same items.

Humans choose between the two orderings in a 2AFC task run on Pavlovia. Models
are scored by the log-odds of one ordering against the other. Both sides are then
fit with the same specification, so the constraint weights are comparable.

## Layout

```
Analysis/     R: loaders, constraint definitions, the analysis notebooks
Data/         conditions file, novelty screens, bigram controls, responses, LLM scores
Scripts/      Python: LLM scoring, novelty screens, bigram counts, experiment build
Writeup/      Quarto paper (ACL format)
Experiment/   the PsychoPy experiment (NOT in this repo, see below)
```

## The item set

**444 novel binomials.** An item is novel only if the phrase itself is
unattested: four surface forms (`w1 and w2`, `W1 and w2`, `w2 and w1`,
`W2 and w1`) must all return zero.

Novelty is **corpus-relative**, and the screen is run against every corpus a
model in the set was trained on:

| Corpus | Models | Novel | Screen |
| --- | --- | --- | --- |
| Dolma v1.7 | OLMo ×3 | 458/458 | `check_dolma.py`, reproduced by `check_novelty_corpora.py` |
| The Pile | Pythia ×7 | 444/458 | `check_novelty_corpora.py` |
| BabyLM | hand-trained ×3 | 458/458 | local trigram table |

The 14 items attested in the Pile (counts of 1–2 in 300B tokens) are dropped
**for every model**, not only for Pythia, so the item set is identical
everywhere. `drop_non_novel()` in `constraints.R` applies this at load time, so
the exclusion stays visible and reversible rather than being baked into the
conditions file.

GPT-2's WebText is not indexed and Qwen3/Llama/Gemma/Mistral corpora are
undisclosed, so for those ten models novelty rests on absence from the three
corpora above rather than on their own training data.

## Item ids: why `item_XXX` and not `nov_XXX`

Ids were renamed because **PsychoJS coerced the old ones into dates at
runtime**. Its CSV reader parses date-like strings, and `nov_` is a month
abbreviation, so `nov_076` was read as November 1976 and written to the data as
Excel serial `28065`. Only two-digit ids were affected (the leading zero is
stripped, so `nov_042` parses as a year while `nov_123` does not): 58 of 89
two-digit items were coerced, against 0 of 370 three-digit ones. It is also
**not deterministic** — 20 items were coerced for one participant and not
another, so it depends on the viewer's environment.

`recover_item_id()` in `load_human_data.R` rebuilds the id by joining on
`(word1, word2)`, which is always recorded correctly. It stays in permanently:
data collected before the rename still carries the old ids, and without it those
trials fail to join and vanish silently.

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
| `check_dolma*.py` | the original novelty screen against Dolma |
| `check_novelty_corpora.py` | the same screen against any infini-gram index |
| `compute_bigram_logodds.py` | component-bigram counts and log-odds, per corpus |

`simulate_human_data.py` exists so the analysis can be exercised before real
participants: responses are drawn from a known `--beta`, so the notebook should
recover roughly that value.

## Corpus bigram control

The trigram `word1 and word2` is zero for every item by construction, so it
carries no information. The surface-statistics control is built from the
component bigrams, which are not zero:

```
bigram_logodds = log C(w1 and) + log C(and w2) - log C(w2 and) - log C(and w1)
```

This is the first-order Markov log-odds of the two orderings; the unigram terms
cancel between them. Counts multiply (add in logs) because both transitions are
needed to build the phrase, so a zero in either position should collapse the
estimate, which a sum would not do. A 0.5 pseudocount applies.

Because it is a contrast, multiplying every count by a constant cancels: corpus
**size** is irrelevant, composition is not. So the control is matched to each
model's training corpus (`BIGRAM_CORPUS_BY_FAMILY` in `constraints.R`). That
matters empirically: Dolma and the Pile agree at r = .95 over these items, both
being web text, while BabyLM agrees with them only at r = .64.

`compute_bigram_logodds.py` writes one file per corpus with the four raw counts
alongside the derived value, so the smoothing choice stays revisable without
re-querying.

## Predictor scaling

| Variable | Treatment |
| --- | --- |
| `AbsPref_c` | `AbsPref - 0.5`, not scaled |
| 9 varying constraints | raw |
| `bigram_logodds_z` | scaled, not centred |
| `preference_z` | scaled within model, not centred |
| `trial_num_z` | centred **and** scaled |
| `alpha_side` | factor, `contr.sum` |

Everything with a meaningful zero keeps it: the constraints are signed
differences where 0 means the two words are equal, and the log-odds quantities
are 0 when neither ordering is favoured. Mean-centring would move the origin
onto the average of whichever items are in the set. `trial_num` is the
exception, having no meaningful zero (there is no trial 0).

Note `scale(x, center = FALSE)` divides by the **root mean square**, not the SD,
so scale-only uses `x / sd(x)` explicitly.

## Experiment changes

The experiment on Pavlovia (`gitlab.pavlovia.org/znhoughton/novel-binom-judgements`):

- **Side counterbalancing.** The alphabetical ordering used to be pinned to the
  left on every trial, confounding side bias with the alphabetical contrast.
  It is now assigned in balanced pairs and logged as `alpha_side`, with the
  computed offset logged as `alpha_x`.
- **Layout.** Text size and offset derive from the window aspect so a constant
  0.18 gap sits between the options. At the old fixed +/-0.3 the longest items
  (28 characters) overlapped at the midline.
- **Item ids** renamed, see above.

Pilot participants who ran before counterbalancing are excluded
(`PILOT_PARTICIPANTS`); keeping them would force the side term out of the model
for everyone, since their `alpha_side` is NA.

## Analyses

Four specifications per side, each fit with and without the bigram control:

| | LLM (per model, Gaussian) | Human (bernoulli) |
| --- | --- | --- |
| Abstract preference | `preference ~ AbsPref_c` | `chose_alpha ~ AbsPref_c` |
| Individual constraints | `preference ~ 9 constraints` | `chose_alpha ~ 9 constraints` |

plus, once human data is in, 23 separate logistic fits of human choice on each
model's preference. Separate rather than pooled: a pooled fit would repeat each
human response once per model, pseudoreplicating it 23-fold.

Random effects follow the rule that a predictor takes a slope only for a
grouping factor within which it varies. Constraints are item properties, so they
can take by-participant slopes but never by-item ones; side and trial position
vary within both.

`MIN_PARTICIPANTS_FOR_FIT` guards the human chunks so the notebook renders
before data collection completes.

## Writeup

`Writeup/writeup.qmd`, ACL format via the `_extensions/acl` extension. Render
with `quarto render --to acl-pdf`; plain `--to pdf` silently bypasses the
extension.
