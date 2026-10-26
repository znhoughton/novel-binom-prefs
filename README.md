# novel-binom-prefs

Ordering preferences for 444 novel (nonce) binomials, measured in people and in
language models over the same items.

Humans choose between the two orderings in a 2AFC task run on Pavlovia. Models
are scored by the log-odds of one ordering against the other. Both sides are then
fit with the same specification, so the constraint weights are comparable.

## Layout

```
Analysis/     R: loaders, constraint definitions, model fitting, the notebooks
Data/          conditions file, novelty screens, bigram controls, LLM scores,
               and the de-identified human trials
Results/       tidy posterior summaries the paper renders from
Scripts/       Python: LLM scoring, novelty screens, bigram counts, experiment build
Writeup/       Quarto paper (ACL format)
Experiment/    the PsychoPy experiment (NOT in this repo, see below)
Data/human-responses/   the raw Pavlovia exports (NOT in this repo, see below)
```

## The item set

**444 novel binomials.** An item is novel only if the phrase itself is
unattested: four surface forms (`w1 and w2`, `W1 and w2`, `w2 and w1`,
`W2 and w1`) must all return zero.

This surface screen is what selects the items. `check_novelty_corpora.py
--variants lemma` additionally checks each binomial with its nouns in the other
number, which is a **stricter** criterion and not one the item set was filtered
on: of the 444, all 444 are unattested in BabyLM on that criterion too, 421 in
the Pile, and 383 in Dolma. The per-item results are in
`Data/novelty_lemma_{dolma,pile,babylm}.csv`.

The lemma check is restricted to the lowercase and capitalised-first spellings.
All-caps and title-case queries return identical counts for both orderings,
which means the index is not discriminating between them, so they cannot
support a novelty claim either way.

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

GPT-2's WebText is not indexed and Qwen3's corpus is undisclosed, so for those
seven models novelty rests on absence from the three corpora above rather than
on their own training data. The single-model families (Llama, Gemma, Mistral)
were dropped from the set: the scale comparison needs at least three sizes
within a family.

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
Analysis/load_human_data.R      Pavlovia responses -> scored trials, or the
                                de-identified CSVs when the raw are absent
Analysis/load_llm_data.R        per-model CSVs -> scored items, + checkpoints
Analysis/prepare_human_data.R   raw exports -> the de-identified CSVs
Analysis/human_prefs_analysis.Rmd
Analysis/llm_prefs_analysis.Rmd
Analysis/fit_llm_predicts_human.R      human choice ~ each model's preference
Analysis/fit_constraint_trajectories.R per-constraint GAMs over training
Analysis/export_results.R              fitted models -> Results/*.csv
```

`export_results.R` is the seam between fitting and writing. It walks the cached
fits in `Analysis/models/` once, reduces each to the few numbers the paper
needs, and writes `Results/*.csv`; the paper then renders from those rather than
loading eighty brms objects. Re-run it after refitting anything, and commit the
CSVs it produces.

Run from `Analysis/`; the loaders use relative paths. Every script sets
`options(contrasts = c("contr.sum", "contr.sum"))` and seed 964.

### The human data is de-identified, and the raw exports are not here

The raw Pavlovia exports carry SONA participant ids, which are
institution-linked and re-identifiable by anyone with SONA access, so
`Data/human-responses/` is gitignored. What ships instead is the output of
`prepare_human_data.R`:

```
Data/human-trials.csv        3,100 retained trials; the exact brms input
Data/human-participants.csv  29 participants, with check counts and exclusions
Data/human-checks.csv        290 scored attention-check trials
```

`load_human_data()` reads the raw exports when they are present and falls back
to these otherwise, returning the same four-element list either way, so a clone
reproduces 3,100 trials over 444 items from 21 of 29 participants with no raw
data at all.

Relabelling `participant` is not by itself enough. The export also carries
`__participant`, which is Pavlovia's own independent id, and per-session `date`
and `__datetime` timestamps, which single a participant out in a sample this
size. `prepare_human_data.R` drops all of them and then re-reads the files it
wrote to check that no raw id survived, failing rather than reporting success.
The participant codes are assigned in seeded shuffled order, so they are stable
across runs and carry no information about who ran when. No crosswalk is
written anywhere in the repo.

If you add raw exports, re-run `prepare_human_data.R` and commit the three
regenerated CSVs. The pilot participants' ids live in an untracked
`Analysis/.pilot_participants`, one per line; without it `PILOT_PARTICIPANTS` is
empty and the two pilot sessions are not excluded, which only matters when
processing raw exports.

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
(`PILOT_PARTICIPANTS`, read from the untracked `Analysis/.pilot_participants`);
keeping them would force the side term out of the model for everyone, since
their `alpha_side` is NA. In the de-identified CSVs their exclusion is already
recorded in `is_pilot`, so the fallback path does not need the ids.

## Analyses

Four specifications per side, each fit with and without the bigram control:

| | LLM (per model, Gaussian) | Human (bernoulli) |
| --- | --- | --- |
| Abstract preference | `preference ~ AbsPref_c` | `chose_alpha ~ AbsPref_c` |
| Individual constraints | `preference ~ 9 constraints` | `chose_alpha ~ 9 constraints` |

plus 20 separate logistic fits of human choice on each model's preference
(`fit_llm_predicts_human.R`). Separate rather than pooled: a pooled fit would
repeat each human response once per model, pseudoreplicating it 20-fold. These
are fit at the trial level, not on item proportions: with a median of seven
judgements per binomial the reliability of an item proportion is about 0.4,
which caps any item-level correlation near 0.65 however well a model predicts.

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

It reads `Results/*.csv` and never opens a model object, so it renders without
the fits. Two things to know before editing it:

- Inline `` `r ` `` in a paragraph is evaluated in document order, so anything a
  paragraph reports has to be computed in a chunk **above** it. Each section
  therefore opens with an `include: false` setup chunk.
- `acl.sty` defines `thebibliography` with its own `\section*{References}` and
  ignores `\bibsection`, so the document must not add a `# References` heading
  of its own or the word appears twice.

## Experiment 3: preferences across training

`Scripts/score_llm_prefs_checkpoints.py` scores the same items and the same
measure at training checkpoints rather than only at the end.

```bash
python Scripts/score_llm_prefs_checkpoints.py --dry-run
python Scripts/score_llm_prefs_checkpoints.py --models pythia-1.4b
```

Checkpoints are discovered from the Hub rather than hardcoded, so a repo that
adds revisions is picked up automatically. Three naming schemes are handled:
`stepN` branches (Pythia), `stage1-stepN` branches (OLMo 3), and `step-N` tags
(the hand-trained models).

Sampling is log-spaced, because learning curves move fastest early and evenly
spaced steps would spend the budget where nothing changes. The first and last
checkpoints are always kept. `tokens_per_step` converts steps to tokens seen,
which is the x-axis worth plotting: a step is a different amount of data for
different models, so the comparison of interest is at matched tokens.

Nine models have been scored at checkpoints, but `load_checkpoint_data()`
requires `min_checkpoints = 8`, which drops `pythia-2.8b` (5 scored) and
`pythia-6.9b` (1). **Seven** models therefore reach the analysis: BabyLM x3 and
Pythia 160m/410m/1.4b at 14 checkpoints each, and OLMo-3-7B at 15, giving 99
checkpoint fits. Finishing 2.8b and 6.9b would bring them in automatically.
Each scoring is 46,716 sequences, so this is a substantial job: run it on the
H100 box rather than locally.

Two analyses run over these:

- **Abstract preference.** `Equation 2` refit separately at each checkpoint, one
  estimate per checkpoint, then a spline smooth across them for the figure.
  Checkpoints below 10^6 tokens are excluded from the smooth: an untrained
  model's coefficient is not identifiable, and for three Pythia models step 0
  and step 1 carry identical scores, which would double-weight the left edge.
- **Individual constraints.** `fit_constraint_trajectories.R`, one
  varying-coefficient GAM per model via `bam()`, `s(log_tokens, by = <constraint>)`.
  With a numeric `by` the smooth multiplies the covariate, so the fitted smooth
  *is* that constraint's coefficient as a function of training. The constraints
  are deliberately not also entered as main effects: a numeric `by` smooth is
  not centred, so adding them would double-count. This replaced a per-checkpoint
  Bayesian version that would have taken about eight days; the GAMs take under a
  minute each and agree with per-checkpoint OLS at r = 0.82 to 0.97 for eight of
  the nine constraints.
