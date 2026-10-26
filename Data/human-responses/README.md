Raw Pavlovia exports go here. They are **not** committed: they carry SONA
participant ids, which are institution-linked and re-identifiable, so this
directory is gitignored apart from this note.

Drop either a Pavlovia database export (a single `.csv`) or the per-participant
`.csv` files here, then run, from `Analysis/`:

```bash
Rscript prepare_human_data.R
```

That writes the de-identified `Data/human-trials.csv`,
`Data/human-participants.csv` and `Data/human-checks.csv`, which are committed
and are what the models and the writeup read. Commit those three files; leave
everything in this directory alone.

`Analysis/load_human_data.R` reads this directory when it has data and falls
back to the de-identified CSVs when it does not, so the analyses run either
way. The `.log` and `.psydat` files Pavlovia also produces are not used.
