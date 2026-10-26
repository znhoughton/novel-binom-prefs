## prepare_human_data.R -------------------------------------------------------
##
## Turns the raw Pavlovia exports into the de-identified files that the models
## and the writeup actually read.
##
## The raw exports carry SONA participant IDs, which are institution-linked and
## re-identifiable by anyone with SONA access, so they are kept out of the
## repository (see .gitignore). This script applies the cleaning and exclusions
## in load_human_data.R, relabels every participant with an arbitrary code, and
## writes three CSVs that contain no identifier of any kind:
##
##   Data/human-trials.csv        one row per retained novel-item trial; this is
##                                the exact input the brms models are fit on
##   Data/human-participants.csv  one row per participant seen, with the
##                                attention-check counts and exclusion status
##   Data/human-checks.csv        the scored attention-check trials
##
## Run from Analysis/ after adding or replacing anything in
## Data/human-responses/. The outputs are committed; the inputs are not.
##
## The relabelling is seeded, so re-running on the same set of participants
## reproduces the same codes and the committed CSVs do not churn. It is a
## one-way map: no crosswalk is written, here or anywhere else in the repo. The
## codes are assigned in shuffled order rather than by arrival or by ID, so a
## code carries no information about who the participant was or when they ran.

suppressPackageStartupMessages({
  library(tidyverse)
})

source("load_human_data.R")

RAW_DIR  <- "../Data/human-responses"
OUT_DIR  <- "../Data"
SEED     <- 964

set.seed(SEED)

d <- load_human_data(RAW_DIR)

## --- the pseudonym map ------------------------------------------------------
## Sort first so the input to sample() does not depend on file order, then
## shuffle, so P01 is not the first participant to run.
ids  <- sort(unique(as.character(d$participants$participant)))
code <- setNames(sprintf("P%02d", seq_along(ids)), sample(ids))

## Columns that identify a participant or a session, beyond the SONA id itself.
## `__participant` is Pavlovia's own id and is a second, independent identifier;
## `date` and `__datetime` are per-session timestamps, which single a
## participant out in a sample this size. Nothing in the analysis code reads any
## of these. Dropping them here is what makes the relabelling worth anything:
## renaming `participant` while shipping `__participant` would disclose exactly
## what it set out to protect.
IDENTIFYING <- c("__participant", "__session", "__datetime", "__experimentName",
                 "date", "session", "expName", "OS", "psychopyVersion")

relabel <- function(df) {
  df %>%
    mutate(participant = factor(code[as.character(participant)],
                                levels = sprintf("P%02d", seq_along(ids)))) %>%
    select(-any_of(IDENTIFYING))
}

if (anyNA(code[as.character(d$trials$participant)])) {
  stop("a trial carries a participant that is not in the participant summary")
}

trials <- relabel(d$trials)
checks <- relabel(d$checks)

## Carry the exclusion reason through, so load_human_prepared() can rebuild the
## `excluded` table without needing the pilot IDs.
parts <- d$participants %>%
  left_join(d$excluded %>% select(participant, reason), by = "participant") %>%
  relabel() %>%
  arrange(participant)

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
write_csv(trials, file.path(OUT_DIR, "human-trials.csv"))
write_csv(checks, file.path(OUT_DIR, "human-checks.csv"))
write_csv(parts,  file.path(OUT_DIR, "human-participants.csv"))

## --- make sure nothing identifying survived ---------------------------------
## Checked on the written files rather than on the data frames, so what is
## verified is what would actually be committed. The test is per cell and exact:
## a substring test flags any six-digit id that happens to fall inside a
## coefficient (AbsPref_c of -0.1234560 contains "123456"), which is noise
## rather than disclosure.
for (nm in c("human-trials.csv", "human-checks.csv", "human-participants.csv")) {
  back <- read_csv(file.path(OUT_DIR, nm), show_col_types = FALSE,
                   progress = FALSE)

  left <- names(back)[names(back) %in% IDENTIFYING]
  if (length(left)) {
    stop(nm, " still carries identifying column(s): ",
         paste(left, collapse = ", "), ". Not safe to publish.")
  }

  bad <- names(back)[vapply(back, function(v)
    any(as.character(v) %in% ids, na.rm = TRUE), logical(1))]
  if (length(bad)) {
    stop(nm, " still has raw participant id(s) in column(s): ",
         paste(bad, collapse = ", "), ". Not safe to publish.")
  }

  if ("participant" %in% names(back) &&
      !all(grepl("^P[0-9]{2}$", as.character(back$participant)))) {
    stop(nm, " has participant values that are not relabelled codes.")
  }
}

message("wrote human-trials.csv: ", nrow(trials), " trials, ",
        n_distinct(trials$participant), " retained participant(s)")
message("wrote human-participants.csv: ", nrow(parts), " participant(s), ",
        sum(!parts$keep), " excluded")
message("wrote human-checks.csv: ", nrow(checks), " attention-check trial(s)")
message("no raw participant ids appear in any of the three files")
