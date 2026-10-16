## load_human_data.R ---------------------------------------------------------
##
## Reads the Pavlovia output of the novel-binomial ordering study, scores each
## trial, computes abstract preference (AbsPref), and applies the attention-check
## exclusion. Sourced by human_prefs_analysis.Rmd; also usable on its own.
##
## Ported from CoNLL-2025-paper/Scripts/human-prefs-analysis.Rmd. The modelling
## there is reused closely, but the data loading is rewritten: that study ran on
## Qualtrics (one row per participant, Q-columns, answers as literal strings)
## whereas this one runs on PsychoPy/Pavlovia (one row per trial, with every
## conditions-file column already attached).

suppressPackageStartupMessages({
  library(tidyverse)
})

## Morgan & Levy constraint weights and AbsPref now live in constraints.R,
## so the LLM analysis scores items with exactly the same definition.
source("constraints.R")


## The two pilot participants (2026-10-14). They ran the build in which the
## alphabetical order was pinned to the left on every trial, before side
## counterbalancing was added, so `alpha_side` is NA for them and any
## side-related effect is unestimable from their data. They are dropped rather
## than pooled, since keeping them would force the side term out of the model
## for everyone. Pass exclude = character(0) to analyse the pilot on its own.
PILOT_PARTICIPANTS <- character(0)  # ids moved to untracked Analysis/.pilot_participants


#' Pull the chosen order out of PsychoPy's mouse.clicked_name field.
#'
#' The field is a list that accumulates EVERY click in the trial, not a single
#' response: a participant who selects one order, switches, then submits yields
#' "['text_10', 'text_11', 'text_9']". Real data from the previous study contains
#' exactly this pattern, plus "[None]" rows for clicks that hit nothing.
#'
#' The response is therefore the LAST text_10/text_11 entry. Taking the first
#' would silently record the opposite answer for anyone who changed their mind.
#'
#' text_10 always holds the alphabetical order, text_11 the reverse; which side
#' they appear on is counterbalanced separately and logged as `alpha_side`.
last_choice <- function(x) {
  hits <- str_extract_all(as.character(x), "text_1[01]")
  vapply(hits, function(h) if (length(h) == 0) NA_character_ else h[[length(h)]],
         character(1))
}


#' Read one Pavlovia CSV, repairing the database export's quoting.
#'
#' Pavlovia's Database_AllParticipants export writes a BOM at the start of EVERY
#' line, not just the file. The first column is `mouse.x`, which holds a list
#' like "[-0.298,-0.002]" with one element per click. Because the BOM sits before
#' the opening quote, the field is not recognised as quoted, so the list's
#' internal commas split it into extra fields and every later column shifts right
#' by (clicks - 1). Rows end up with 64 to 70 fields against a 64-name header,
#' and values land under the wrong names without any parser error.
#'
#' Stripping the BOMs restores the alignment. On the pilot export this takes all
#' 334 rows to exactly 64 fields, and `Alpha == word1 + " and " + word2` then
#' holds for 100% of trial rows, which is the check that the columns line up.
read_pavlovia_csv <- function(path) {
  txt <- read_file(path)
  txt <- str_remove_all(txt, "﻿")

  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp), add = TRUE)
  write_file(txt, tmp)

  read_csv(tmp, show_col_types = FALSE, progress = FALSE,
           col_types = cols(.default = col_character()))
}


#' Recover the canonical item_id from the word pair.
#'
#' The Pavlovia export's `item_id` column is not reliable: on a minority of
#' trials it carries a bare number instead of the conditions file's nov_XXX /
#' check_XX id. It affects every participant (22 of 163 trials for the first
#' post-counterbalancing one) and is not a parsing artefact, since those rows
#' are otherwise perfectly aligned and `Alpha == word1 + " and " + word2` holds
#' throughout the file.
#'
#' What is always correct is the word pair, and every affected pair resolves to
#' exactly one conditions-file row. So the id is rebuilt by joining on
#' (word1, word2) rather than trusting the column. Without this the affected
#' trials fail to join against the item codings and disappear silently, which
#' at 14% of a participant's data would be a serious loss.
#'
#' @param df         trial rows carrying word1 and word2
#' @param items_csv  conditions file; resolved like load_llm_data does
recover_item_id <- function(df, items_csv = NULL) {
  if (is.null(items_csv)) {
    candidates <- c("../Experiment/novel_binoms_conditions.csv",
                    "../Data/novel_binoms_conditions.csv")
    items_csv <- candidates[file.exists(candidates)][1]
    if (is.na(items_csv)) stop("conditions file not found")
  }
  key <- read_csv(items_csv, show_col_types = FALSE, progress = FALSE) %>%
    select(item_id_canonical = item_id, word1, word2)

  dup <- key %>% count(word1, word2) %>% filter(n > 1)
  if (nrow(dup)) {
    stop(nrow(dup), " word pair(s) map to more than one item_id in ", items_csv,
         "; the pair cannot be used as a key.")
  }

  out <- df %>%
    left_join(key, by = c("word1", "word2"))

  unmatched <- sum(is.na(out$item_id_canonical))
  if (unmatched) {
    stop(unmatched, " trial(s) have a word pair that is not in ", items_csv)
  }

  n_fixed <- sum(out$item_id != out$item_id_canonical, na.rm = TRUE)
  if (n_fixed) {
    message("recovered item_id from the word pair for ", n_fixed, " of ",
            nrow(out), " trials (export carried a numeric id)")
  }

  out %>%
    mutate(item_id = item_id_canonical) %>%
    select(-item_id_canonical)
}


#' Load and score the pilot/experiment responses.
#'
#' @param path          either a Pavlovia database export (a single .csv) or a
#'                      directory of per-participant .csv files. Both are read
#'                      the same way; the database export is the usual input.
#' @param min_checks    attention checks a participant must pass to be kept
#' @return list(trials = scored novel-item trials for modelling,
#'              checks = scored attention-check trials,
#'              participants = per-participant summary,
#'              excluded = ids dropped, with reasons)
load_human_data <- function(path = "../Data/human-responses",
                            min_checks = 8,
                            exclude = PILOT_PARTICIPANTS) {

  if (dir.exists(path)) {
    files <- list.files(path, pattern = "\\.csv$", full.names = TRUE)
    if (!length(files)) stop("no .csv files found in ", path)
    message("reading ", length(files), " file(s) from ", path)
  } else if (file.exists(path)) {
    files <- path
    message("reading database export: ", basename(path))
  } else {
    stop("no such file or directory: ", path)
  }

  raw <- map_dfr(files, read_pavlovia_csv)

  ## Pavlovia writes the loop columns both bare and prefixed with the loop name,
  ## and only one of the pair is populated depending on PsychoPy version.
  ##
  ## Not named `pick`: dplyr 1.1.0 added a `pick()` verb, and inside mutate()
  ## dplyr resolves its own before a local function of the same name, so the
  ## call failed with "Can't select columns with `.`".
  pick_col <- function(df, bare) {
    prefixed <- paste0("trials.", bare)
    a <- if (bare %in% names(df)) df[[bare]] else NA_character_
    b <- if (prefixed %in% names(df)) df[[prefixed]] else NA_character_
    coalesce(na_if(as.character(a), ""), na_if(as.character(b), ""))
  }

  ## alpha_side records which side the alphabetical order was shown on. The
  ## pilot build does not log it, so the side-bias check has nothing to run on;
  ## carry it as NA rather than failing, and report it as unavailable.
  if (!"alpha_side" %in% names(raw)) {
    message("note: alpha_side is not in this data, so the side-bias check cannot be run")
    raw$alpha_side <- NA_character_
  }

  dat <- raw %>%
    mutate(
      clicked    = pick_col(., "mouse.clicked_name"),
      rt_raw     = pick_col(., "mouse.time"),
      trial_raw  = pick_col(., "trials.thisN"),
      participant = as.character(participant)
    ) %>%
    ## Instruction, pause and debrief rows carry no item, so drop them.
    filter(!is.na(item_id), item_id != "") %>%
    mutate(
      choice       = last_choice(clicked),
      alpha_answer = case_when(choice == "text_10" ~ 1L,
                               choice == "text_11" ~ 0L,
                               TRUE ~ NA_integer_),
      is_filler    = as.integer(is_filler),
      Item         = Alpha,
      ## first click time in the trial, a rough RT
      rt           = suppressWarnings(as.numeric(str_extract(rt_raw, "[0-9.]+"))),

      ## Position in the run, counted from 1. `trials.thisN` is the loop's own
      ## 0-based counter and covers novel items and attention checks alike, so
      ## it is the order the participant actually saw.
      trial_num    = suppressWarnings(as.numeric(trial_raw)) + 1,

      ## Which side the alphabetical order was drawn on. Two levels, so it goes
      ## in as a centred fixed effect (and a by-participant slope), never as a
      ## grouping factor. NA throughout for data collected before the
      ## counterbalancing was added.
      alpha_side   = factor(alpha_side, levels = c("left", "right")),
      alpha_side_c = dplyr::case_when(alpha_side == "left"  ~ -0.5,
                                      alpha_side == "right" ~  0.5,
                                      TRUE ~ NA_real_)
    )

  ## A directory is read whole, so an old export left beside a newer one that
  ## supersedes it would silently double every shared participant's trials.
  ## Each (participant, trial position) should occur exactly once.
  dupes <- dat %>%
    count(participant, trial_num) %>%
    filter(n > 1)
  if (nrow(dupes)) {
    stop(nrow(dupes), " duplicated (participant, trial) row(s) across the files in ",
         path, ". Affected participant(s): ",
         paste(unique(dupes$participant), collapse = ", "),
         ". Two exports probably cover the same session; keep only the newest.")
  }

  n_no_response <- sum(is.na(dat$alpha_answer))
  if (n_no_response > 0) {
    message("dropping ", n_no_response,
            " trial(s) with no interpretable selection")
  }
  dat <- dat %>% filter(!is.na(alpha_answer))

  ## --- attention checks ------------------------------------------------------
  checks <- dat %>%
    filter(is_filler == 1) %>%
    mutate(check_correct = as.integer(
      (expected == "Alpha"    & alpha_answer == 1) |
      (expected == "Nonalpha" & alpha_answer == 0)
    ))

  per_participant <- checks %>%
    group_by(participant) %>%
    summarise(n_checks = n(),
              n_correct = sum(check_correct),
              .groups = "drop") %>%
    mutate(
      is_pilot = participant %in% exclude,
      keep     = n_correct >= min_checks & !is_pilot
    )

  ## Pilot first, so someone who is both a pilot participant and below the
  ## attention-check threshold is reported under the reason that actually
  ## disqualifies them regardless of their accuracy.
  excluded <- per_participant %>%
    filter(!keep) %>%
    mutate(reason = dplyr::case_when(
      is_pilot ~ "pilot: ran before side counterbalancing was added",
      TRUE     ~ sprintf("passed %d/%d attention checks (threshold %d)",
                         n_correct, n_checks, min_checks)
    ))

  if (nrow(excluded)) {
    n_pilot <- sum(excluded$is_pilot)
    n_check <- nrow(excluded) - n_pilot
    message("excluding ", nrow(excluded), " of ", nrow(per_participant),
            " participants (", n_pilot, " pilot, ", n_check, " attention checks)")
  }

  keepers <- per_participant %>% filter(keep) %>% pull(participant)

  if (!length(keepers)) {
    stop("no participants left after exclusions (", nrow(per_participant),
         " seen, all dropped). If this is the pilot data on its own, pass ",
         "exclude = character(0) to keep it.")
  }

  ## --- novel items, the modelling set ---------------------------------------
  trials <- dat %>%
    filter(is_filler == 0, participant %in% keepers) %>%
    recover_item_id() %>%
    drop_non_novel() %>%
    add_abs_pref() %>%
    add_bigram_logodds() %>%
    mutate(participant = factor(participant),
           Item = factor(Item),
           ## Standardised so its coefficient is comparable with the others and
           ## the intercept sits mid-run rather than at trial 1.
           trial_num_z = as.numeric(scale(trial_num)))

  if (all(is.na(trials$alpha_side))) {
    message("note: alpha_side is NA for every trial, so side cannot be modelled here")
  } else if (any(is.na(trials$alpha_side))) {
    message("note: alpha_side is missing for ",
            sum(is.na(trials$alpha_side)), " of ", nrow(trials),
            " trials (data collected before counterbalancing was added)")
  }

  list(trials = trials,
       checks = checks,
       participants = per_participant,
       excluded = excluded)
}
