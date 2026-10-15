## load_llm_data.R -----------------------------------------------------------
##
## Reads the per-model CSVs written by Scripts/score_llm_prefs.py, attaches the
## constraint codings and AbsPref, and returns them in the shape the human
## analysis uses, so the two sides can be modelled and plotted side by side.
##
## The LLM analogue of a participant is a model, and the analogue of a trial is
## one (model, prompt, item) scoring. The outcome is `preference`, the log-odds
## of the alphabetical order, which is what the human bernoulli models estimate
## on the logit scale. That is the same specification used in
## LLM-Ordering-Preferences (`log_odds ~ GenPref + (GenPref | prompt_value) +
## (1 | Alpha)`), with GenPref renamed AbsPref to match this project.

suppressPackageStartupMessages({
  library(tidyverse)
})

source("constraints.R")


#' Load every per-model CSV in `data_dir`.
#'
#' @param data_dir   directory of score_llm_prefs.py output, one CSV per model
#' @param items_csv  the conditions file, the single source of truth for codings
#' @param models     optional character vector to keep only some models
#' @return list(scores = one row per (model, prompt, item),
#'              items  = one row per (model, item), averaged over prompts,
#'              coverage = prompts and items present per model)
load_llm_data <- function(data_dir  = "../Data/llm-prefs",
                          items_csv = NULL,
                          models    = NULL) {

  ## Experiment/ is its own repo (synced to Pavlovia) and is not part of this
  ## one, so Data/ carries a mirror of the conditions file. Prefer Experiment/
  ## when it is checked out, since that is the copy participants saw.
  if (is.null(items_csv)) {
    candidates <- c("../Experiment/novel_binoms_conditions.csv",
                    "../Data/novel_binoms_conditions.csv")
    items_csv <- candidates[file.exists(candidates)][1]
    if (is.na(items_csv)) {
      stop("conditions file not found at either ", paste(candidates, collapse = " or "))
    }
  }

  files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)
  if (!length(files)) stop("no .csv files found in ", data_dir)
  message("reading ", length(files), " model file(s)")

  raw <- files %>%
    map_dfr(~ read_csv(.x, show_col_types = FALSE, progress = FALSE))

  if (!is.null(models)) {
    raw <- raw %>% filter(model %in% models | model_family %in% models)
    if (!nrow(raw)) stop("no rows left after filtering on models = ",
                         paste(models, collapse = ", "))
  }

  ## Constraint codings come from the conditions file rather than from the
  ## scoring output, so a recoding only ever has to happen in one place.
  items <- read_csv(items_csv, show_col_types = FALSE, progress = FALSE) %>%
    filter(is_filler == 0) %>%
    select(item_id, Alpha, all_of(CONSTRAINTS)) %>%
    add_abs_pref()

  ## Two files can carry the same model: a renamed copy, a backup left in the
  ## directory, a half-finished rerun. Concatenating them silently double-counts
  ## every row, which survives averaging but inflates n and duplicates rows into
  ## the brms fits. Catch it here rather than in a results table.
  dupes <- raw %>%
    count(model, prompt_value, item_id) %>%
    filter(n > 1)
  if (nrow(dupes)) {
    stop(nrow(dupes), " duplicated (model, prompt, item) row(s) across the files in ",
         data_dir, ". Affected model(s): ",
         paste(unique(dupes$model), collapse = ", "),
         ". Keep one CSV per model and move backups elsewhere.")
  }

  missing_codings <- setdiff(unique(raw$item_id), items$item_id)
  if (length(missing_codings)) {
    stop(length(missing_codings), " scored item(s) are not in ", items_csv,
         ", first: ", missing_codings[[1]])
  }

  ## Continuous predictors are standardised over the 458 unique items. The human
  ## script scales over trials instead, where an item appears once per
  ## participant; with a roughly balanced design the two are near-identical, but
  ## they are not guaranteed to agree to the last decimal.
  items <- items %>%
    mutate(across(c(Freq, Len, Lapse), ~ as.numeric(scale(.x)), .names = "{.col}_z"))

  scores <- raw %>%
    left_join(items, by = "item_id", suffix = c("", ".items")) %>%
    mutate(
      p_alpha = plogis(preference),
      model   = factor(model),
      Item    = factor(Item),
      prompt_value = factor(prompt_value)
    ) %>%
    ## Log-odds magnitudes differ by an order of magnitude across models, which
    ## makes raw coefficients incomparable between them. The z-scored version is
    ## for cross-model comparison only; single-model fits use `preference`.
    group_by(model) %>%
    mutate(preference_z = as.numeric(scale(preference))) %>%
    ungroup()

  coverage <- scores %>%
    group_by(model, model_family, params) %>%
    summarise(n_prompts = n_distinct(prompt_value),
              n_items   = n_distinct(item_id),
              n_rows    = n(),
              .groups   = "drop") %>%
    arrange(params)

  incomplete <- coverage %>% filter(n_items != nrow(items))
  if (nrow(incomplete)) {
    warning("model(s) with fewer than ", nrow(items), " items: ",
            paste(incomplete$model, collapse = ", "))
  }

  ## One point per (model, item): the model's preference averaged over prompts.
  ## Averaging the log-odds rather than p(alpha) keeps the scale additive, which
  ## is what the models are fit on.
  item_level <- scores %>%
    group_by(model, model_family, params, item_id, Item) %>%
    ## sd() comes first: redefining `preference` in the same summarise() would
    ## leave sd() looking at the one-element mean.
    summarise(sd_preference = sd(preference),
              n_prompts     = n(),
              preference    = mean(preference),
              .groups       = "drop") %>%
    left_join(items, by = "item_id") %>%
    mutate(p_alpha = plogis(preference))

  list(scores = scores, items = item_level, coverage = coverage)
}


#' Join LLM item-level preferences to human item-level proportions.
#'
#' @param llm_items   the `items` element of load_llm_data()
#' @param trials      the `trials` element of load_human_data()
#' @return one row per (model, item) with both sides' preferences
join_human_llm <- function(llm_items, trials) {
  human_items <- trials %>%
    group_by(item_id) %>%
    summarise(human_p = mean(alpha_answer),
              n_judgements = n(),
              .groups = "drop") %>%
    ## The empirical proportion hits 0 and 1 for some items, where the logit is
    ## undefined, so the log-odds column uses the Haldane-Anscombe correction
    ## (add 0.5 of a success and a failure). Use `human_p` for plots and
    ## `human_logodds` only where a logit scale is needed.
    mutate(human_logodds = qlogis((human_p * n_judgements + 0.5) /
                                    (n_judgements + 1)))

  llm_items %>%
    inner_join(human_items, by = "item_id") %>%
    rename(llm_p = p_alpha, llm_preference = preference)
}
