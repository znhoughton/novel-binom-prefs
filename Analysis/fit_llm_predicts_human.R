## fit_llm_predicts_human.R ---------------------------------------------------
##
## Does a language model's ordering preference predict what people choose?
##
## One logistic regression per model, on the human trial-level data:
##
##     alpha_answer ~ llm_pref_z + (1 | participant) + (1 | Item)
##
## Trial level rather than item level on purpose. Aggregating to one proportion
## per item throws away most of the signal here: with a median of seven
## judgements per binomial the reliability of an item proportion is about 0.4,
## which caps any item-level correlation near 0.65 however well a model actually
## predicts. The trial-level model does not pay that cost.
##
## The predictor is the model's item-level preference, averaged over prompts and
## divided by its own SD. Models differ substantially in spread (SD 1.46 to
## 3.07), so on raw scales a wide-spread model would earn a smaller coefficient
## for the same predictive accuracy. Dividing makes the 20 coefficients
## per-SD and therefore comparable. It is scaled but not centred: zero is the
## point where a model favours neither ordering, and that origin is worth
## keeping.
##
## Writes Results/llm_predicts_human.csv.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(posterior)
})

source("load_llm_data.R")
source("load_human_data.R")

OUT_DIR    <- "../Results"
MODELS_DIR <- "models"
HUMAN_DIR  <- "../Data/human-responses"
SEED       <- 964
ITER       <- 4000
WARMUP     <- 2000
CHAINS     <- 4
CORES      <- 4

options(contrasts = c("contr.sum", "contr.sum"))
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

llm   <- load_llm_data("../Data/llm-prefs")
human <- load_human_data(HUMAN_DIR)

## One row per (model, item): the model's preference averaged over prompts,
## scaled within model. `llm$items` already carries that average.
llm_items <- llm$items %>%
  group_by(model) %>%
  mutate(llm_pref_z = preference / sd(preference)) %>%
  ungroup() %>%
  select(model, model_family, params, item_id, llm_pref_z)

trials <- human$trials %>% select(participant, item_id, Item, alpha_answer)

model_names <- levels(droplevels(factor(llm_items$model)))
message(length(model_names), " model(s); ", nrow(trials), " human trials")

fit_one <- function(m) {
  d <- trials %>%
    inner_join(llm_items %>% filter(model == m), by = "item_id")
  if (!nrow(d)) stop("no rows for ", m)
  brm(
    alpha_answer ~ llm_pref_z + (1 | participant) + (1 | Item),
    data = d, family = bernoulli(link = "logit"),
    iter = ITER, warmup = WARMUP, chains = CHAINS, cores = CORES, seed = SEED,
    file = file.path(MODELS_DIR, paste0("h2llm_", make.names(m))),
    refresh = 0
  )
}

res <- map_dfr(model_names, function(m) {
  fit <- fit_one(m)
  dr  <- as_draws_df(fit)
  s   <- summary(fit)
  meta <- llm_items %>% filter(model == m) %>% slice(1)
  out <- tibble(
    model  = m,
    model_family = meta$model_family,
    params = meta$params,
    n_trials = nobs(fit),
    est  = s$fixed["llm_pref_z", "Estimate"],
    lo   = s$fixed["llm_pref_z", "l-95% CI"],
    hi   = s$fixed["llm_pref_z", "u-95% CI"],
    pgt0 = 100 * mean(dr$b_llm_pref_z > 0),
    max_rhat = max(summarise_draws(dr)$rhat, na.rm = TRUE)
  )
  rm(fit, dr); gc(verbose = FALSE)
  message(sprintf("  %-34s %+.3f [%.3f, %.3f]  P>0 %.1f%%",
                  str_remove(m, "^.*/"), out$est, out$lo, out$hi, out$pgt0))
  out
})

write_csv(res, file.path(OUT_DIR, "llm_predicts_human.csv"))
message("wrote llm_predicts_human.csv: ", nrow(res), " model(s)")
