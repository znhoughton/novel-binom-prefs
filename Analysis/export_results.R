## export_results.R -----------------------------------------------------------
##
## Pulls small, tidy posterior summaries out of the cached brms fits in models/
## so that the writeup reads CSVs rather than loading model objects at render
## time. The companion papers use the same pattern: the fits stay in the
## analysis repo and the paper depends only on Results/.
##
## Run from Analysis/. Each fit is read once, reduced to the few columns the
## paper needs, and dropped before the next one loads, so peak memory is one
## model object rather than twenty. That matters because this is usually run
## while the render holds its own set of chains.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(posterior)
})

## load_llm_data.R brings in constraints.R as well as load_checkpoint_data(),
## which the Experiment 3 export needs for the token counts and the 444-item
## filter.
source("load_llm_data.R")

OUT_DIR    <- "../Results"
MODELS_DIR <- "models"
DATA_DIR   <- "../Data/llm-prefs"
DRAWS_KEEP <- 1000   # thinned draws per model: plenty for a ribbon, small on disk
SEED       <- 964

set.seed(SEED)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)


## ---- model metadata --------------------------------------------------------
## `make.names()` is what the fit filenames were built from, so it is also how a
## fit is matched back to the model that produced it.
meta <- list.files(DATA_DIR, pattern = "[.]csv$", full.names = TRUE) %>%
  map_dfr(~ read_csv(.x, n_max = 1, show_col_types = FALSE, progress = FALSE,
                     col_select = c(model, model_family, params))) %>%
  mutate(stem = make.names(model),
         name = str_remove(model, "^.*/"))
write_csv(meta, file.path(OUT_DIR, "model_meta.csv"))


## ---- item-level AbsPref ----------------------------------------------------
## The plotting range for the predicted curves, and the item table the appendix
## reports. Same 444-item set every fit uses, so the curves are drawn over the
## range the models were actually estimated on rather than over (0, 1).
items_csv <- c("../Experiment/novel_binoms_conditions.csv",
               "../Data/novel_binoms_conditions.csv")
items_csv <- items_csv[file.exists(items_csv)][1]
if (is.na(items_csv)) stop("conditions file not found")

items <- read_csv(items_csv, show_col_types = FALSE, progress = FALSE) %>%
  filter(is_filler == 0) %>%
  select(item_id, Alpha, all_of(CONSTRAINTS)) %>%
  drop_non_novel() %>%
  add_abs_pref()

write_csv(items %>% select(item_id, Alpha, AbsPref, AbsPref_c),
          file.path(OUT_DIR, "item_abspref.csv"))
message("items: ", nrow(items),
        "  AbsPref range [", round(min(items$AbsPref), 3), ", ",
        round(max(items$AbsPref), 3), "]")


## ---- posterior draws, bigram-controlled AbsPref arm ------------------------
## Only the population-level terms are kept. The predicted curve the paper draws
## is the population mean over prompts and items, so the group-level parameters
## are not needed and would dominate the file size.
fits <- list.files(MODELS_DIR, pattern = "^llm_abspref_bg_.*[.]rds$",
                   full.names = TRUE)
if (!length(fits)) stop("no arm-2 fits found in ", MODELS_DIR)
message("reading ", length(fits), " fit(s)")

draws <- map_dfr(fits, function(f) {
  stem <- sub("^llm_abspref_bg_", "", tools::file_path_sans_ext(basename(f)))
  m <- readRDS(f)
  d <- as_draws_df(m) %>%
    as_tibble() %>%
    select(any_of(c("b_Intercept", "b_AbsPref_c", "b_bigram_logodds_z")))
  ## Drop the fit before the next one loads, so twenty objects never coexist.
  rm(m)
  gc(verbose = FALSE)
  d %>%
    slice_sample(n = min(DRAWS_KEEP, nrow(d))) %>%
    mutate(stem = stem, .draw = row_number(), .before = 1)
})

draws <- draws %>%
  left_join(meta %>% select(stem, model, name, model_family, params), by = "stem")

if (anyNA(draws$model)) {
  stop(sum(is.na(draws$model)), " draw row(s) could not be matched to a model")
}

write_csv(draws, file.path(OUT_DIR, "abspref_bg_draws.csv"))
message("wrote ", nrow(draws), " draw rows for ",
        n_distinct(draws$model), " model(s)")


## ---- constraint arms -------------------------------------------------------
## Only summaries here, not draws: the heatmap needs a point estimate, an
## interval and the posterior mass per cell, which is 9 numbers per model rather
## than 9 x 1000.
##
## The constraints are fit on their raw scales, which are not commensurable with
## each other (Freq is a log ratio, Len a count difference). The SD of each
## constraint over the item set is exported alongside, so a figure that needs to
## compare across constraints can convert to a per-SD effect without the fits
## having to be refit on standardised predictors.
constraint_sd <- items %>%
  summarise(across(all_of(CONSTRAINTS_MODELLED), sd)) %>%
  pivot_longer(everything(), names_to = "term", values_to = "sd_item")
write_csv(constraint_sd, file.path(OUT_DIR, "constraint_sd.csv"))

summarise_fits <- function(prefix, exclude_bg) {
  fits <- list.files(MODELS_DIR, pattern = paste0("^", prefix, ".*[.]rds$"),
                     full.names = TRUE)
  ## "^llm_abspref_" also matches "llm_abspref_bg_", and likewise for the
  ## constraint arms, so the no-control arm has to exclude its own bg sibling.
  if (exclude_bg) fits <- fits[!grepl(paste0("^", prefix, "bg_"), basename(fits))]
  if (!length(fits)) return(NULL)
  message("reading ", length(fits), " ", prefix, "fit(s)")

  terms <- paste0("b_", c("AbsPref_c", CONSTRAINTS_MODELLED, "bigram_logodds_z"))
  map_dfr(fits, function(f) {
    stem <- sub(paste0("^", prefix), "", tools::file_path_sans_ext(basename(f)))
    m <- readRDS(f)
    d <- as_draws_df(m) %>%
      as_tibble() %>%
      select(any_of(terms))
    rm(m)
    gc(verbose = FALSE)

    d %>%
      pivot_longer(everything(), names_to = "term", values_to = "v") %>%
      group_by(term) %>%
      summarise(est  = median(v),
                lo   = quantile(v, .025),
                hi   = quantile(v, .975),
                ## Mass on the estimate's own side, so a cell reads as "how
                ## resolved is this effect" regardless of its sign.
                mass = 100 * max(mean(v > 0), mean(v < 0)),
                pgt0 = 100 * mean(v > 0),
                .groups = "drop") %>%
      mutate(term = str_remove(term, "^b_"), stem = stem, .before = 1)
  })
}

## ---- Experiment 2: human models --------------------------------------------
## Fit by human_prefs_analysis.Rmd. Same shape as the LLM summaries so the
## paper can put the two side by side without reshaping either.
human_fits <- c(abspref = "m_abspref", constraints = "m_constraints")
human_rows <- imap_dfr(human_fits, function(stem, arm) {
  f <- file.path(MODELS_DIR, paste0(stem, ".rds"))
  if (!file.exists(f)) return(NULL)
  m <- readRDS(f)
  s <- summary(m)
  d <- as_draws_df(m) %>% as_tibble()
  rm(m); gc(verbose = FALSE)
  tibble(term = rownames(s$fixed)) %>%
    mutate(arm  = arm,
           est  = s$fixed[term, "Estimate"],
           err  = s$fixed[term, "Est.Error"],
           lo   = s$fixed[term, "l-95% CI"],
           hi   = s$fixed[term, "u-95% CI"],
           pgt0 = map_dbl(term, function(v) {
             col <- paste0("b_", v)
             if (col %in% names(d)) 100 * mean(d[[col]] > 0) else NA_real_
           }))
})
if (!is.null(human_rows) && nrow(human_rows)) {
  write_csv(human_rows, file.path(OUT_DIR, "human.csv"))
  message("wrote human.csv: ", n_distinct(human_rows$arm), " arm(s), ",
          nrow(human_rows), " term(s)")
} else {
  message("no human fits found; human.csv not written")
}


## ---- Experiment 3: checkpoint trajectories ---------------------------------
## Stage-1 estimates and the stage-2 fitted curve, so the paper plots from two
## small CSVs rather than opening 106 model objects.
##
## The curve comes from the trimmed fits (`trajtrim_`), which exclude the
## untrained checkpoints below 10^6 tokens. Those points are still exported and
## still plotted; they are left out of the smooth because an untrained model's
## coefficient is near-unidentifiable, and for three Pythia models step 0 and
## step 1 carry identical scores, so including them also double-weights the
## left edge of the trajectory.
exp3_stage1 <- NULL
ck_dir <- "../Data/llm-prefs-checkpoints"
if (dir.exists(ck_dir) && length(list.files(ck_dir, pattern = "[.]csv$"))) {
  ck <- load_checkpoint_data(ck_dir)
  ck_grid <- ck$scores %>%
    distinct(model, checkpoint, tokens, log_tokens, model_family, params)

  exp3_stage1 <- pmap_dfr(list(ck_grid$model, ck_grid$checkpoint), function(m, ckn) {
    f <- file.path(MODELS_DIR, paste0("ckpt_", make.names(m), "_", ckn, ".rds"))
    if (!file.exists(f)) return(NULL)
    s <- summary(readRDS(f))
    gc(verbose = FALSE)
    tibble(model = as.character(m), checkpoint = ckn,
           est = s$fixed["AbsPref_c", "Estimate"],
           se  = s$fixed["AbsPref_c", "Est.Error"],
           sigma = s$spec_pars["sigma", "Estimate"])
  }) %>%
    left_join(ck_grid %>% mutate(model = as.character(model)),
              by = c("model", "checkpoint")) %>%
    mutate(est_z = est / sigma, se_z = se / sigma) %>%
    group_by(model) %>%
    mutate(prop = tokens / max(tokens)) %>%
    ungroup()
  write_csv(exp3_stage1, file.path(OUT_DIR, "exp3_stage1.csv"))
  message("wrote exp3_stage1.csv: ", nrow(exp3_stage1), " checkpoint(s), ",
          n_distinct(exp3_stage1$model), " model(s)")

  traj <- map_dfr(unique(exp3_stage1$model), function(m) {
    f <- file.path(MODELS_DIR, paste0("trajtrim_", make.names(m), ".rds"))
    if (!file.exists(f)) return(NULL)
    fit <- readRDS(f)
    d <- exp3_stage1 %>% filter(model == m, tokens > 1e6)
    g <- tibble(log_tokens = seq(min(d$log_tokens), max(d$log_tokens),
                                 length.out = 80), se_z = 0)
    out <- fitted(fit, newdata = g, re_formula = NA) %>%
      as_tibble() %>%
      bind_cols(g) %>%
      mutate(model = m,
             prop = 10^log_tokens / max(exp3_stage1$tokens[exp3_stage1$model == m]))
    rm(fit); gc(verbose = FALSE)
    out
  })
  if (nrow(traj)) {
    write_csv(traj, file.path(OUT_DIR, "exp3_traj.csv"))
    message("wrote exp3_traj.csv: ", n_distinct(traj$model), " model(s)")
  } else {
    message("no trajtrim_ fits found; exp3_traj.csv not written")
  }
}


for (arm in list(list(p = "llm_abspref_",        bg = TRUE,  f = "abspref.csv"),
                 list(p = "llm_abspref_bg_",     bg = FALSE, f = "abspref_bg.csv"),
                 list(p = "llm_constraints_",    bg = TRUE,  f = "constraints.csv"),
                 list(p = "llm_constraints_bg_", bg = FALSE, f = "constraints_bg.csv"))) {
  res <- summarise_fits(arm$p, arm$bg)
  if (is.null(res)) {
    message("no fits yet for ", arm$p, " - skipping ", arm$f)
    next
  }
  res <- res %>%
    left_join(meta %>% select(stem, model, name, model_family, params), by = "stem") %>%
    left_join(constraint_sd, by = "term")
  if (anyNA(res$model)) stop("unmatched fit(s) in ", arm$p)
  write_csv(res, file.path(OUT_DIR, arm$f))
  message("wrote ", arm$f, ": ", n_distinct(res$model), " model(s), ",
          n_distinct(res$term), " term(s)")
}


## ---- constraint collinearity -----------------------------------------------
## VIFs over the item set, reported in the appendix. Computed from the
## correlation matrix of the nine modelled constraints: VIF_j is the jth
## diagonal element of its inverse.
vif_tab <- local({
  X <- items %>% select(all_of(CONSTRAINTS_MODELLED)) %>% as.matrix()
  R <- cor(X)
  tibble(term = CONSTRAINTS_MODELLED, vif = diag(solve(R)))
})
write_csv(vif_tab, file.path(OUT_DIR, "constraint_vif.csv"))
message("wrote constraint_vif.csv: max VIF ", sprintf("%.2f", max(vif_tab$vif)))
