## fit_trajectory_smooths.R ---------------------------------------------------
##
## Stage 2 of Experiment 3: the fitted trajectory of abstract-preference
## sensitivity across training, one smooth per model.
##
## `export_results.R` reads these `trajtrim_` fits to write exp3_traj.csv, which
## is what the paper plots as the line and band in the trajectory figure. They
## had been produced ad hoc and were not reproducible from the repository; this
## script is that missing step.
##
## Two things distinguish it from the `traj_` fits in llm_prefs_analysis.Rmd:
##
##   - Checkpoints at or below 10^6 tokens are dropped. An untrained model's
##     coefficient is not identifiable, and because the checkpoints are
##     log-spaced that point sits far left of everything else, where the smooth
##     extrapolates wildly. For three Pythia models step 0 and step 1 also carry
##     identical scores, so keeping them double-weights the left edge.
##   - k = 4 rather than the notebook's default, since dropping a checkpoint
##     leaves 13-14 points per model and the basis has to stay well under that.
##
## The outcome is the stage-1 coefficient in units of its own residual scale
## (est_z), carried in with its standard error through se(), so stage 2 treats
## the stage-1 estimates as measured with error rather than as observed.
##
## Run from Analysis/, after refit_nonconverged_checkpoints.R and before
## export_results.R.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(posterior)
})

source("load_llm_data.R")

MODELS_DIR <- "models"
CKPT_DIR   <- "../Data/llm-prefs-checkpoints"
SEED       <- 964
ITER       <- 4000
WARMUP     <- 2000
CHAINS     <- 4
CORES      <- 4
SMOOTH_K   <- 4
MIN_TOKENS <- 1e6
CONTROL    <- list(adapt_delta = 0.95)

set.seed(SEED)

## ---- stage 1, read back from the cached per-checkpoint fits ---------------
ck <- load_checkpoint_data(CKPT_DIR)
grid <- ck$scores %>%
  distinct(model, checkpoint, tokens, log_tokens, model_family, params)

message("reading ", nrow(grid), " stage-1 fit(s)")
stage1 <- pmap_dfr(list(grid$model, grid$checkpoint), function(m, ckn) {
  f <- file.path(MODELS_DIR,
                 paste0("ckpt_", make.names(as.character(m)), "_", ckn, ".rds"))
  if (!file.exists(f)) {
    message("  missing, skipped: ", basename(f))
    return(NULL)
  }
  s <- summary(readRDS(f))
  gc(verbose = FALSE)
  tibble(model = as.character(m), checkpoint = ckn,
         est = s$fixed["AbsPref_c", "Estimate"],
         se  = s$fixed["AbsPref_c", "Est.Error"],
         sigma = s$spec_pars["sigma", "Estimate"])
}) %>%
  left_join(grid %>% mutate(model = as.character(model)),
            by = c("model", "checkpoint")) %>%
  ## Spread differs across models and across training, so the raw coefficient is
  ## not comparable between them; dividing by the residual SD makes it per-SD.
  mutate(est_z = est / sigma, se_z = se / sigma)

## ---- stage 2 ---------------------------------------------------------------
model_names <- unique(stage1$model)
message("\nfitting ", length(model_names), " trajectory smooth(s), k = ", SMOOTH_K)

res <- map_dfr(model_names, function(m) {
  d <- stage1 %>% filter(model == m, tokens > MIN_TOKENS)
  nm <- str_remove(m, "^.*/")
  if (nrow(d) <= SMOOTH_K + 1) {
    message("  ", nm, ": only ", nrow(d), " usable checkpoint(s), skipped")
    return(NULL)
  }
  f <- file.path(MODELS_DIR, paste0("trajtrim_", make.names(m), ".rds"))
  if (file.exists(f)) file.remove(f)

  fit <- brm(
    bf(as.formula(sprintf(
      "est_z | se(se_z, sigma = TRUE) ~ s(log_tokens, k = %d)", SMOOTH_K))),
    data = d, family = gaussian(),
    iter = ITER, warmup = WARMUP, chains = CHAINS, cores = CORES, seed = SEED,
    control = CONTROL,
    file = tools::file_path_sans_ext(f),
    refresh = 0
  )

  s <- summarise_draws(as_draws_df(fit))
  s <- s[!is.na(s$rhat), ]
  out <- tibble(model = nm, n = nrow(d),
                rhat = max(s$rhat), min_ess = min(s$ess_bulk, na.rm = TRUE),
                divergent = sum(subset(nuts_params(fit),
                                       Parameter == "divergent__")$Value))
  message(sprintf("  %-34s %2d points  rhat %.4f  min ESS %.0f  divergent %d",
                  nm, out$n, out$rhat, out$min_ess, out$divergent))
  rm(fit); gc(verbose = FALSE)
  out
})

cat("\n==================== stage 2 summary ====================\n")
as.data.frame(res %>% mutate(rhat = sprintf("%.4f", rhat))) %>%
  print(row.names = FALSE)

bad <- res %>% filter(rhat > 1.05)
if (nrow(bad)) {
  warning(nrow(bad), " trajectory fit(s) above Rhat 1.05: ",
          paste(bad$model, collapse = ", "))
}
message("\nnext: export_results.R")
