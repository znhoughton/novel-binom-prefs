## refit_nonconverged_checkpoints.R -------------------------------------------
##
## Some early Pythia checkpoints did not converge at the default sampler
## settings: five of the 99 stage-1 fits carried Rhat above 1.05, in four cases
## on the AbsPref_c coefficient itself rather than on a nuisance parameter.
##
## They are not caught by the `tokens > 1e6` guard that excludes the untrained
## checkpoints. A Pythia step is about 2.1M tokens, so that threshold only ever
## removed step 0; the offenders are steps 2, 4 and 16, which sit at the extreme
## left of the trajectory where the curve is steepest and a few points carry the
## most leverage over the fitted smooth.
##
## This refits exactly those fits, with the same model and the same data, under
## a longer, more conservative sampler. Nothing about the target posterior
## changes; it is only sampled properly. Converged fits are left alone, so this
## is cheap to re-run and is a no-op once everything passes.
##
## Run from Analysis/, then fit_trajectory_smooths.R (the stage-1 estimates it
## reads will have moved), then export_results.R.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(posterior)
})

source("load_llm_data.R")

MODELS_DIR <- "models"
CKPT_DIR   <- "../Data/llm-prefs-checkpoints"
SEED       <- 964
RHAT_MAX   <- 1.05

## Deliberately heavier than the defaults the first pass used (4000/2000,
## adapt_delta 0.8). These fits have 444 items x 51 prompts but a coefficient
## near zero and a large residual scale early in training, which is what makes
## the posterior awkward to sample rather than the data being small.
##
## This is a long job: three cost multipliers stack here (3x the draws, a much
## smaller step size from adapt_delta, and up to 2^15 leapfrog steps per
## iteration instead of 2^10), so budget on the order of hours per fit and run
## it overnight. The point is to settle these five checkpoints once rather than
## iterate on sampler settings.
ITER    <- 12000
WARMUP  <- 6000
CHAINS  <- 4
CORES   <- 4
CONTROL <- list(adapt_delta = 0.99, max_treedepth = 15)

set.seed(SEED)

## ---- which fits need redoing ----------------------------------------------
fits <- list.files(MODELS_DIR, pattern = "^ckpt_.*[.]rds$", full.names = TRUE)
if (!length(fits)) stop("no ckpt_ fits in ", MODELS_DIR)
message("checking ", length(fits), " stage-1 fit(s) for Rhat > ", RHAT_MAX)

worst <- map_dfr(fits, function(f) {
  m <- readRDS(f)
  s <- summarise_draws(as_draws_df(m))
  s <- s[!is.na(s$rhat), ]
  fx <- s[grepl("^b_", s$variable), ]
  out <- tibble(file = f,
                rhat = max(s$rhat),
                rhat_fixed = max(fx$rhat),
                n_bad = sum(s$rhat > RHAT_MAX))
  rm(m); gc(verbose = FALSE)
  out
})

bad <- worst %>% filter(rhat > RHAT_MAX) %>% arrange(desc(rhat))
if (!nrow(bad)) {
  message("all ", nrow(worst), " fits already converge; nothing to do")
  quit(save = "no", status = 0)
}
message("\n", nrow(bad), " fit(s) to refit:")
walk2(bad$file, bad$rhat, ~ message(sprintf("  %-54s rhat %.3f",
                                            basename(.x), .y)))

## ---- the data, and the specification the first pass used -------------------
ck <- load_checkpoint_data(CKPT_DIR)
scores <- ck$scores

## basename is ckpt_<make.names(model)>_<checkpoint>.rds, and a checkpoint name
## can itself contain underscores, so recover the pair by matching against the
## grid rather than by splitting the string.
grid <- scores %>%
  distinct(model, checkpoint) %>%
  mutate(stem = paste0("ckpt_", make.names(as.character(model)), "_",
                       checkpoint, ".rds"))

todo <- tibble(file = bad$file, stem = basename(bad$file)) %>%
  left_join(grid, by = "stem")
if (anyNA(todo$model)) {
  stop(sum(is.na(todo$model)), " fit file(s) could not be matched to a ",
       "(model, checkpoint) pair, first: ",
       todo$stem[is.na(todo$model)][[1]])
}

## ---- refit -----------------------------------------------------------------
res <- pmap_dfr(list(todo$file, todo$model, todo$checkpoint),
                function(f, m, ckn) {
  d <- scores %>% filter(model == m, checkpoint == ckn)
  if (!nrow(d)) stop("no scored rows for ", m, " / ", ckn)
  message(sprintf("\nrefitting %s (%s), %d rows",
                  str_remove(as.character(m), "^.*/"), ckn, nrow(d)))

  ## Sample to a scratch path and move it into place only once the fit is in
  ## hand. brms would otherwise load the cached fit instead of sampling, so the
  ## obvious version of this deletes the old file first -- and then an
  ## interrupted run has destroyed a fit it has not replaced.
  tmp <- paste0(tools::file_path_sans_ext(f), ".refitting")
  if (file.exists(paste0(tmp, ".rds"))) file.remove(paste0(tmp, ".rds"))

  t0 <- Sys.time()
  fit <- brm(
    preference ~ AbsPref_c + (AbsPref_c | prompt_value) + (1 | Item),
    data = d, family = gaussian(),
    iter = ITER, warmup = WARMUP, chains = CHAINS, cores = CORES, seed = SEED,
    control = CONTROL,
    file = tmp,
    refresh = 0
  )
  el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  if (!file.rename(paste0(tmp, ".rds"), f)) {
    stop("could not move the refitted model into place: ", f)
  }

  s  <- summarise_draws(as_draws_df(fit))
  s  <- s[!is.na(s$rhat), ]
  fx <- s[grepl("^b_", s$variable), ]
  sm <- summary(fit)
  out <- tibble(
    file = basename(f),
    model = str_remove(as.character(m), "^.*/"), checkpoint = ckn,
    est = sm$fixed["AbsPref_c", "Estimate"],
    se  = sm$fixed["AbsPref_c", "Est.Error"],
    rhat = max(s$rhat), rhat_fixed = max(fx$rhat),
    min_ess = min(s$ess_bulk, na.rm = TRUE),
    divergent = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
    minutes = el
  )
  message(sprintf("  rhat %.4f (fixed %.4f)  min ESS %.0f  divergent %d  %.1f min",
                  out$rhat, out$rhat_fixed, out$min_ess, out$divergent, el))
  rm(fit); gc(verbose = FALSE)
  out
})

## ---- report ----------------------------------------------------------------
cat("\n==================== refit summary ====================\n")
res %>%
  mutate(across(c(est, se), ~ sprintf("%.3f", .x)),
         rhat = sprintf("%.4f", rhat)) %>%
  select(model, checkpoint, est, se, rhat, min_ess, divergent) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

still <- res %>% filter(rhat > RHAT_MAX)
if (nrow(still)) {
  warning(nrow(still), " fit(s) still above Rhat ", RHAT_MAX,
          ": ", paste(still$file, collapse = ", "),
          ". Raise ITER or adapt_delta again, or reconsider the ",
          "specification at these checkpoints.")
} else {
  message("\nall refitted fits now converge (Rhat <= ", RHAT_MAX, ")")
}
message("\nnext: fit_trajectory_smooths.R, then export_results.R")
