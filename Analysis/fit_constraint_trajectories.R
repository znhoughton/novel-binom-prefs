## fit_constraint_trajectories.R ----------------------------------------------
##
## How does each constraint's effect change over training?
##
## One varying-coefficient GAM per model rather than one Bayesian model per
## checkpoint. `s(log_tokens, by = Constraint)` fits a smooth multiplied by the
## constraint value, so the fitted smooth IS that constraint's coefficient as a
## function of training. Nine constraints plus an intercept smooth gives ten
## curves per model, from a single fit over all of that model's checkpoints.
##
## bam() rather than gam(): each model contributes ~317,000 rows (444 items x
## 51 prompts x 14 checkpoints), and `discrete = TRUE` makes that tractable.
##
## Writes Results/exp3_constraints.csv, one row per (model, constraint, token
## grid point) with the fitted coefficient and its standard error.

suppressPackageStartupMessages({
  library(tidyverse)
  library(mgcv)
})

source("load_llm_data.R")

OUT_DIR <- "../Results"
CK_DIR  <- "../Data/llm-prefs-checkpoints"
GRID_N  <- 80
SEED    <- 964

## Checkpoints are sampled log-uniformly and there are only 14-15 of them, so
## the smooth cannot support many basis functions. k = 5 leaves room for a rise
## and a levelling-off without chasing individual checkpoints.
K <- 5

set.seed(SEED)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ck <- load_checkpoint_data(CK_DIR)
## Untrained checkpoints are dropped, as in the AbsPref trajectories. At step 0
## a model's coefficients are not identifiable, and because the checkpoints are
## log-spaced that point sits far to the left of everything else, where the
## smooth extrapolates wildly: fit on all checkpoints, one by-smooth reaches
## +5.6 at the origin and flattens every other curve in the figure.
scores <- ck$scores %>%
  filter(tokens > 1e6) %>%
  mutate(prompt_value = factor(prompt_value), Item = factor(Item))

CONS <- CONSTRAINTS_MODELLED

## preference ~ s(log_tokens) + s(log_tokens, by = Constraint_k) for each k
##              + random intercepts for prompt and item
##
## The smooth is over log_tokens with the constraint as the `by` variable, not
## the reverse. From ?s: with a numeric `by`, "the elements multiply the smooth,
## evaluated at the corresponding covariate values (a varying coefficient model
## results)". So s(log_tokens, by = Form) fits f(log_tokens) * Form, and f is
## Form's coefficient as a function of training, which is what we want.
## s(Form, by = log_tokens) would instead smooth over Form -- which takes three
## values -- and let training enter linearly.
##
## The constraints are deliberately NOT included as main effects. Again from ?s:
## with a numeric `by` the smooth "is not usually subject to a centering
## constraint (so the by variable should not be added as an additional main
## effect)". Adding them would double-count every constraint.
rhs <- c(
  sprintf("s(log_tokens, k = %d)", K),
  sprintf("s(log_tokens, by = %s, k = %d)", CONS, K),
  "s(prompt_value, bs = 're')",
  "s(Item, bs = 're')"
)
FORM <- as.formula(paste("preference ~", paste(rhs, collapse = " + ")))
message("formula: ", deparse1(FORM))

model_names <- levels(droplevels(factor(scores$model)))

res <- map_dfr(model_names, function(m) {
  d <- scores %>% filter(model == m) %>% droplevels()
  message(sprintf("  %-34s %7d rows, %2d checkpoints",
                  str_remove(m, "^.*/"), nrow(d), n_distinct(d$checkpoint)))
  t0 <- Sys.time()
  fit <- bam(FORM, data = d, discrete = TRUE, nthreads = 4, method = "fREML")
  el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  saveRDS(fit, file.path("models", paste0("cgam_", make.names(m), ".rds")))

  ## Read each constraint's coefficient off its smooth. Setting one constraint
  ## to 1 and the rest to 0 makes the linear predictor the intercept smooth plus
  ## that constraint's smooth; subtracting a prediction with all constraints at
  ## 0 isolates the constraint term itself.
  grid <- tibble(log_tokens = seq(min(d$log_tokens), max(d$log_tokens),
                                  length.out = GRID_N),
                 prompt_value = d$prompt_value[1], Item = d$Item[1])
  base <- grid
  base[CONS] <- 0
  p0 <- predict(fit, newdata = base, type = "response", se.fit = TRUE,
                exclude = c("s(prompt_value)", "s(Item)"))

  out <- map_dfr(CONS, function(cn) {
    nd <- base
    nd[[cn]] <- 1
    p1 <- predict(fit, newdata = nd, type = "response", se.fit = TRUE,
                  exclude = c("s(prompt_value)", "s(Item)"))
    tibble(model = m, term = cn,
           log_tokens = grid$log_tokens,
           tokens = 10^grid$log_tokens,
           est = as.numeric(p1$fit - p0$fit),
           ## The two predictions share the intercept smooth, so its uncertainty
           ## cancels in the difference; what remains is the by-smooth's own.
           se  = as.numeric(sqrt(pmax(p1$se.fit^2 - p0$se.fit^2, 0))))
  }) %>%
    mutate(prop = tokens / max(d$tokens),
           model_family = d$model_family[1],
           params = d$params[1])

  message(sprintf("    done in %.1f min, dev.expl %.1f%%",
                  el, 100 * summary(fit)$dev.expl))
  rm(fit); gc(verbose = FALSE)
  out
})

write_csv(res, file.path(OUT_DIR, "exp3_constraints.csv"))
message("wrote exp3_constraints.csv: ", n_distinct(res$model), " model(s), ",
        n_distinct(res$term), " constraint(s)")


## ---- smooth summaries for the appendix -------------------------------------
## EDF / F / p for each by-smooth, matching the table the companion paper
## reports for its GAMs.
smooth_tab <- map_dfr(model_names, function(m) {
  f <- file.path("models", paste0("cgam_", make.names(m), ".rds"))
  if (!file.exists(f)) return(NULL)
  fit <- readRDS(f)
  st <- summary(fit)$s.table
  out <- as_tibble(st, rownames = "smooth") %>%
    filter(str_detect(smooth, ":")) %>%
    transmute(model = m,
              term = str_remove(smooth, "^.*:"),
              edf = edf, f_stat = F, p_value = `p-value`)
  rm(fit); gc(verbose = FALSE)
  out
})
write_csv(smooth_tab, file.path(OUT_DIR, "exp3_constraints_smooth.csv"))
message("wrote exp3_constraints_smooth.csv: ", nrow(smooth_tab), " smooth(s)")
