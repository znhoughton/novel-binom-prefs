## constraints.R --------------------------------------------------------------
##
## Morgan & Levy's constraint weights and the AbsPref (abstract preference)
## calculation, shared by the human and the LLM analyses so the two sides are
## scored by one definition. Sourced by load_human_data.R and load_llm_data.R;
## defines constants and functions only, with no side effects.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})


#' Items to drop because they are not novel in every corpus the analysis uses.
#'
#' The item set was screened for novelty against Dolma (Scripts/check_dolma.py):
#' a pair is kept only if "word1 and word2" and three other surface forms all
#' return 0. But novelty is corpus-relative, and the models are trained on
#' different data. Re-running the same screen against the Pile
#' (Scripts/check_novelty_corpora.py) found 14 of 458 items attested there, at
#' counts of 1 or 2 in 300B tokens, so single incidental occurrences rather than
#' conventionalised phrases. All 14 remain at 0 in Dolma, and BabyLM has none of
#' the 916 surface forms at all.
#'
#' They are dropped for every model rather than only for the Pythia models that
#' saw them, so the item set is identical everywhere and the 23 models stay
#' directly comparable. The cost is 3.1% of items.
#'
#' @param df   must contain item_id
#' @param path novelty_pile.csv from check_novelty_corpora.py
#' @return df with the non-novel items removed
drop_non_novel <- function(df, path = "../Data/novelty_pile.csv") {
  if (!file.exists(path)) {
    stop("Pile novelty results not found at ", path,
         ". Run: python Scripts/check_novelty_corpora.py ",
         "--index v4_piletrain_llama --corpus pile")
  }
  nov <- read_csv(path, show_col_types = FALSE, progress = FALSE)
  drop <- nov$item_id[!nov$novel]

  n_before <- dplyr::n_distinct(df$item_id)
  out <- df %>% filter(!item_id %in% drop)
  n_after <- dplyr::n_distinct(out$item_id)

  if (n_before != n_after) {
    message("dropped ", n_before - n_after, " item(s) attested in the Pile; ",
            n_after, " remain")
  }
  out
}


## Which corpus supplies the bigram control for which model family.
##
## The point of matching is composition, not size: the log-odds is a contrast,
## so multiplying every count by a constant cancels exactly, and a bigger corpus
## buys only less sampling noise. What does not cancel is register. Dolma and
## the Pile agree closely (r = .95 over these items, 88% sign agreement), both
## being scraped web text, while BabyLM — child-directed and transcribed speech —
## agrees with them only at r = .64. So matching earns its keep for the BabyLM
## models in particular.
##
## Dolma is the fallback for families whose training corpus is not indexed
## (GPT-2's WebText) or not disclosed (Qwen3, Llama 3.2, Gemma 3, Mistral), and
## is also what the human analysis uses, people having no training corpus. It is
## the corpus the novelty screen was run against, so the "trigram count is zero"
## guarantee and the bigram control come from the same text.
BIGRAM_CORPUS_BY_FAMILY <- c(pythia = "pile", olmo = "dolma", babylm = "babylm")
BIGRAM_CORPUS_DEFAULT   <- "dolma"

bigram_corpus_for <- function(family) {
  out <- unname(BIGRAM_CORPUS_BY_FAMILY[as.character(family)])
  ifelse(is.na(out), BIGRAM_CORPUS_DEFAULT, out)
}


#' Attach corpus bigram log-odds, the surface-statistics control.
#'
#' The trigram "word1 and word2" is zero for every item by construction (that is
#' the novelty criterion), so the surface baseline is built from the component
#' bigrams, which are not: see Scripts/compute_bigram_logodds.py. Positive means
#' the corpus favours the alphabetical order, matching the outcome's sign.
#'
#' Scaled but not centred: it is already a log-odds, so 0 means "the corpus
#' favours neither ordering", and mean-centring would move the origin onto the
#' average of whichever items happen to be in the set. Dividing by sd() rather
#' than scale(..., center = FALSE), which divides by the root mean square.
#'
#' @param df   must contain item_id
#' @param path bigram_logodds.csv from compute_bigram_logodds.py
#' @return df plus bigram_logodds (raw) and bigram_logodds_z (scaled)
add_bigram_logodds <- function(df, corpus = BIGRAM_CORPUS_DEFAULT,
                               dir = "../Data") {
  path <- file.path(dir, paste0("bigram_logodds_", corpus, ".csv"))
  if (!file.exists(path)) {
    stop("bigram log-odds not found at ", path,
         ". Run: python Scripts/compute_bigram_logodds.py --corpus ", corpus)
  }
  bg <- read_csv(path, show_col_types = FALSE, progress = FALSE) %>%
    select(item_id, bigram_logodds)

  missing <- setdiff(unique(df$item_id), bg$item_id)
  if (length(missing)) {
    stop(length(missing), " item(s) have no bigram log-odds, first: ", missing[[1]])
  }

  ## The divisor comes from the one-row-per-item file, not from `df`. In the
  ## human data an item appears once per participant, so scaling within df would
  ## make the unit depend on how many people happened to see each item and would
  ## drift as the sample grows. This way the unit is a property of the item set.
  sd_items <- sd(bg$bigram_logodds)

  df %>%
    left_join(bg, by = "item_id") %>%
    mutate(bigram_logodds_z = bigram_logodds / sd_items)
}

## Morgan & Levy abstract-preference weights -----------------------------------
##
## Posterior means of the constraint weights from the logistic regression
## component of Morgan & Levy's (2015) hierarchical Bayesian model of binomial
## ordering preferences, read off Figure 2(a), "Beta-binomial model with
## variable concentration". Higher weights indicate a greater contribution to
## productive-knowledge-based preferences.
##
## Recorded here because the chain of scripts these were copied through
## (small-lm-binom-ordering-prefs, hand-trained-model-prefs) carries the
## constants without naming a source, which makes the provenance impossible to
## check from the code alone.
##
## AbsPref/GenPref is a logistic over the weighted constraint sum. Keep these
## exact, so AbsPref is comparable across studies.
##
## Icon carries by far the largest weight (1.30), but every novel item in this
## study is coded Icon = 0, so the term contributes nothing here. It is kept in
## the formula for fidelity rather than dropped.
ML_WEIGHTS <- c(
  Intercept =  0.02191943,
  Form      =  0.23925834,
  Percept   =  0.24889543,
  Culture   =  0.41836997,
  Power     =  0.25967334,
  Intense   =  0.01867604,
  Icon      =  1.30365980,
  Freq      =  0.08553552,
  Len       =  0.15241566,
  Lapse     = -0.19381657,   # note: subtracted in the source formula
  BStress   =  0.36019221    # `*BStress` in the older files; renamed, same sign
)

CONSTRAINTS <- c("Form", "Percept", "Culture", "Power", "Intense",
                 "Icon", "Freq", "Len", "Lapse", "BStress")

## Constraints that actually vary across the 458 novel items. Icon is coded 0 for
## every item, so it has no variance and cannot be estimated as a predictor;
## including it would make the model unidentifiable.
CONSTRAINTS_MODELLED <- setdiff(CONSTRAINTS, "Icon")


#' Add AbsPref (abstract preference) to a data frame of binomials.
#'
#' @param df  must contain the constraint columns in `CONSTRAINTS`
#' @return    df plus `abs_pref_logit`, `AbsPref` (0-1) and `AbsPref_c`
#'            (centred at 0.5, which is what goes into the models)
add_abs_pref <- function(df) {
  missing <- setdiff(CONSTRAINTS, names(df))
  if (length(missing)) {
    stop("missing constraint columns: ", paste(missing, collapse = ", "))
  }
  df %>%
    mutate(across(all_of(CONSTRAINTS), as.numeric)) %>%
    mutate(
      abs_pref_logit =
        ML_WEIGHTS[["Intercept"]] +
        ML_WEIGHTS[["Form"]]    * Form +
        ML_WEIGHTS[["Percept"]] * Percept +
        ML_WEIGHTS[["Culture"]] * Culture +
        ML_WEIGHTS[["Power"]]   * Power +
        ML_WEIGHTS[["Intense"]] * Intense +
        ML_WEIGHTS[["Icon"]]    * Icon +
        ML_WEIGHTS[["Freq"]]    * Freq +
        ML_WEIGHTS[["Len"]]     * Len +
        ML_WEIGHTS[["Lapse"]]   * Lapse +
        ML_WEIGHTS[["BStress"]] * BStress,
      AbsPref   = plogis(abs_pref_logit),
      AbsPref_c = AbsPref - 0.5
    )
}
