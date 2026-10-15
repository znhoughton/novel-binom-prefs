## constraints.R --------------------------------------------------------------
##
## Morgan & Levy's constraint weights and the AbsPref (abstract preference)
## calculation, shared by the human and the LLM analyses so the two sides are
## scored by one definition. Sourced by load_human_data.R and load_llm_data.R;
## defines constants and functions only, with no side effects.

suppressPackageStartupMessages({
  library(dplyr)
})

## Morgan & Levy abstract-preference weights -----------------------------------
##
## Lifted verbatim from the CoNLL script, where AbsPref/GenPref is defined as a
## logistic over the weighted constraint sum. Keep these exact, so AbsPref is
## comparable across the two studies.
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
