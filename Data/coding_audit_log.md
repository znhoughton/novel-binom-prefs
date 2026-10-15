# Coding Audit Log: curated_500_binoms.csv

Audit of Form, Percept, Culture, Power, Intense, and Icon columns, based on Morgan & Levy's constraint coding guide (adapted from Benor & Levy, 2006).

## Constraint definitions used for this audit

- **Form** (Relative Formal Markedness): Less marked (broader meaning, more general) goes first.
- **Percept** (Perceptual Markedness): Less marked goes first. Key dimensions: Positive > Negative, Animate > Inanimate, Concrete > Abstract, Here > There, etc.
- **Culture** (Cultural Centrality): More culturally central or common goes first.
- **Power**: More powerful or culturally prioritized goes first (includes social hierarchy, value hierarchy).
- **Intense** (Intensity): More intense goes first.
- **Icon** (Iconic/Scalar Sequencing): Temporal or causal sequence preserved.

**Note:** Form and Icon are coded 0 for all 458 pairs. No issues found (appropriate for these novel binomial nouns).

**Transitivity check:** No transitivity violations were found for Culture, Power, or Intense across the entire dataset. The coding is internally consistent.

---

## Intense: Fierce predator vs mild animal undercoding

The most systematic issue found. Several fierce predator animals are coded Intense=0 against clearly less-intense animals, while comparable pairs ARE correctly coded. This suggests the coder may have missed these or considered them borderline when they shouldn't be.

**Correctly coded pairs (for comparison):**

| Row | Pair | Intense | |
|-----|------|---------|---|
| 112 | jays, vipers | -1 | vipers more intense ✓ |
| 176 | moths, vipers | -1 | ✓ |
| 211 | robins, vipers | -1 | ✓ |
| 252 | llamas, wolverines | -1 | ✓ |
| 266 | grasshoppers, hyenas | -1 | ✓ |
| 271 | hares, vipers | -1 | ✓ |
| 283 | caterpillars, cobras | -1 | ✓ |
| 288 | ferrets, vipers | -1 | ✓ |
| 328 | blackbirds, vipers | -1 | ✓ |
| 353 | bumblebees, vipers | -1 | ✓ |
| 399 | bumblebees, cobras | -1 | ✓ |

**Pairs coded Intense=0 that likely should be non-zero:**

| Row | word1 | word2 | Intense | Expected | Reasoning |
|-----|-------|-------|---------|----------|-----------|
| 96 | jays | wolverines | 0 | -1 | Wolverines are famously fierce. Compare jays/vipers (row 112, Intense=-1). If jays and vipers have an intensity difference, jays and wolverines should too. |
| 134 | crickets | wolverines | 0 | -1 | Same as above. Crickets are mild; wolverines are extremely aggressive. |
| 282 | pheasants | wolverines | 0 | -1 | Compare llamas/wolverines (row 252, Intense=-1). Pheasants are less intense than llamas. |
| 324 | bumblebees | wolverines | 0 | -1 | Compare bumblebees/vipers (row 353, Intense=-1) and bumblebees/cobras (row 399, Intense=-1). |
| 256 | pheasants | scorpions | 0 | -1 | Scorpions are venomous predators. Pheasants are mild birds. Compare grasshoppers/hyenas (row 266, Intense=-1). |
| 312 | dragonflies | vipers | 0 | -1 | Compare bumblebees/vipers (row 353, Intense=-1). Similar insect-viper pairing but different coding. |
| 343 | jaguars | mantises | 0 | +1 | Jaguars are apex predators. Compare caterpillars/jaguars (row 111, Intense=-1) and alligators/mantises (row 426, Intense=+1). |
| 273 | pumas | toads | 0 | +1 | Pumas are large predators. Compare caterpillars/jaguars (row 111, Intense=-1, i.e. jaguars more intense). |
| 230 | fireflies | leopards | 0 | -1 | Leopards are apex predators. Compare caterpillars/jaguars (row 111, Intense=-1). |
| 318 | bumblebees | leopards | 0 | -1 | Compare bumblebees/vipers (row 353, Intense=-1). Leopards are fierce predators. |

---

## Percept: Hornet-bird pairs coded 0 where pattern predicts non-zero

Every other insect-bird pair in the dataset has non-zero Percept (birds = positive/vertebrate = less perceptually marked). But two hornet-bird pairs are coded Percept=0.

**Correctly coded insect-bird pairs (for comparison):**

| Row | Pair | Percept | |
|-----|------|---------|---|
| 203 | hornets, pheasants | -1 | pheasants less marked ✓ |
| 145 | caterpillars, doves | -1 | ✓ |
| 205 | locusts, robins | -1 | ✓ |
| 214 | moths, pheasants | -1 | ✓ |
| 329 | bumblebees, magpies | -1 | ✓ |
| 356 | fireflies, quails | -1 | ✓ |
| 379 | locusts, wrens | -1 | ✓ |
| 381 | bumblebees, pheasants | -1 | ✓ |
| 415 | mantises, sparrows | -1 | ✓ |

**Pairs coded 0 that likely should be -1:**

| Row | word1 | word2 | Percept | Expected | Reasoning |
|-----|-------|-------|---------|----------|-----------|
| 39 | hornets | jays | 0 | -1 | Jays are birds (vertebrate, positive/neutral). Compare hornets/pheasants (row 203, Percept=-1). Both pheasants and jays are birds, so the coding should match. |
| 53 | hornets | swans | 0 | -1 | Swans are large, elegant birds (very positive). Compare bumblebees/pheasants (row 381, Percept=-1). Swans should be at least as "less marked" as pheasants. |

---

## Percept: Executioners-sailors undercoding

| Row | word1 | word2 | Percept | Expected | Reasoning |
|-----|-------|-------|---------|----------|-----------|
| 103 | executioners | sailors | 0 | -1 | Executioners are consistently coded as perceptually more marked (negative) against all other neutral/positive occupations: executioners/professors -1, executioners/masons -1, executioners/tailors -1. Sailors are a neutral/positive occupation, so this should follow the same pattern. |

---

## Power: Rank inconsistencies

**Duchesses vs countesses vs sheriffs:**

Duchesses outrank countesses in the nobility hierarchy. But countesses are coded as MORE powerful than sheriffs, while duchesses are coded as EQUAL to sheriffs.

| Row | word1 | word2 | Power | Issue |
|-----|-------|-------|-------|-------|
| 441 | countesses | sheriffs | +1 | Countesses coded as more powerful than sheriffs |
| 366 | duchesses | sheriffs | 0 | Duchesses coded as EQUAL to sheriffs |

All other 7 shared opponents of duchesses and countesses are coded identically (both Power=+1). Sheriffs is the only inconsistency. Duchesses/sheriffs should be Power=+1 to match.

**Corporals vs earls:**

| Row | word1 | word2 | Power | Expected | Reasoning |
|-----|-------|-------|-------|----------|-----------|
| 365 | corporals | earls | 0 | -1 | Earls are high nobility; corporals are very low military rank (below sergeants). Compare corporals/lords (row 107, Power=-1) and earls/sergeants (row 233, Power=+1, i.e. earls > sergeants > corporals). |

---

## Intense: Hornets vs wolverines direction

| Row | word1 | word2 | Intense | Possible issue |
|-----|-------|-------|---------|----------------|
| 124 | hornets | wolverines | +1 | Coded as hornets being more intense. Wolverines are legendary for their ferocity (known to fight bears, one of the most aggressive mammals). From a human-centric perspective, hornets may be more immediately threatening (swarm attacks, anaphylaxis risk), so this is a genuine judgment call. Worth double-checking. |

---

## Culture: Debatable judgment calls (internally consistent, but worth reviewing)

The following are NOT coding errors (no transitivity violations exist), but represent judgment calls on the ambiguous "culturally central or common" criterion that may be worth a second look. The coder appears to weight "commonly encountered" (favoring insects) in some pairs but "culturally prominent" (favoring large mammals) in others.

| Row | word1 | word2 | Culture | Question |
|-----|-------|-------|---------|----------|
| 195 | fireflies | gorillas | +1 | Fireflies more culturally central than gorillas? Gorillas are extremely culturally prominent (King Kong, conservation icon, media). Fireflies are commonly encountered but less culturally prominent. Compare chimps > fireflies (row 216). |
| 292 | bumblebees | gorillas | +1 | Bumblebees more culturally central than gorillas? Same question. Bumblebees are very common, but gorillas are very culturally prominent. |
| 308 | bumblebees | chimpanzees | +1 | Bumblebees more culturally central than chimps? Chimps are very prominent (Jane Goodall, research, media). |

Note: These are consistent with the coder's overall pattern (bumblebees and fireflies are coded as highly culturally central, presumably due to everyday commonality). No action needed unless you want to prioritize "culturally prominent" over "commonly encountered" for these pairs.

---

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| Percept undercoding | 2 | Hornet-bird pairs (rows 39, 53) coded 0 where all other insect-bird pairs are non-zero |
| Percept undercoding | 1 | Executioners/sailors (row 103) coded 0, inconsistent with all other executioner pairs |
| Power rank inconsistency | 1 | Duchesses/sheriffs Power=0 but countesses/sheriffs Power=+1 (row 366) |
| Power undercoding | 1 | Corporals/earls Power=0, should be -1 (row 365) |
| Intense undercoding | 10 | Fierce predators coded Intense=0 against mild animals where similar pairs have non-zero values |
| Intense direction | 1 | Hornets vs wolverines (row 124, debatable) |
| Culture judgment calls | 3 | Insect-mammal pairs where "common" vs "prominent" interpretation differs |

**Total items to double-check: 19**
