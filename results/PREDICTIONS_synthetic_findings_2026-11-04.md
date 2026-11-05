# Pre-registered predictions: synthetic-only v2 of the findings (2026-11-04)

Written before `python -m cardsegments.synth_findings build` was first run
with the frozen settings below. The bundle pins this file by a sha256 prefix,
and `tests/test_readme_drift.py` fails if the file changes afterwards. Each
verdict is decided in code (`synth_findings.judge_predictions`) by the rule
written here; partial verdicts are not allowed: a rule holds or it does not.

Everything below is about SYNTHETIC data from `cardsegments.synth`. None of it
is a claim about the real market or the private cohort.

## Frozen settings

- Segmentation data: `generate_segmented(n_rows=2000, n_clusters=4,
  separation=2.0, seed=1105, missingness=True)`; V1 = the screened feature
  set; V4 = price kept as one axis, every other kept feature residualized on
  log raw price.
- Weak-separation sensitivity: the same call with separation 1.0.
  Selection-effect sensitivity: the same call with `selection_effect=True`.
  Core sensitivity: V1 without the two nuisance axes (gem rate, volatility).
- k grid 1-10, gap statistic B = 50, stability and tiebreak bootstrap B = 200.
- Supervised data: the illustrative preset (`synth.illustrative_preset`) at
  cluster_signal 0 (primary case) and 0.9 (planted case); in-fold k-means
  k = 3 on the six features; 5 folds x 10 seeds; group bootstrap B = 1000.

## Predictions and their deciding rules

1. **V1 recovers the planted k.** The gap statistic (one-SE) on V1 picks k = 4.
2. **V1 recovers the planted labels.** Adjusted Rand index of the chosen V1
   solution against the planted labels >= 0.90.
3. **The chosen V1 solution is stable.** Bootstrap mean ARI (B = 200) >= 0.90.
4. **Both null controls find no structure.** The gap statistic (one-SE) picks
   k = 1 on the matched-Gaussian null AND on the column-shuffled null.
5. **Weak separation reads as a continuum.** On the separation-1.0
   sensitivity the gap statistic (one-SE) picks k = 1.
6. **Era composition is recovered without era features.** On V4, at least one
   cluster has a vintage lift CI entirely above 1 AND at least one cluster has
   a modern lift CI entirely above 1.
7. **Gem rate is a nuisance axis.** |Spearman rho(gem rate, log multiplier)|
   < 0.10 on V1 complete cases (the generator plants no centre on gem rate).
8. **Honest null at signal 0.** The nested cluster gate (era + velocity +
   cluster over era + velocity) FAILS the ship rule (95% CI excludes 0 AND
   mean ΔAUC >= +0.01) at cluster_signal 0.
9. **The gate can pass.** The same gate PASSES at cluster_signal 0.9.
10. **Planted-noise forward price.** The forward-price regression, whose
    target the generator draws as pure noise, has out-of-fold R² <= 0 at both
    signals.
