# Pre-registered predictions: synthetic model ladders

Registered 2026-11-04, before the frozen run. Written by `python -m cardsegments.ladder preregister` from `results/ladder_trial.json`; the frozen results pin a sha256 prefix of this file and a test fails if it changes.

All data here is SYNTHETIC (the illustrative preset: round values, never the private cohort). The trial run used a different data draw (seed 2104) and CV seeds 100, 101, 102; the frozen run uses the preset's data seed (1104) and CV seeds 0-9.

## Frozen protocol

- StratifiedGroupKFold, 5 folds x 10 seeds, groups = set ids; impute + scale refit in every training fold; the same folds for every rung of both ladders.
- Classification: majority, velocity logistic, full logistic unpenalized (C=inf), full logistic L2 C=1, gbc; oracle row.
- Count regression on log1p(sold count): mean, velocity OLS, OLS, Huber, Poisson GLM (raw count), gbr; oracle row.
- Beats rule: paired group-bootstrap 95% CI (B = 1000) of the pooled delta excludes 0 AND delta >= +0.01 AUC, or >= the R^2 gate.
- Headline KPI: parked-capital rate at t* = 2/3 (m = h = 0.10), full logistic vs velocity logistic, paired group bootstrap; descriptive.

## Numbers fixed from the trial seeds

- Leakage band (fold-mean shuffled AUC): [0.466, 0.534] = 0.5 +/- 3 x 0.0113 (pooled within-rung sd of 120 single-fold shuffled AUCs, rounded up).
- Shuffled R^2 bound: fold-mean R^2 <= 0.026 = 3 x 0.0085 (pooled within-rung sd of 150 single-fold shuffled R^2s, rounded up).
- R^2 gate: delta R^2 vs velocity OLS >= 0.014 (the larger trial fold sd of velocity OLS R^2 across the two signals, rounded up).
- Constant rungs (majority, mean) are left out of the band: a per-fold constant has fold AUC 0.5 by definition and a biased pooled AUC.

## Predictions

- **P1 (leakage control).** Under row-level shuffled labels (grouped folds kept), every non-constant rung's fold-mean AUC lies inside the band and its fold-mean R^2 is at or below the bound, at both signals.
- **P2 (fold integrity).** Every rung of both ladders is scored on identical outer folds, no set group sits in both the train and test side of any fold, and no rung makes an internal validation split, at both signals.
- **P3 (oracle bound).** No rung beats the planted truth: fold-mean AUC and R^2 never exceed the oracle's by more than 0.01 and no count model's deviance falls more than 0.01 below it, at both signals; at signal 0 (where the planted log-rate is linear in trailing sales) the Poisson GLM's fold-mean deviance is within 0.03 of the oracle's.
- **P4 (null (signal 0)).** At cluster_signal 0, no classification rung beats the velocity logistic under the rule (paired CI excludes 0 AND delta AUC >= +0.01).
- **P5 (planted effect (signal 0.9)).** At cluster_signal 0.9, the full logistic (L2, C=1) beats the velocity logistic under the same rule.
- **P6 (headline KPI, null (signal 0)).** At cluster_signal 0, the paired CI of parked(velocity) - parked(model) at t* = 2/3 includes 0 (descriptive; no gate).
- **P7 (headline KPI, planted (signal 0.9)).** At cluster_signal 0.9, the paired CI of parked(velocity) - parked(model) at t* lies above 0: the model's buy list parks less capital (descriptive).
- **P8 (regression family).** At both signals the Poisson GLM beats velocity OLS on log1p(sold count) under the R^2 gate, and it has a higher fold-mean R^2 than OLS on the same features (the planted count is Poisson with a log link, so a linear fit to log1p misses its curvature).
- **P9 (regression null (signal 0)).** At cluster_signal 0, OLS on the full features does not beat velocity OLS under the R^2 gate.
- **P10 (complexity).** At cluster_signal 0.9, no rung's fold-mean AUC exceeds the full logistic (L2, C=1)'s by 0.01 or more: boosting buys less than the +0.01 bar.
