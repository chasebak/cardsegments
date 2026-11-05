"""Two grouped k-fold model ladders on the SAME outer folds, plus the leakage
and planted-truth checks that must pass before any ladder number is read.

Classification ladder (target y_liquid):
    majority -> velocity logistic (trailing sales only) -> full logistic,
    unpenalized (C = np.inf) -> full logistic, L2 with C = 1 (the rung the
    supervised protocol used to call "primary_logistic") -> gbc
    (GradientBoostingClassifier, fixed documented defaults, random_state 42)
    -> synthetic only: ORACLE (the generator's planted P(liquid)).
Count-regression ladder (target log1p(n_sold_outcome)):
    mean -> velocity OLS -> OLS -> Huber -> Poisson GLM (unpenalized, on the
    raw count) -> gbr (GradientBoostingRegressor, fixed defaults) ->
    synthetic only: ORACLE (E[log1p N] under the planted Poisson rate).

Protocol (fixed before the frozen run, see results/PREDICTIONS_*.md):
  - Outer folds: StratifiedGroupKFold(5, shuffle=True, random_state=seed) on
    group_id (set ids), stratified on y_liquid, built ONCE per seed and handed
    to every rung of BOTH ladders, so every comparison is paired.
  - Median-impute + z-scale refit inside every training fold
    (supervised.fit_impute_scale). No rung tunes anything, and no rung makes
    an internal validation split (gbc/gbr keep n_iter_no_change=None), so
    there are no inner folds that could leak across groups.
  - Fold metrics: mean and sd (ddof = 1) over the 5 x n_seeds fold-seed
    cells. Pooled metrics: per-card out-of-fold predictions averaged over
    seeds, with a group(set)-level bootstrap CI. Fold sds are descriptive;
    there are no t-tests on folds.
  - "Beats the baseline" = the paired group-bootstrap 95% CI of the pooled
    delta excludes 0 AND the delta clears the bar (+0.01 AUC, the existing
    ship rule; for R^2 the gate written down from trial seeds).
  - Leakage control: labels permuted at ROW level with the grouped folds
    kept (the same fold indices as the real-label run); every non-constant
    rung's fold-mean AUC must fall inside the band written down from trial
    seeds, and its fold-mean R^2 under the bound. Constant rungs (majority,
    mean) are left out: a per-fold constant has fold AUC 0.5 by definition
    and its pooled out-of-fold AUC is biased by the fold-to-fold base-rate
    differences, so it would only test the splitter.
"""

import argparse
import hashlib
import json
import os
import time
import warnings

import numpy as np
from scipy.stats import poisson
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
)
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from cardsegments import synth
from cardsegments.supervised import (
    SHIP_MIN_DELTA,
    apply_impute_scale,
    confusion_at_threshold,
    cost_threshold,
    fit_impute_scale,
    group_bootstrap_indices,
    group_index_lists,
)

FIT_SEED = 42
N_FOLDS = 5
FROZEN_CV_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
TRIAL_CV_SEEDS = (100, 101, 102)
TRIAL_DATA_SEED = 2104  # trial data is a different draw from the frozen preset
SHUFFLE_SEED_OFFSET = 5000
SIGNALS = (0.0, 0.9)
B_BOOT = 1000
BOOT_SEED = 2000
DEFAULT_M = 0.10
DEFAULT_H = 0.10
ORACLE_AUC_TOLERANCE = 0.01
GLM_ORACLE_DEVIANCE_TOLERANCE = 0.03
BAND_SD_MULTIPLIER = 3.0
REGISTRATION_DATE = "2026-11-04"
PREDICTIONS_FILENAME = "PREDICTIONS_synthetic_ladder_%s.md" % REGISTRATION_DATE
TRIAL_FILENAME = "ladder_trial.json"
FROZEN_FILENAME = "ladder_frozen.json"
LEAK_FILENAME = "leak_detection.json"
# Moderate-leak control (measures the registered band's detection limit; it
# never changes the band): the shuffled label plus N(0, sd) noise as a feature.
LEAK_NOISE_SDS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)
LEAK_NOISE_SEED = 7100
LEAK_DATA_SEED = 0
LEAK_N_ROWS = 3000
LEAK_CV_SEEDS = (0, 1)
SHA_PREFIX_LEN = 16
LABEL_PENALTY_NONE_WARNING = "Setting penalty=None will ignore the C and l1_ratio parameters"


# ---------------------------------------------------------------- rungs

def make_majority():
    return None


def make_velocity_logistic():
    return LogisticRegression(max_iter=1000)


def make_unpenalized_logistic():
    # C = np.inf is the scikit-learn >= 1.8 spelling of "no penalty"
    # (penalty= is deprecated there); older versions read it differently,
    # hence the >= 1.8 floor in pyproject.toml.
    return LogisticRegression(C=np.inf, max_iter=5000)


def make_l2_logistic():
    return LogisticRegression(max_iter=1000)


def make_gbc():
    return GradientBoostingClassifier(random_state=FIT_SEED)


def make_ols():
    return LinearRegression()


def make_huber():
    return HuberRegressor(max_iter=1000)


def make_poisson_glm():
    return PoissonRegressor(alpha=0.0, max_iter=1000)


def make_gbr():
    return GradientBoostingRegressor(random_state=FIT_SEED)


CLASSIFICATION_RUNGS = [
    {"name": "majority", "label": "majority (train-fold base rate)",
     "design": "none", "make": make_majority, "constant": True},
    {"name": "velocity", "label": "velocity logistic (trailing sales, L2 C=1)",
     "design": "baseline", "make": make_velocity_logistic, "constant": False},
    {"name": "logistic_unpenalized", "label": "full logistic, unpenalized (C=inf)",
     "design": "features", "make": make_unpenalized_logistic, "constant": False},
    {"name": "logistic_l2_c1", "label": "full logistic, L2 C=1 (was primary_logistic)",
     "design": "features", "make": make_l2_logistic, "constant": False},
    {"name": "gbc", "label": "gradient boosting (gbc, defaults)",
     "design": "features", "make": make_gbc, "constant": False},
]

REGRESSION_RUNGS = [
    {"name": "mean", "label": "mean (train-fold mean)", "design": "none",
     "make": make_majority, "constant": True, "count_model": True},
    {"name": "velocity_ols", "label": "velocity OLS (trailing sales)",
     "design": "baseline", "make": make_ols, "constant": False, "count_model": False},
    {"name": "ols", "label": "OLS, full features", "design": "features",
     "make": make_ols, "constant": False, "count_model": False},
    {"name": "huber", "label": "Huber (robust, epsilon 1.35)", "design": "features",
     "make": make_huber, "constant": False, "count_model": False},
    {"name": "poisson_glm", "label": "Poisson GLM on the raw count (unpenalized)",
     "design": "features", "make": make_poisson_glm, "constant": False,
     "count_model": True},
    {"name": "gbr", "label": "gradient boosting regressor (gbr, defaults)",
     "design": "features", "make": make_gbr, "constant": False, "count_model": False},
]

BASELINE_CLASSIFIER = "velocity"
BASELINE_REGRESSOR = "velocity_ols"
HEADLINE_MODEL = "logistic_l2_c1"


def rung_names(rungs):
    names = []
    for rung in rungs:
        names.append(rung["name"])
    return names


def rung_by_name(rungs, name):
    for rung in rungs:
        if rung["name"] == name:
            return rung
    raise KeyError(name)


def internal_split_violations(rungs):
    """Names of rungs whose estimator would make its own validation split.

    Why: an internal early-stopping split is NOT grouped, so it would leak
    set-level information inside a training fold (the HGB gotcha in the
    plan). Rule: a rung passes only if its estimator has no
    n_iter_no_change / early_stopping switched on and is not a *CV
    estimator. Failure mode: an estimator with a differently named internal
    split would pass; the ladder uses only the classes listed above.
    """
    violations = []
    for rung in rungs:
        estimator = rung["make"]()
        if estimator is None:
            continue
        params = estimator.get_params()
        early = params.get("early_stopping", False)
        no_change = params.get("n_iter_no_change", None)
        class_name = type(estimator).__name__
        is_cv_class = class_name.endswith("CV")
        if early is True or no_change is not None or is_cv_class:
            violations.append(rung["name"])
    return violations


# ---------------------------------------------------------------- folds

def outer_folds(y, groups, n_folds, seed):
    """List of (train_idx, test_idx): StratifiedGroupKFold, built once per seed."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    placeholder = np.zeros((len(y), 1))
    split_iter = splitter.split(placeholder, y, groups)
    folds = []
    for train_idx, test_idx in split_iter:
        folds.append((train_idx, test_idx))
    return folds


def shared_groups_count(folds, groups):
    """Number of (fold, group) pairs where a group sits in train AND test.

    Must be 0 for grouped CV; a plain KFold on the same rows gives > 0 (the
    control in tests/test_ladder.py)."""
    shared = 0
    for train_idx, test_idx in folds:
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        overlap = train_groups & test_groups
        shared += len(overlap)
    return shared


def fold_fingerprint(folds, n_rows):
    """Test-fold id per row (the fold each row was scored in)."""
    fold_id = np.full(n_rows, -1, dtype=int)
    fold_number = 0
    for _train_idx, test_idx in folds:
        fold_id[test_idx] = fold_number
        fold_number += 1
    return fold_id


# ---------------------------------------------------------------- metrics

def expected_log1p_poisson(mu):
    """E[log(1 + N)] for N ~ Poisson(mu), elementwise.

    Formula: sum over k = 0..K of log1p(k) * pmf(k; mu), with K =
    ceil(max mu + 12 sqrt(max mu) + 20), so the dropped upper tail has
    probability far below 1e-12. This is the best possible log1p-scale
    prediction for a card whose sale rate is mu (squared loss), so it gives
    the oracle row and turns the Poisson GLM's mean into a log1p-scale
    prediction. Failure mode: memory grows with n x K; fine for K in the
    hundreds.
    """
    mu = np.asarray(mu, dtype=float)
    mu_max = float(mu.max())
    k_max = int(np.ceil(mu_max + 12.0 * np.sqrt(mu_max) + 20.0))
    counts = np.arange(k_max + 1)
    log_values = np.log1p(counts)
    pmf_matrix = poisson.pmf(counts[None, :], mu[:, None])
    return pmf_matrix @ log_values


def beats_baseline(delta, ci_low, ci_high, min_delta):
    """The ship rule: the CI excludes 0 on the positive side AND delta >= bar.

    A CI entirely below 0 also "excludes 0" but means the rung is WORSE, so
    it never beats. Failure mode: a None CI (bootstrap had no valid
    resample) returns False."""
    if delta is None or ci_low is None or ci_high is None:
        return False
    excludes_zero_above = ci_low > 0.0
    clears_bar = delta >= min_delta
    return bool(excludes_zero_above and clears_bar)


def parked_capital_rate(y, probs, threshold):
    """Headline KPI: FP / (TP + FP) among cards with P(liquid) >= threshold.

    Formula: of the cards the buy list flags, the share that sat unsold
    (capital parked for the outcome window). Baseline: the velocity
    logistic at the same threshold on the same folds. Failure mode: an
    empty buy list has no rate; None is returned and the caller reports it.
    """
    flagged = probs >= threshold
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return None
    n_sat = int(np.sum(flagged & (y == 0)))
    return n_sat / n_flagged


def paired_group_bootstrap_delta(y, pred_a, pred_b, groups, metric, n_boot, seed):
    """Paired group-level bootstrap of metric(a) - metric(b).

    The SAME resampled set groups score both prediction vectors in every
    replicate, so the CI is for the paired difference. metric(y, pred)
    returns a float or None; replicates where either is None (one class
    only, empty buy list) are skipped and counted.
    """
    rng = np.random.default_rng(seed)
    index_lists = group_index_lists(groups)
    deltas = []
    n_skipped = 0
    replicates = range(n_boot)
    for _replicate in replicates:
        idx = group_bootstrap_indices(groups, rng, index_lists)
        y_b = y[idx]
        value_a = metric(y_b, pred_a[idx])
        value_b = metric(y_b, pred_b[idx])
        if value_a is None or value_b is None:
            n_skipped += 1
            continue
        deltas.append(value_a - value_b)
    if len(deltas) == 0:
        return {"ci_low": None, "ci_high": None, "n_boot_valid": 0,
                "n_boot_skipped": n_skipped}
    delta_arr = np.array(deltas)
    return {"ci_low": float(np.percentile(delta_arr, 2.5)),
            "ci_high": float(np.percentile(delta_arr, 97.5)),
            "n_boot_valid": int(len(delta_arr)), "n_boot_skipped": n_skipped}


def group_bootstrap_ci(y, pred, groups, metric, n_boot, seed):
    """Group-level bootstrap 95% CI of metric(y, pred)."""
    rng = np.random.default_rng(seed)
    index_lists = group_index_lists(groups)
    values = []
    replicates = range(n_boot)
    for _replicate in replicates:
        idx = group_bootstrap_indices(groups, rng, index_lists)
        value = metric(y[idx], pred[idx])
        if value is None:
            continue
        values.append(value)
    value_arr = np.array(values)
    return {"ci_low": float(np.percentile(value_arr, 2.5)),
            "ci_high": float(np.percentile(value_arr, 97.5)),
            "n_boot_valid": int(len(value_arr))}


def auc_or_none(y, pred):
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, pred))


def r2_or_none(y, pred):
    if np.all(y == y[0]):
        return None
    return float(r2_score(y, pred))


def mean_sd(values):
    """Mean and sd (ddof = 1) of fold-seed cells; None for an empty list."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": None, "sd": None, "n_cells": 0}
    if len(arr) == 1:
        return {"mean": float(arr.mean()), "sd": None, "n_cells": 1}
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
            "n_cells": int(len(arr))}


# ---------------------------------------------------------------- the run

def design_for(rung, x_feat, x_base, label_column):
    design = rung["design"]
    if design == "features":
        return x_feat
    if design == "baseline":
        return x_base
    if design == "features_plus_label":
        # leakage CONTROL only (tests): the label itself as a feature
        return np.column_stack([x_feat, label_column])
    if design == "features_plus_noisy_label":
        # leakage CONTROL only: the label plus N(0, sd) noise as a feature, so
        # the leak's strength is set by sd (seeded by the column length, so a
        # rerun draws the same noise)
        noise_rng = np.random.default_rng(LEAK_NOISE_SEED + len(label_column))
        noise = noise_rng.normal(0.0, rung["leak_noise_sd"], len(label_column))
        return np.column_stack([x_feat, np.asarray(label_column, dtype=float) + noise])
    return None


def fit_predict_classifier(rung, train_design, test_design, y_train, n_test):
    if rung["constant"]:
        return np.full(n_test, float(y_train.mean()))
    model = rung["make"]()
    with warnings.catch_warnings():
        # scikit-learn 1.8 warns that C=np.inf "ignores C" even though C=np.inf
        # is its own documented way to ask for no penalty; only that message
        # is silenced.
        warnings.filterwarnings("ignore", message=LABEL_PENALTY_NONE_WARNING)
        model.fit(train_design, y_train)
    return model.predict_proba(test_design)[:, 1]


def fit_predict_regressor(rung, train_design, test_design, t_train, count_train, n_test):
    """Returns (log1p-scale prediction, count-scale mean or None)."""
    if rung["constant"]:
        log_pred = np.full(n_test, float(t_train.mean()))
        count_mean = np.full(n_test, float(count_train.mean()))
        return log_pred, count_mean
    model = rung["make"]()
    if rung["name"] == "poisson_glm":
        model.fit(train_design, count_train)
        count_mean = model.predict(test_design)
        log_pred = expected_log1p_poisson(count_mean)
        return log_pred, count_mean
    model.fit(train_design, t_train)
    return model.predict(test_design), None


def cross_validate_ladder(frame, feature_cols, baseline_cols, cv_seeds,
                          n_folds=N_FOLDS, truth=None, shuffle=False,
                          classification_rungs=None, regression_rungs=None,
                          outcome_col="y_liquid", count_col="n_sold_outcome"):
    """Run both ladders on identical outer folds (named after HW2's
    cross_validate: returns per-rung fold-metric lists plus out-of-fold
    predictions per seed).

    shuffle=True permutes y and the count TOGETHER at row level (one
    permutation per CV seed, seeded SHUFFLE_SEED_OFFSET + seed) while the
    folds stay the ones built from the real labels; the oracle is skipped.
    """
    if classification_rungs is None:
        classification_rungs = CLASSIFICATION_RUNGS
    if regression_rungs is None:
        regression_rungs = REGRESSION_RUNGS
    y_real = frame[outcome_col].to_numpy(dtype=int)
    count_real = frame[count_col].to_numpy(dtype=float)
    groups = frame["group_id"].to_numpy()
    feature_block = frame[feature_cols].to_numpy(dtype=float)
    baseline_block = frame[baseline_cols].to_numpy(dtype=float)
    n_rows = len(frame)
    n_seeds = len(cv_seeds)
    use_oracle = truth is not None and not shuffle

    class_names = rung_names(classification_rungs)
    reg_names = rung_names(regression_rungs)
    oof_class = {}
    fold_seen_class = {}
    cells_class = {}
    seconds_class = {}
    for name in class_names:
        oof_class[name] = np.full((n_seeds, n_rows), np.nan)
        fold_seen_class[name] = np.full((n_seeds, n_rows), -1, dtype=int)
        cells_class[name] = {"auc": [], "brier": []}
        seconds_class[name] = 0.0
    oof_reg = {}
    fold_seen_reg = {}
    cells_reg = {}
    seconds_reg = {}
    for name in reg_names:
        oof_reg[name] = np.full((n_seeds, n_rows), np.nan)
        fold_seen_reg[name] = np.full((n_seeds, n_rows), -1, dtype=int)
        cells_reg[name] = {"mse": [], "mae": [], "r2": [], "deviance": []}
        seconds_reg[name] = 0.0
    oracle_cells = {"auc": [], "brier": [], "mse": [], "mae": [], "r2": [], "deviance": []}
    if use_oracle:
        oracle_prob = np.asarray(truth["p_liquid"], dtype=float)
        oracle_lambda = np.asarray(truth["lambda_sold"], dtype=float)
        oracle_log = expected_log1p_poisson(oracle_lambda)
    shared_total = 0
    y_by_seed = []
    t_by_seed = []

    seed_positions = range(n_seeds)
    for seed_position in seed_positions:
        seed = cv_seeds[seed_position]
        folds = outer_folds(y_real, groups, n_folds, seed)
        shared_total += shared_groups_count(folds, groups)
        if shuffle:
            perm_rng = np.random.default_rng(SHUFFLE_SEED_OFFSET + seed)
            permutation = perm_rng.permutation(n_rows)
            y_used = y_real[permutation]
            count_used = count_real[permutation]
        else:
            y_used = y_real
            count_used = count_real
        t_used = np.log1p(count_used)
        y_by_seed.append(y_used)
        t_by_seed.append(t_used)
        fold_number = 0
        for train_idx, test_idx in folds:
            n_test = len(test_idx)
            feat_stats = fit_impute_scale(feature_block[train_idx].copy())
            x_feat_train = apply_impute_scale(feature_block[train_idx].copy(), feat_stats)
            x_feat_test = apply_impute_scale(feature_block[test_idx].copy(), feat_stats)
            base_stats = fit_impute_scale(baseline_block[train_idx].copy())
            x_base_train = apply_impute_scale(baseline_block[train_idx].copy(), base_stats)
            x_base_test = apply_impute_scale(baseline_block[test_idx].copy(), base_stats)
            y_train = y_used[train_idx]
            y_test = y_used[test_idx]
            t_train = t_used[train_idx]
            t_test = t_used[test_idx]
            count_train = count_used[train_idx]
            count_test = count_used[test_idx]

            for rung in classification_rungs:
                name = rung["name"]
                train_design = design_for(rung, x_feat_train, x_base_train, y_train)
                test_design = design_for(rung, x_feat_test, x_base_test, y_test)
                started = time.perf_counter()
                probs = fit_predict_classifier(rung, train_design, test_design,
                                               y_train, n_test)
                seconds_class[name] += time.perf_counter() - started
                oof_class[name][seed_position, test_idx] = probs
                fold_seen_class[name][seed_position, test_idx] = fold_number
                fold_auc = auc_or_none(y_test, probs)
                if fold_auc is not None:
                    cells_class[name]["auc"].append(fold_auc)
                cells_class[name]["brier"].append(float(brier_score_loss(y_test, probs)))

            for rung in regression_rungs:
                name = rung["name"]
                train_design = design_for(rung, x_feat_train, x_base_train, t_train)
                test_design = design_for(rung, x_feat_test, x_base_test, t_test)
                started = time.perf_counter()
                log_pred, count_mean = fit_predict_regressor(
                    rung, train_design, test_design, t_train, count_train, n_test)
                seconds_reg[name] += time.perf_counter() - started
                oof_reg[name][seed_position, test_idx] = log_pred
                fold_seen_reg[name][seed_position, test_idx] = fold_number
                cells_reg[name]["mse"].append(float(mean_squared_error(t_test, log_pred)))
                cells_reg[name]["mae"].append(float(mean_absolute_error(t_test, log_pred)))
                cells_reg[name]["r2"].append(float(r2_score(t_test, log_pred)))
                if count_mean is not None:
                    deviance = float(mean_poisson_deviance(count_test, count_mean))
                    cells_reg[name]["deviance"].append(deviance)

            if use_oracle:
                oracle_test_prob = oracle_prob[test_idx]
                oracle_test_log = oracle_log[test_idx]
                oracle_auc = auc_or_none(y_test, oracle_test_prob)
                if oracle_auc is not None:
                    oracle_cells["auc"].append(oracle_auc)
                oracle_cells["brier"].append(float(brier_score_loss(y_test, oracle_test_prob)))
                oracle_cells["mse"].append(float(mean_squared_error(t_test, oracle_test_log)))
                oracle_cells["mae"].append(float(mean_absolute_error(t_test, oracle_test_log)))
                oracle_cells["r2"].append(float(r2_score(t_test, oracle_test_log)))
                oracle_deviance = float(mean_poisson_deviance(count_test,
                                                              oracle_lambda[test_idx]))
                oracle_cells["deviance"].append(oracle_deviance)
            fold_number += 1

    return {
        "cv_seeds": list(cv_seeds), "n_folds": n_folds, "shuffle": shuffle,
        "y": y_real, "t": np.log1p(count_real), "groups": groups,
        "y_by_seed": y_by_seed, "t_by_seed": t_by_seed,
        "oof_class": oof_class, "oof_reg": oof_reg,
        "fold_seen_class": fold_seen_class, "fold_seen_reg": fold_seen_reg,
        "cells_class": cells_class, "cells_reg": cells_reg,
        "seconds_class": seconds_class, "seconds_reg": seconds_reg,
        "oracle_cells": oracle_cells if use_oracle else None,
        "oracle_prob": oracle_prob if use_oracle else None,
        "oracle_log": oracle_log if use_oracle else None,
        "shared_group_pairs": int(shared_total),
        "classification_rungs": classification_rungs,
        "regression_rungs": regression_rungs,
    }


def fold_checks(raw):
    """Fold-integrity facts computed from what each rung actually saw."""
    names = list(raw["fold_seen_class"].keys())
    reference = raw["fold_seen_class"][names[0]]
    identical = True
    for name in names:
        if not np.array_equal(raw["fold_seen_class"][name], reference):
            identical = False
    reg_names = list(raw["fold_seen_reg"].keys())
    for name in reg_names:
        if not np.array_equal(raw["fold_seen_reg"][name], reference):
            identical = False
    every_row_scored = bool(np.all(reference >= 0))
    return {"identical_folds_all_rungs": bool(identical),
            "every_row_scored_once_per_seed": every_row_scored,
            "shared_group_pairs_outer": raw["shared_group_pairs"],
            "internal_split_rungs": internal_split_violations(
                raw["classification_rungs"] + raw["regression_rungs"])}


def pooled(oof):
    """Per-card out-of-fold prediction averaged over CV seeds."""
    return oof.mean(axis=0)


def summarize_classification(raw, n_boot, boot_seed):
    """Classification ladder rows: fold mean/sd, pooled AUC + CI, paired
    delta vs velocity + CI, beats?, Brier, seconds."""
    y = raw["y"]
    groups = raw["groups"]
    base_pooled = pooled(raw["oof_class"][BASELINE_CLASSIFIER])
    rows = []
    for rung in raw["classification_rungs"]:
        name = rung["name"]
        cells = raw["cells_class"][name]
        row = {"name": name, "label": rung["label"], "constant": rung["constant"],
               "auc_folds": mean_sd(cells["auc"]),
               "brier_folds": mean_sd(cells["brier"]),
               "fit_seconds": float(raw["seconds_class"][name])}
        if rung["constant"]:
            # Shown only to document WHY constants are excluded: per fold the
            # AUC is exactly 0.5, pooled it is not (fold base rates differ).
            constant_pooled = pooled(raw["oof_class"][name])
            row["pooled_auc"] = None
            row["pooled_auc_biased_constant"] = float(roc_auc_score(y, constant_pooled))
            row["delta_auc_vs_velocity"] = None
            row["beats_velocity"] = None
        else:
            rung_pooled = pooled(raw["oof_class"][name])
            pooled_value = float(roc_auc_score(y, rung_pooled))
            ci = group_bootstrap_ci(y, rung_pooled, groups, auc_or_none, n_boot, boot_seed)
            row["pooled_auc"] = {"value": pooled_value, **ci}
            if name == BASELINE_CLASSIFIER:
                row["delta_auc_vs_velocity"] = None
                row["beats_velocity"] = None
            else:
                delta = pooled_value - float(roc_auc_score(y, base_pooled))
                delta_ci = paired_group_bootstrap_delta(
                    y, rung_pooled, base_pooled, groups, auc_or_none, n_boot, boot_seed)
                row["delta_auc_vs_velocity"] = {"value": delta, **delta_ci}
                row["beats_velocity"] = beats_baseline(
                    delta, delta_ci["ci_low"], delta_ci["ci_high"], SHIP_MIN_DELTA)
        rows.append(row)
    if raw["oracle_cells"] is not None:
        oracle_value = float(roc_auc_score(y, raw["oracle_prob"]))
        rows.append({"name": "oracle", "label": "ORACLE (planted P(liquid))",
                     "constant": False,
                     "auc_folds": mean_sd(raw["oracle_cells"]["auc"]),
                     "brier_folds": mean_sd(raw["oracle_cells"]["brier"]),
                     "fit_seconds": None,
                     "pooled_auc": {"value": oracle_value},
                     "delta_auc_vs_velocity": None, "beats_velocity": None})
    return rows


def summarize_regression(raw, r2_gate, n_boot, boot_seed):
    """Count-regression ladder rows on log1p(sold count)."""
    t = raw["t"]
    groups = raw["groups"]
    base_pooled = pooled(raw["oof_reg"][BASELINE_REGRESSOR])
    rows = []
    for rung in raw["regression_rungs"]:
        name = rung["name"]
        cells = raw["cells_reg"][name]
        rung_pooled = pooled(raw["oof_reg"][name])
        deviance_folds = None
        if rung.get("count_model"):
            deviance_folds = mean_sd(cells["deviance"])
        row = {"name": name, "label": rung["label"], "constant": rung["constant"],
               "mse_folds": mean_sd(cells["mse"]), "mae_folds": mean_sd(cells["mae"]),
               "r2_folds": mean_sd(cells["r2"]), "deviance_folds": deviance_folds,
               "pooled_r2": float(r2_score(t, rung_pooled)),
               "fit_seconds": float(raw["seconds_reg"][name])}
        if rung["constant"] or name == BASELINE_REGRESSOR:
            row["delta_r2_vs_velocity"] = None
            row["beats_velocity"] = None
        else:
            delta = row["pooled_r2"] - float(r2_score(t, base_pooled))
            delta_ci = paired_group_bootstrap_delta(
                t, rung_pooled, base_pooled, groups, r2_or_none, n_boot, boot_seed)
            row["delta_r2_vs_velocity"] = {"value": delta, **delta_ci}
            if r2_gate is None:
                row["beats_velocity"] = None
            else:
                row["beats_velocity"] = beats_baseline(
                    delta, delta_ci["ci_low"], delta_ci["ci_high"], r2_gate)
        rows.append(row)
    if raw["oracle_cells"] is not None:
        rows.append({"name": "oracle", "label": "ORACLE (planted Poisson rate)",
                     "constant": False,
                     "mse_folds": mean_sd(raw["oracle_cells"]["mse"]),
                     "mae_folds": mean_sd(raw["oracle_cells"]["mae"]),
                     "r2_folds": mean_sd(raw["oracle_cells"]["r2"]),
                     "deviance_folds": mean_sd(raw["oracle_cells"]["deviance"]),
                     "pooled_r2": float(r2_score(t, raw["oracle_log"])),
                     "fit_seconds": None,
                     "delta_r2_vs_velocity": None, "beats_velocity": None})
    return rows


def buy_list_kpi(raw, n_boot, boot_seed, m=DEFAULT_M, h=DEFAULT_H):
    """Headline KPI block: parked-capital rate at t*, model vs velocity.

    Formula: parked = FP / (TP + FP) among cards with pooled out-of-fold
    P(liquid) >= t*, t* = (m + h) / (2m + h) = 2/3 at m = h = 0.10.
    Delta = parked(velocity) - parked(model): positive means the model's
    buy list parks less capital. Paired group bootstrap (same set groups
    for both lists). Baseline: the velocity logistic, same folds, same t*.
    "Buy everything" (parked = 1 - base rate) is context only. Failure
    mode: the threshold is set from costs, not tuned on test data; if the
    probabilities are miscalibrated a fixed t* flags the wrong number of
    cards, which Brier in the ladder table shows.
    """
    y = raw["y"]
    groups = raw["groups"]
    t_star = cost_threshold(m, h)
    model_probs = pooled(raw["oof_class"][HEADLINE_MODEL])
    velocity_probs = pooled(raw["oof_class"][BASELINE_CLASSIFIER])

    def parked_metric(y_b, probs_b):
        return parked_capital_rate(y_b, probs_b, t_star)

    model_confusion = confusion_at_threshold(y, model_probs, t_star)
    velocity_confusion = confusion_at_threshold(y, velocity_probs, t_star)
    model_parked = parked_capital_rate(y, model_probs, t_star)
    velocity_parked = parked_capital_rate(y, velocity_probs, t_star)
    if model_parked is None or velocity_parked is None:
        delta_value = None
    else:
        delta_value = velocity_parked - model_parked
    delta_ci = paired_group_bootstrap_delta(
        y, velocity_probs, model_probs, groups, parked_metric, n_boot, boot_seed)
    auc_delta = float(roc_auc_score(y, model_probs) - roc_auc_score(y, velocity_probs))
    auc_ci = paired_group_bootstrap_delta(
        y, model_probs, velocity_probs, groups, auc_or_none, n_boot, boot_seed)
    block = {"m": m, "h": h, "t_star": t_star, "model": HEADLINE_MODEL,
             "baseline": BASELINE_CLASSIFIER,
             "model_parked": model_parked, "velocity_parked": velocity_parked,
             "model_confusion": model_confusion,
             "velocity_confusion": velocity_confusion,
             "delta_parked": {"value": delta_value, **delta_ci},
             "delta_auc": {"value": auc_delta, **auc_ci},
             "auc_beats_velocity": beats_baseline(auc_delta, auc_ci["ci_low"],
                                                  auc_ci["ci_high"], SHIP_MIN_DELTA),
             "buy_everything_parked": float(1.0 - y.mean())}
    if raw["oracle_prob"] is not None:
        block["oracle_parked"] = parked_capital_rate(y, raw["oracle_prob"], t_star)
        block["oracle_confusion"] = confusion_at_threshold(y, raw["oracle_prob"], t_star)
    return block


def within_rung_residuals(values):
    """Each fold value minus its own rung's mean (for a pooled noise sd)."""
    arr = np.asarray(values, dtype=float)
    centre = float(arr.mean())
    residuals = []
    for value in arr:
        residuals.append(float(value) - centre)
    return residuals


def shuffled_summary(raw_shuffled):
    """Fold-mean AUC / R^2 per NON-constant rung under shuffled labels,
    plus every fold value's deviation from its own rung's mean (the noise
    the band is set from; rung means differ for reasons that are not fold
    noise, e.g. the Poisson GLM's log1p mapping under shuffled counts)."""
    class_rows = {}
    class_residuals = []
    for rung in raw_shuffled["classification_rungs"]:
        if rung["constant"]:
            continue
        name = rung["name"]
        values = raw_shuffled["cells_class"][name]["auc"]
        class_rows[name] = mean_sd(values)
        residuals = within_rung_residuals(values)
        for value in residuals:
            class_residuals.append(value)
    reg_rows = {}
    reg_residuals = []
    for rung in raw_shuffled["regression_rungs"]:
        if rung["constant"]:
            continue
        name = rung["name"]
        values = raw_shuffled["cells_reg"][name]["r2"]
        reg_rows[name] = mean_sd(values)
        residuals = within_rung_residuals(values)
        for value in residuals:
            reg_residuals.append(value)
    return {"auc_by_rung": class_rows, "r2_by_rung": reg_rows,
            "auc_residuals": class_residuals, "n_auc_rungs": len(class_rows),
            "r2_residuals": reg_residuals, "n_r2_rungs": len(reg_rows)}


def pooled_within_sd(residuals, n_rungs):
    """sqrt(sum of squared within-rung residuals / (cells - rungs))."""
    arr = np.asarray(residuals, dtype=float)
    degrees = len(arr) - n_rungs
    return float(np.sqrt(np.sum(arr ** 2) / degrees))


def band_check(shuffled, band):
    """Which non-constant rungs fall OUTSIDE the pre-registered leakage band.

    Rule: fold-mean shuffled AUC must lie in [band.auc_low, band.auc_high]
    and fold-mean shuffled R^2 must be <= band.r2_high. Returns lists of
    violating rung names (empty = pass). Failure mode: a leak that is
    weaker than the band's half-width would pass; the band is set to 3 sd
    of single-fold null AUCs, and the fold mean is far tighter than that,
    so leak_detection measures the limit (results/leak_detection.json).
    """
    auc_violations = []
    names = list(shuffled["auc_by_rung"].keys())
    for name in names:
        value = shuffled["auc_by_rung"][name]["mean"]
        if value < band["auc_low"] or value > band["auc_high"]:
            auc_violations.append(name)
    r2_violations = []
    reg_names = list(shuffled["r2_by_rung"].keys())
    for name in reg_names:
        value = shuffled["r2_by_rung"][name]["mean"]
        if value > band["r2_high"]:
            r2_violations.append(name)
    return {"auc_outside_band": auc_violations, "r2_above_bound": r2_violations,
            "passes": len(auc_violations) == 0 and len(r2_violations) == 0}


def oracle_bound_violations(class_rows, reg_rows, tolerance=ORACLE_AUC_TOLERANCE):
    """Rungs that beat the planted truth by more than the tolerance.

    Rule: fold-mean AUC(rung) <= fold-mean AUC(oracle) + tolerance;
    fold-mean R^2(rung) <= fold-mean R^2(oracle) + tolerance; fold-mean
    Poisson deviance(rung) >= oracle - tolerance. A model "beating" the
    planted truth by more than noise is a leak. Failure mode: none of these
    can catch a leak smaller than the tolerance.
    """
    oracle_class = None
    for row in class_rows:
        if row["name"] == "oracle":
            oracle_class = row
    oracle_reg = None
    for row in reg_rows:
        if row["name"] == "oracle":
            oracle_reg = row
    violations = []
    if oracle_class is not None:
        bound = oracle_class["auc_folds"]["mean"] + tolerance
        for row in class_rows:
            if row["name"] == "oracle" or row["constant"]:
                continue
            if row["auc_folds"]["mean"] > bound:
                violations.append(row["name"] + ":auc")
    if oracle_reg is not None:
        r2_bound = oracle_reg["r2_folds"]["mean"] + tolerance
        deviance_floor = oracle_reg["deviance_folds"]["mean"] - tolerance
        for row in reg_rows:
            if row["name"] == "oracle":
                continue
            if row["r2_folds"]["mean"] > r2_bound:
                violations.append(row["name"] + ":r2")
            deviance = row["deviance_folds"]
            if deviance is not None and deviance["mean"] < deviance_floor:
                violations.append(row["name"] + ":deviance")
    return violations


def glm_oracle_deviance_gap(reg_rows):
    """Fold-mean Poisson deviance of the GLM minus the oracle's."""
    glm = None
    oracle = None
    for row in reg_rows:
        if row["name"] == "poisson_glm":
            glm = row
        if row["name"] == "oracle":
            oracle = row
    if glm is None or oracle is None:
        return None
    return glm["deviance_folds"]["mean"] - oracle["deviance_folds"]["mean"]


def dataset_block(frame, truth):
    y = frame["y_liquid"].to_numpy(dtype=int)
    counts = frame["n_sold_outcome"].to_numpy(dtype=float)
    return {"n_rows": int(len(frame)), "n_groups": int(frame["group_id"].nunique()),
            "base_rate": float(y.mean()),
            "planted_base_rate": float(np.mean(truth["p_liquid"])),
            "share_zero_sold": float(np.mean(counts == 0)),
            "cluster_signal": float(truth["cluster_signal"]),
            "outcome_mode": truth["outcome_mode"]}


def run_signal(cluster_signal, data_seed, cv_seeds, n_boot, r2_gate, band,
               verbose=True):
    """Both ladders + shuffled control + KPI for one planted signal."""
    frame, manifest, truth = synth.illustrative_preset(cluster_signal, seed=data_seed)
    feature_cols = manifest["feature_cols"]
    baseline_cols = manifest["baseline_cols"]
    started = time.perf_counter()
    raw = cross_validate_ladder(frame, feature_cols, baseline_cols, cv_seeds, truth=truth)
    if verbose:
        print("[ladder] signal %.1f: real-label ladders %.1f s"
              % (cluster_signal, time.perf_counter() - started))
    started = time.perf_counter()
    raw_shuffled = cross_validate_ladder(frame, feature_cols, baseline_cols, cv_seeds,
                                         shuffle=True)
    if verbose:
        print("[ladder] signal %.1f: shuffled-label control %.1f s"
              % (cluster_signal, time.perf_counter() - started))
    class_rows = summarize_classification(raw, n_boot, BOOT_SEED)
    reg_rows = summarize_regression(raw, r2_gate, n_boot, BOOT_SEED)
    shuffled = shuffled_summary(raw_shuffled)
    velocity_r2_cells = raw["cells_reg"][BASELINE_REGRESSOR]["r2"]
    block = {
        "dataset": dataset_block(frame, truth),
        "folds": fold_checks(raw),
        "shuffled_folds": fold_checks(raw_shuffled),
        "classification": class_rows,
        "regression": reg_rows,
        "kpi": buy_list_kpi(raw, n_boot, BOOT_SEED),
        "leakage": {"auc_by_rung": shuffled["auc_by_rung"],
                    "r2_by_rung": shuffled["r2_by_rung"]},
        "oracle_violations": oracle_bound_violations(class_rows, reg_rows),
        "glm_oracle_deviance_gap": glm_oracle_deviance_gap(reg_rows),
        "velocity_ols_r2_fold_sd": mean_sd(velocity_r2_cells)["sd"],
    }
    if band is not None:
        block["leakage"]["band_check"] = band_check(shuffled, band)
    return block, shuffled


# ---------------------------------------------------------------- trial / frozen

def round_up(value, decimals=3):
    factor = 10.0 ** decimals
    return float(np.ceil(value * factor) / factor)


def band_from_trial(trial_signals):
    """Leakage band and R^2 gate from the trial seeds (written down BEFORE
    the frozen run).

    auc band = 0.5 +/- 3 x the pooled within-rung sd of single-fold shuffled
    AUCs (every non-constant rung, both signals), rounded up to 3 decimals;
    r2_high = 0 + 3 x the same pooled sd for shuffled R^2, rounded up ("R^2
    at or below about 0"). R^2 gate = the larger of the two signals'
    velocity-OLS fold sd on real labels, rounded up: a rung must beat
    velocity OLS by more than one fold-to-fold wobble. Failure mode: 3 x a
    single-fold sd is generous for a 50-cell fold mean, so only leaks larger
    than the half-width are caught (the planted label leak in the tests
    scores AUC 1; leak_detection records the weakest leak caught). A future
    registration should use the sd of the fold MEAN across trial seeds.
    """
    auc_residuals = []
    r2_residuals = []
    n_auc_rungs = 0
    n_r2_rungs = 0
    gate_candidates = []
    signal_keys = sorted(trial_signals.keys())
    for key in signal_keys:
        entry = trial_signals[key]
        for value in entry["shuffled_auc_residuals"]:
            auc_residuals.append(value)
        for value in entry["shuffled_r2_residuals"]:
            r2_residuals.append(value)
        n_auc_rungs += entry["shuffled_n_auc_rungs"]
        n_r2_rungs += entry["shuffled_n_r2_rungs"]
        gate_candidates.append(entry["velocity_ols_r2_fold_sd"])
    auc_sd = pooled_within_sd(auc_residuals, n_auc_rungs)
    r2_sd = pooled_within_sd(r2_residuals, n_r2_rungs)
    half_width = round_up(BAND_SD_MULTIPLIER * auc_sd)
    band = {"auc_low": round(0.5 - half_width, 3), "auc_high": round(0.5 + half_width, 3),
            "auc_half_width": half_width, "r2_high": round_up(BAND_SD_MULTIPLIER * r2_sd),
            "single_fold_auc_sd": auc_sd, "single_fold_r2_sd": r2_sd,
            "n_auc_cells": len(auc_residuals), "n_r2_cells": len(r2_residuals)}
    gate = round_up(max(gate_candidates))
    return band, gate


def signal_key(cluster_signal):
    return "%.1f" % cluster_signal


def noisy_leak_rung(noise_sd):
    """A leakage-CONTROL rung: full features plus the label + N(0, sd)."""
    return {"name": "leak_sd_%g" % noise_sd, "label": "LEAK CONTROL (label + noise)",
            "design": "features_plus_noisy_label", "make": make_l2_logistic,
            "constant": False, "leak_noise_sd": float(noise_sd)}


def honest_shuffled_auc_range(frozen):
    """(lowest, highest) shuffled fold-mean AUC over every non-constant rung
    of the frozen run, both signals: the spread the band has to guard."""
    lowest = None
    highest = None
    signal_keys = sorted(frozen["signals"].keys())
    for key in signal_keys:
        by_rung = frozen["signals"][key]["leakage"]["auc_by_rung"]
        names = sorted(by_rung.keys())
        for name in names:
            value = by_rung[name]["mean"]
            if lowest is None or value < lowest:
                lowest = value
            if highest is None or value > highest:
                highest = value
    return lowest, highest


def leak_detection(band, noise_sds=LEAK_NOISE_SDS, n_rows=LEAK_N_ROWS,
                   data_seed=LEAK_DATA_SEED, cv_seeds=LEAK_CV_SEEDS):
    """The registered band's measured detection limit (a few seconds).

    Rule: under the row-level label shuffle, one L2-logistic rung per noise
    sd sees the full features plus the SHUFFLED label + N(0, sd); its
    shuffled fold-mean AUC is judged by band_check against the registered
    band, unchanged. largest_missed = the strongest leak (highest fold-mean
    AUC) still inside the band; smallest_caught = the weakest leak outside
    it, so the detection limit lies between the two. Data: a threshold-mode
    generate_supervised frame (n = 3,000, signal 0.9). Failure mode: the
    limit is measured for one leak shape (one noisy copy of the label);
    a leak spread thinly over many features could sit at a different AUC.
    """
    frame, manifest, _truth = synth.generate_supervised(
        n_rows=n_rows, n_clusters=3, separation=2.5, seed=data_seed, cluster_signal=0.9,
        outcome_mode="threshold")
    rungs = []
    for noise_sd in noise_sds:
        rungs.append(noisy_leak_rung(noise_sd))
    raw = cross_validate_ladder(frame, manifest["feature_cols"], manifest["baseline_cols"],
                                cv_seeds, shuffle=True, classification_rungs=rungs,
                                regression_rungs=[])
    summary = shuffled_summary(raw)
    check = band_check(summary, band)
    outside = set(check["auc_outside_band"])
    rows = []
    largest_missed = None
    smallest_caught = None
    for rung in rungs:
        name = rung["name"]
        fold_mean = summary["auc_by_rung"][name]["mean"]
        caught = name in outside
        row = {"noise_sd": rung["leak_noise_sd"], "fold_mean_auc": fold_mean,
               "fold_sd_auc": summary["auc_by_rung"][name]["sd"], "caught": caught}
        rows.append(row)
        if caught:
            if smallest_caught is None or fold_mean < smallest_caught["fold_mean_auc"]:
                smallest_caught = row
        elif largest_missed is None or fold_mean > largest_missed["fold_mean_auc"]:
            largest_missed = row
    return {"label": "SYNTHETIC leakage-band detection limit (control; the band is "
                     "the registered one, unchanged)",
            "band": {"auc_low": band["auc_low"], "auc_high": band["auc_high"],
                     "auc_half_width": band["auc_half_width"]},
            "n_rows": n_rows, "data_seed": data_seed, "cv_seeds": list(cv_seeds),
            "n_cells": n_folds_cells(raw), "rows": rows,
            "largest_missed": largest_missed, "smallest_caught": smallest_caught}


def n_folds_cells(raw):
    """Fold-seed cells per rung of a cross_validate_ladder result."""
    return raw["n_folds"] * len(raw["cv_seeds"])


def run_trial(n_boot=200, verbose=True):
    """Trial seeds on a DIFFERENT data draw; only sets the band and gate."""
    signals = {}
    for cluster_signal in SIGNALS:
        block, shuffled = run_signal(cluster_signal, TRIAL_DATA_SEED, TRIAL_CV_SEEDS,
                                     n_boot, None, None, verbose=verbose)
        block["shuffled_auc_residuals"] = shuffled["auc_residuals"]
        block["shuffled_r2_residuals"] = shuffled["r2_residuals"]
        block["shuffled_n_auc_rungs"] = shuffled["n_auc_rungs"]
        block["shuffled_n_r2_rungs"] = shuffled["n_r2_rungs"]
        signals[signal_key(cluster_signal)] = block
    band, gate = band_from_trial(signals)
    signal_keys = list(signals.keys())
    for key in signal_keys:
        signals[key].pop("shuffled_auc_residuals")
        signals[key].pop("shuffled_r2_residuals")
        signals[key].pop("shuffled_n_auc_rungs")
        signals[key].pop("shuffled_n_r2_rungs")
    return {"label": "SYNTHETIC trial run (sets the band and gate only; not results)",
            "data_seed": TRIAL_DATA_SEED, "cv_seeds": list(TRIAL_CV_SEEDS),
            "n_boot": n_boot, "band": band, "r2_gate": gate, "signals": signals}


def sha_prefix(path):
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return digest[:SHA_PREFIX_LEN]


def run_frozen(trial, predictions_path, n_boot=B_BOOT, verbose=True):
    """The frozen run: preset data, 10 CV seeds, band and gate from trial."""
    from cardsegments import ladder_predictions
    band = trial["band"]
    gate = trial["r2_gate"]
    signals = {}
    for cluster_signal in SIGNALS:
        block, _shuffled = run_signal(cluster_signal, synth.ILLUSTRATIVE_SEED,
                                      FROZEN_CV_SEEDS, n_boot, gate, band, verbose=verbose)
        signals[signal_key(cluster_signal)] = block
    frozen = {"label": "SYNTHETIC - illustrative preset (round values; not the "
                       "private cohort)",
              "preset": {"n_rows": synth.ILLUSTRATIVE_N_ROWS,
                         "n_groups": 3 * synth.ILLUSTRATIVE_GROUPS_PER_CLUSTER,
                         "target_base_rate": synth.THRESHOLD_TARGET_BASE_RATE,
                         "liquid_min_sold": synth.LIQUID_MIN_SOLD,
                         "data_seed": synth.ILLUSTRATIVE_SEED},
              "cv_seeds": list(FROZEN_CV_SEEDS), "n_folds": N_FOLDS, "n_boot": n_boot,
              "band": band, "r2_gate": gate, "ship_min_delta_auc": SHIP_MIN_DELTA,
              "predictions_file": os.path.basename(predictions_path),
              "predictions_sha256_prefix": sha_prefix(predictions_path),
              "sklearn_version": sklearn_version,
              "signals": signals}
    frozen["verdicts"] = ladder_predictions.judge_all(frozen)
    return frozen


# ---------------------------------------------------------------- JSON

def round_sig(value, digits=10):
    if value == 0 or not np.isfinite(value):
        return float(value)
    return float(f"{value:.{digits}g}")


def clean_for_json(value):
    """numpy -> python, floats rounded to 10 significant digits."""
    if isinstance(value, dict):
        cleaned = {}
        keys = list(value.keys())
        for key in keys:
            cleaned[str(key)] = clean_for_json(value[key])
        return cleaned
    if isinstance(value, (list, tuple)):
        cleaned_list = []
        for item in value:
            cleaned_list.append(clean_for_json(item))
        return cleaned_list
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round_sig(float(value))
    return value


def write_json(payload, path):
    cleaned = clean_for_json(payload)
    text = json.dumps(cleaned, indent=1, sort_keys=True)
    with open(path, "w") as handle:
        handle.write(text + "\n")
    return path


def results_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, "results")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic model ladders (heavy; not CI).")
    parser.add_argument("stage", choices=["trial", "preregister", "frozen", "leakdetect"])
    args = parser.parse_args(argv)
    out_dir = results_dir()
    os.makedirs(out_dir, exist_ok=True)
    trial_path = os.path.join(out_dir, TRIAL_FILENAME)
    predictions_path = os.path.join(out_dir, PREDICTIONS_FILENAME)
    frozen_path = os.path.join(out_dir, FROZEN_FILENAME)
    if args.stage == "trial":
        trial = run_trial()
        print("wrote", write_json(trial, trial_path))
        return
    if args.stage == "preregister":
        from cardsegments import ladder_predictions
        if os.path.exists(predictions_path):
            raise SystemExit("predictions file exists; it is immutable once written")
        with open(trial_path) as handle:
            trial = json.load(handle)
        text = ladder_predictions.predictions_markdown(trial)
        with open(predictions_path, "w") as handle:
            handle.write(text)
        print("wrote", predictions_path)
        return
    if args.stage == "leakdetect":
        with open(frozen_path) as handle:
            frozen = json.load(handle)
        leak = leak_detection(frozen["band"])
        print("wrote", write_json(leak, os.path.join(out_dir, LEAK_FILENAME)))
        return
    if not os.path.exists(predictions_path):
        raise SystemExit("pre-register first: the frozen run needs the predictions file")
    with open(trial_path) as handle:
        trial = json.load(handle)
    frozen = run_frozen(trial, predictions_path)
    print("wrote", write_json(frozen, frozen_path))


if __name__ == "__main__":
    main()
