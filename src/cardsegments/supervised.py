"""Leak-audited supervised validation of the segmentation.

Protocol (fixed ex ante):
  - StratifiedGroupKFold (groups = group_id, stratified on y), 5 folds x 10
    seeds; scaler AND imputer refit INSIDE each training fold.
  - Models: LogisticRegression on the full features (bundle key
    "primary_logistic"; scikit-learn defaults, which are an L2 penalty with
    C = 1, max_iter=1000 - NOT an unpenalized "plain" logistic; the model
    ladder in ladder.py adds the unpenalized rung beside it) and
    GradientBoostingClassifier (fixed documented defaults, no test-fold
    tuning). Baselines: majority, era-only logistic, trailing-velocity
    logistic.
  - CLUSTER VALUE-ADD: clusters are RE-FIT inside each training fold on that
    fold's training rows with complete cluster features (k frozen at k*,
    a mild selection leak disclosed in Design Decisions); test-fold cards
    with complete features get the fold-fitted assignment, all others get an
    explicit "no_cluster" category. The cluster one-hots INCLUDE no_cluster,
    so the nested comparison runs on the FULL cohort.
  - NESTED GATE: delta AUC of (era + velocity + cluster) over (era +
    velocity); pooled out-of-fold predictions; paired GROUP-level bootstrap
    (B=1000) for the delta-AUC CI; fold x seed spread reported. The
    predictive claim ships ONLY if the 95% CI excludes 0 AND mean delta AUC
    >= +0.01; otherwise the finding is a null, stated plainly.
  - No t-tests on CV folds (Nadeau-Bengio is the rejected alternative).
  - Cost threshold: t* = C_FP / (C_FP + C_FN), C_FP = m + h, C_FN = m;
    defaults m = h = 0.10 give t* = 2/3; the threshold-cost curve is
    computed for three (m, h) settings so no conclusion hinges on one.
  - ALL CIs here: group(set)-level bootstrap.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from cardsegments.cluster import fit_gmm, fit_kmeans

ERA_ORDER = ["vintage", "mid", "modern"]
COST_SETTINGS_DEFAULT = ((0.10, 0.10), (0.10, 0.05), (0.10, 0.20))
SHIP_MIN_DELTA = 0.01

# Honest display names for the bundle keys. The key "primary_logistic" stays
# so the committed aggregates bundle still renders; its label says what the
# model is (DECISIONS.md, 2026-11-04).
MODEL_LABELS = {
    "majority": "majority (train-fold base rate)",
    "era_only": "era-only logistic",
    "velocity": "velocity logistic (trailing sales)",
    "era_velocity": "era + velocity logistic",
    "era_velocity_cluster": "era + velocity + cluster logistic",
    "primary_logistic": "full logistic, L2 C=1 (key primary_logistic)",
    "gbc": "gradient boosting (gbc, defaults)",
}


def cost_threshold(m, h):
    """t* = C_FP/(C_FP+C_FN) with C_FP = m+h (purchase + holding cost of a
    card that sits) and C_FN = m (margin missed by passing)."""
    c_fp = m + h
    c_fn = m
    return c_fp / (c_fp + c_fn)


def era_onehots(era_values):
    n_rows = len(era_values)
    matrix = np.zeros((n_rows, len(ERA_ORDER)))
    for era_position in range(len(ERA_ORDER)):
        matrix[:, era_position] = (era_values == ERA_ORDER[era_position])
    return matrix.astype(float)


def fit_impute_scale(train_block):
    """Median-impute + z-scale stats from TRAINING rows only."""
    medians = np.nanmedian(train_block, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    filled = train_block.copy()
    n_cols = filled.shape[1]
    for col in range(n_cols):
        column = filled[:, col]
        column[np.isnan(column)] = medians[col]
    means = filled.mean(axis=0)
    stds = filled.std(axis=0, ddof=0)
    stds = np.where(stds == 0, 1.0, stds)
    return {"medians": medians, "means": means, "stds": stds}


def apply_impute_scale(block, stats):
    filled = block.copy()
    n_cols = filled.shape[1]
    for col in range(n_cols):
        column = filled[:, col]
        column[np.isnan(column)] = stats["medians"][col]
    return (filled - stats["means"]) / stats["stds"]


def fold_cluster_onehots(cluster_block, train_idx, test_idx, complete_mask,
                         cluster_spec, fit_seed=42):
    """Re-fit clusters on the fold's training rows with complete cluster
    features; assign complete test rows; everything else -> no_cluster.

    Returns (train_onehots, test_onehots) with k+1 columns (last=no_cluster).
    """
    k = int(cluster_spec["k"])
    method = cluster_spec["method"]
    cov_type = cluster_spec.get("cov_type", "")
    train_complete = train_idx[complete_mask[train_idx]]
    test_complete = test_idx[complete_mask[test_idx]]
    block_train = cluster_block[train_complete]
    means = block_train.mean(axis=0)
    stds = block_train.std(axis=0, ddof=0)
    stds = np.where(stds == 0, 1.0, stds)
    X_train = (block_train - means) / stds
    if method == "kmeans":
        model = fit_kmeans(X_train, k, fit_seed)
    else:
        model = fit_gmm(X_train, k, cov_type, fit_seed)

    def onehots_for(idx, complete_rows):
        matrix = np.zeros((len(idx), k + 1))
        position_of = {}
        for position in range(len(idx)):
            position_of[int(idx[position])] = position
        if len(complete_rows) > 0:
            X_rows = (cluster_block[complete_rows] - means) / stds
            assigned = model.predict(X_rows)
            for row_position in range(len(complete_rows)):
                row_index = int(complete_rows[row_position])
                matrix[position_of[row_index], int(assigned[row_position])] = 1.0
        for position in range(len(idx)):
            if matrix[position].sum() == 0:
                matrix[position, k] = 1.0  # no_cluster
        return matrix

    train_onehots = onehots_for(train_idx, train_complete)
    test_onehots = onehots_for(test_idx, test_complete)
    return train_onehots, test_onehots


def confusion_at_threshold(y, probs, threshold):
    predicted = (probs >= threshold).astype(int)
    tp = int(np.sum((predicted == 1) & (y == 1)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = None
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = None
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
            "tn": tn, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1, "n": int(len(y))}


def calibration_bins(y, probs, min_bin_n=50, n_bins=10):
    """Quantile bins accumulated left-to-right until every bin has
    n >= min_bin_n.

    Raw quantile intervals are accumulated into the CURRENT bin until it
    reaches min_bin_n, then a new bin starts; a trailing short bin merges
    into its predecessor. (The earlier version tested only the current raw
    interval's width, so a run of small intervals chain-merged into one
    bin.)
    """
    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_y = y[order]
    n_rows = len(y)
    edges = []
    for b in range(1, n_bins):
        edges.append(int(round(b * n_rows / n_bins)))
    edges.append(n_rows)
    raw_intervals = []
    start = 0
    for edge in edges:
        if edge <= start:
            continue
        raw_intervals.append((start, edge))
        start = edge
    merged = []
    current_start = None
    for interval in raw_intervals:
        if current_start is None:
            current_start = interval[0]
        current_end = interval[1]
        if current_end - current_start >= min_bin_n:
            merged.append((current_start, current_end))
            current_start = None
    if current_start is not None:
        # trailing short bin: merge into the predecessor when there is one
        if merged:
            previous = merged.pop()
            merged.append((previous[0], n_rows))
        else:
            merged.append((current_start, n_rows))
    rows = []
    for interval in merged:
        low, high = interval
        bin_probs = sorted_probs[low:high]
        bin_y = sorted_y[low:high]
        rows.append({"n": int(high - low),
                     "mean_predicted": float(bin_probs.mean()),
                     "observed_rate": float(bin_y.mean())})
    return rows


def threshold_cost_curve(y, probs, m, h, grid=None):
    if grid is None:
        grid = np.arange(0.02, 0.99, 0.02)
    c_fp = m + h
    c_fn = m
    n_rows = len(y)
    points = []
    for threshold in grid:
        predicted = (probs >= threshold).astype(int)
        fp = int(np.sum((predicted == 1) & (y == 0)))
        fn = int(np.sum((predicted == 0) & (y == 1)))
        cost = (c_fp * fp + c_fn * fn) / n_rows
        points.append({"threshold": float(threshold), "cost": float(cost)})
    return {"m": m, "h": h, "t_star": cost_threshold(m, h), "points": points}


def group_index_lists(groups):
    """Precompute each unique group's row indices ONCE (bootstrap loops
    then only concatenate arrays, which keeps B=1000 runs fast)."""
    codes, unique_groups = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.searchsorted(sorted_codes, np.arange(len(unique_groups) + 1))
    index_lists = []
    for position in range(len(unique_groups)):
        index_lists.append(order[boundaries[position]:boundaries[position + 1]])
    return index_lists


def group_bootstrap_indices(groups, rng, index_lists=None):
    """Resample unique groups with replacement; rows concatenated."""
    if index_lists is None:
        index_lists = group_index_lists(groups)
    n_groups = len(index_lists)
    sampled = rng.integers(0, n_groups, size=n_groups)
    chosen_lists = []
    for group_position in sampled:
        chosen_lists.append(index_lists[group_position])
    return np.concatenate(chosen_lists)


def paired_group_bootstrap_delta_auc(y, probs_full, probs_base, groups,
                                     n_boot=1000, seed=2000):
    """Paired group-level bootstrap for AUC(full) - AUC(base)."""
    rng = np.random.default_rng(seed)
    index_lists = group_index_lists(groups)
    deltas = []
    n_skipped = 0
    for _b in range(n_boot):
        idx = group_bootstrap_indices(groups, rng, index_lists)
        y_b = y[idx]
        if y_b.min() == y_b.max():
            n_skipped += 1
            continue
        auc_full = roc_auc_score(y_b, probs_full[idx])
        auc_base = roc_auc_score(y_b, probs_base[idx])
        deltas.append(auc_full - auc_base)
    delta_arr = np.array(deltas)
    return {"mean": float(delta_arr.mean()),
            "ci_low": float(np.percentile(delta_arr, 2.5)),
            "ci_high": float(np.percentile(delta_arr, 97.5)),
            "n_boot_valid": int(len(delta_arr)), "n_boot_skipped": n_skipped}


def group_bootstrap_auc_ci(y, probs, groups, n_boot=1000, seed=2000):
    rng = np.random.default_rng(seed)
    index_lists = group_index_lists(groups)
    values = []
    for _b in range(n_boot):
        idx = group_bootstrap_indices(groups, rng, index_lists)
        y_b = y[idx]
        if y_b.min() == y_b.max():
            continue
        values.append(roc_auc_score(y_b, probs[idx]))
    arr = np.array(values)
    return {"ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "n_boot_valid": int(len(arr))}


def coefficient_ci_table(feature_block, y, groups, feature_names,
                         n_boot=200, seed=3000):
    """Group-bootstrap CIs for the full logistic's coefficients (L2, C = 1,
    the "primary_logistic" key; standardized scale; impute+scale refit
    inside every resample)."""
    stats = fit_impute_scale(feature_block.copy())
    X_full = apply_impute_scale(feature_block.copy(), stats)
    full_model = LogisticRegression(max_iter=1000)
    full_model.fit(X_full, y)
    full_coefs = full_model.coef_[0]
    rng = np.random.default_rng(seed)
    index_lists = group_index_lists(groups)
    boot_coefs = []
    for _b in range(n_boot):
        idx = group_bootstrap_indices(groups, rng, index_lists)
        y_b = y[idx]
        if y_b.min() == y_b.max():
            continue
        block_b = feature_block[idx].copy()
        stats_b = fit_impute_scale(block_b.copy())
        X_b = apply_impute_scale(block_b, stats_b)
        model_b = LogisticRegression(max_iter=1000)
        model_b.fit(X_b, y_b)
        boot_coefs.append(model_b.coef_[0])
    boot_arr = np.array(boot_coefs)
    rows = []
    n_features = len(feature_names)
    for position in range(n_features):
        rows.append({"feature": feature_names[position],
                     "coef": float(full_coefs[position]),
                     "ci_low": float(np.percentile(boot_arr[:, position], 2.5)),
                     "ci_high": float(np.percentile(boot_arr[:, position], 97.5)),
                     "n_boot_valid": int(len(boot_arr))})
    return rows


def forward_regression(feature_block, fwd_values, groups, n_folds=5):
    """Secondary target: out-of-fold R^2 on forward log price change, on the
    ELIGIBLE subset only. The eligibility screen selects liquid cards, so
    this is reported with that caveat, never as a population claim."""
    eligible = ~np.isnan(fwd_values)
    n_eligible = int(eligible.sum())
    result = {"n_eligible": n_eligible}
    if n_eligible < 50:
        result["oof_r2"] = None
        result["note"] = "fewer than 50 eligible rows; not fit"
        return result
    block = feature_block[eligible]
    y = fwd_values[eligible]
    grp = groups[eligible]
    predictions = np.full(n_eligible, np.nan)
    splitter = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in splitter.split(block, y, groups=grp):
        stats = fit_impute_scale(block[train_idx].copy())
        X_train = apply_impute_scale(block[train_idx].copy(), stats)
        X_test = apply_impute_scale(block[test_idx].copy(), stats)
        model = LinearRegression()
        model.fit(X_train, y[train_idx])
        predictions[test_idx] = model.predict(X_test)
    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    result["oof_r2"] = 1.0 - ss_res / ss_tot
    return result


def run_supervised(frame, feature_cols, baseline_cols, cluster_feature_cols,
                   cluster_spec, n_folds=5, n_seeds=10, seed_base=0,
                   fit_seed=42, b_group_boot=1000, group_boot_seed=2000,
                   b_coef_boot=200, coef_boot_seed=3000,
                   cost_settings=COST_SETTINGS_DEFAULT,
                   calibration_min_bin_n=50, outcome_col="y_liquid",
                   fwd_col="fwd_log_price_change", verbose=False):
    """The full supervised protocol on one adapter-schema frame.

    Returns an aggregates-only results dict (bundle-ready).
    """
    y = frame[outcome_col].to_numpy(dtype=int)
    groups = frame["group_id"].to_numpy()
    era_values = frame["era_label"].to_numpy()
    n_rows = len(frame)
    feature_block = frame[feature_cols].to_numpy(dtype=float)
    baseline_block = frame[baseline_cols].to_numpy(dtype=float)
    era_block = era_onehots(era_values)
    cluster_block = frame[cluster_feature_cols].to_numpy(dtype=float)
    complete_mask = ~np.isnan(cluster_block).any(axis=1)

    model_names = ["majority", "era_only", "velocity", "era_velocity",
                   "era_velocity_cluster", "primary_logistic", "gbc"]
    oof_by_seed = {}
    for name in model_names:
        oof_by_seed[name] = np.full((n_seeds, n_rows), np.nan)
    fold_deltas = []

    for seed_position in range(n_seeds):
        seed = seed_base + seed_position
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed)
        fold_number = 0
        for train_idx, test_idx in splitter.split(feature_block, y, groups):
            y_train = y[train_idx]
            feat_stats = fit_impute_scale(feature_block[train_idx].copy())
            X_feat_train = apply_impute_scale(feature_block[train_idx].copy(),
                                              feat_stats)
            X_feat_test = apply_impute_scale(feature_block[test_idx].copy(),
                                             feat_stats)
            base_stats = fit_impute_scale(baseline_block[train_idx].copy())
            X_base_train = apply_impute_scale(baseline_block[train_idx].copy(),
                                              base_stats)
            X_base_test = apply_impute_scale(baseline_block[test_idx].copy(),
                                             base_stats)
            era_train = era_block[train_idx]
            era_test = era_block[test_idx]
            cl_train, cl_test = fold_cluster_onehots(
                cluster_block, train_idx, test_idx, complete_mask,
                cluster_spec, fit_seed=fit_seed)

            designs = {
                "era_only": (era_train, era_test),
                "velocity": (X_base_train, X_base_test),
                "era_velocity": (np.column_stack([era_train, X_base_train]),
                                 np.column_stack([era_test, X_base_test])),
                "era_velocity_cluster": (
                    np.column_stack([era_train, X_base_train, cl_train]),
                    np.column_stack([era_test, X_base_test, cl_test])),
                "primary_logistic": (X_feat_train, X_feat_test),
            }
            oof_by_seed["majority"][seed_position, test_idx] = y_train.mean()
            design_names = list(designs.keys())
            for name in design_names:
                train_design, test_design = designs[name]
                model = LogisticRegression(max_iter=1000)
                model.fit(train_design, y_train)
                oof_by_seed[name][seed_position, test_idx] = \
                    model.predict_proba(test_design)[:, 1]
            gbc = GradientBoostingClassifier(random_state=fit_seed)
            gbc.fit(X_feat_train, y_train)
            oof_by_seed["gbc"][seed_position, test_idx] = \
                gbc.predict_proba(X_feat_test)[:, 1]

            y_test = y[test_idx]
            if y_test.min() != y_test.max():
                auc_full = roc_auc_score(
                    y_test, oof_by_seed["era_velocity_cluster"][seed_position,
                                                                test_idx])
                auc_base = roc_auc_score(
                    y_test, oof_by_seed["era_velocity"][seed_position, test_idx])
                fold_deltas.append({"seed": seed, "fold": fold_number,
                                    "delta_auc": float(auc_full - auc_base)})
            fold_number += 1
        if verbose:
            print("  seed %d done" % seed)

    # per-seed AUCs and deltas
    per_seed_auc = {}
    for name in model_names:
        values = []
        for seed_position in range(n_seeds):
            values.append(float(roc_auc_score(y, oof_by_seed[name][seed_position])))
        per_seed_auc[name] = values
    per_seed_delta = []
    for seed_position in range(n_seeds):
        delta = per_seed_auc["era_velocity_cluster"][seed_position] \
            - per_seed_auc["era_velocity"][seed_position]
        per_seed_delta.append(float(delta))
    mean_delta = float(np.mean(per_seed_delta))

    # pooled per-card probabilities (mean over seeds)
    pooled = {}
    for name in model_names:
        pooled[name] = oof_by_seed[name].mean(axis=0)

    delta_ci = paired_group_bootstrap_delta_auc(
        y, pooled["era_velocity_cluster"], pooled["era_velocity"], groups,
        n_boot=b_group_boot, seed=group_boot_seed)
    ships = bool(delta_ci["ci_low"] > 0 and mean_delta >= SHIP_MIN_DELTA)

    pooled_auc = {}
    for name in model_names:
        auc_value = float(roc_auc_score(y, pooled[name]))
        ci = group_bootstrap_auc_ci(y, pooled[name], groups,
                                    n_boot=b_group_boot, seed=group_boot_seed)
        pooled_auc[name] = {"auc": auc_value, **ci}

    fold_delta_values = []
    for row in fold_deltas:
        fold_delta_values.append(row["delta_auc"])
    fold_delta_arr = np.array(fold_delta_values)

    base_rate = float(y.mean())
    if base_rate <= 0.5:
        minority_label = "liquid (y=1)"
        pr_auc = float(average_precision_score(y, pooled["primary_logistic"]))
    else:
        minority_label = "sits (y=0)"
        pr_auc = float(average_precision_score(1 - y,
                                               1 - pooled["primary_logistic"]))

    default_m, default_h = cost_settings[0]
    t_star = cost_threshold(default_m, default_h)
    confusion = confusion_at_threshold(y, pooled["primary_logistic"], t_star)
    curves = []
    for m, h in cost_settings:
        curves.append(threshold_cost_curve(y, pooled["primary_logistic"], m, h))
    calibration = calibration_bins(y, pooled["primary_logistic"],
                                   min_bin_n=calibration_min_bin_n)
    coef_table = coefficient_ci_table(feature_block, y, groups, feature_cols,
                                      n_boot=b_coef_boot, seed=coef_boot_seed)
    fwd_values = frame[fwd_col].to_numpy(dtype=float)
    fwd_result = forward_regression(feature_block, fwd_values, groups,
                                    n_folds=n_folds)

    results = {
        "n_rows": int(n_rows),
        "n_groups": int(len(np.unique(groups))),
        "base_rate": base_rate,
        "n_cluster_feature_complete": int(complete_mask.sum()),
        "cluster_spec": dict(cluster_spec),
        "cluster_feature_cols": list(cluster_feature_cols),
        "feature_cols": list(feature_cols),
        "baseline_cols": list(baseline_cols),
        "n_folds": n_folds, "n_seeds": n_seeds,
        "per_seed_auc": per_seed_auc,
        "pooled_auc": pooled_auc,
        "nested_delta_auc": {
            "mean_over_seeds": mean_delta,
            "per_seed": per_seed_delta,
            "group_bootstrap": delta_ci,
            "fold_seed_spread": {
                "mean": float(fold_delta_arr.mean()),
                "min": float(fold_delta_arr.min()),
                "p5": float(np.percentile(fold_delta_arr, 5.0)),
                "n_cells": int(len(fold_delta_arr)),
            },
            "ship_rule": "95% CI excludes 0 AND mean delta AUC >= +0.01",
            "ships": ships,
        },
        "minority_class": minority_label,
        "pr_auc_minority": pr_auc,
        "confusion_at_t_star": confusion,
        "threshold_cost_curves": curves,
        "calibration_bins": calibration,
        "coefficients": coef_table,
        "forward_regression": fwd_result,
    }
    return results
