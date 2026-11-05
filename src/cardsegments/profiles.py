"""Aggregate tables over a fitted clustering: segment profiles with bootstrap
CIs, era cross-tabs and lifts, Cramer's V, robustness cells, and per-segment
trend statistics. Everything here returns AGGREGATES with each cell's n -
this is the only shape of real-data result the public repo ever carries.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from cardsegments.cluster import fit_gmm, fit_kmeans
from cardsegments.stability import hungarian_map

ERA_ORDER = ["vintage", "mid", "modern"]
PROFILE_BOOT_B = 1000
PROFILE_BOOT_SEED = 4000


def median_ci(values, n_boot=PROFILE_BOOT_B, seed=PROFILE_BOOT_SEED):
    """Median + card-level bootstrap percentile CI on non-missing values."""
    clean = values[~np.isnan(values)]
    n_clean = int(len(clean))
    if n_clean == 0:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    median = float(np.median(clean))
    rng = np.random.default_rng(seed)
    boot_medians = []
    for _b in range(n_boot):
        idx = rng.integers(0, n_clean, size=n_clean)
        boot_medians.append(float(np.median(clean[idx])))
    boot_arr = np.array(boot_medians)
    return {"median": median,
            "ci_low": float(np.percentile(boot_arr, 2.5)),
            "ci_high": float(np.percentile(boot_arr, 97.5)),
            "n": n_clean}


def global_profile(frame, profile_cols):
    """Per-feature global median, IQR and bootstrap CI (the reference the
    narrative rule compares against)."""
    stats = {}
    for column in profile_cols:
        values = frame[column].to_numpy(dtype=float)
        cell = median_ci(values)
        clean = values[~np.isnan(values)]
        if len(clean) > 0:
            q1 = float(np.percentile(clean, 25))
            q3 = float(np.percentile(clean, 75))
            cell["iqr"] = q3 - q1
        else:
            cell["iqr"] = None
        stats[column] = cell
    return stats


def cluster_profile_table(frame, labels, profile_cols):
    """Per cluster x feature: median + bootstrap CI + cell n."""
    cluster_ids = sorted(set(labels))
    rows = []
    for cluster_id in cluster_ids:
        in_cluster = labels == cluster_id
        cluster_frame = frame[in_cluster]
        row = {"cluster": int(cluster_id), "n": int(in_cluster.sum()),
               "features": {}}
        for column in profile_cols:
            values = cluster_frame[column].to_numpy(dtype=float)
            row["features"][column] = median_ci(values)
        rows.append(row)
    return rows


def era_crosstab(era_values, labels):
    """Counts and within-cluster percent shares, clusters x eras."""
    counts = pd.crosstab(pd.Series(labels, name="cluster"),
                         pd.Series(era_values, name="era_label"))
    ordered_cols = []
    for era in ERA_ORDER:
        if era in counts.columns:
            ordered_cols.append(era)
    counts = counts[ordered_cols]
    row_totals = counts.sum(axis=1)
    pct = counts.div(row_totals, axis=0) * 100.0
    return counts, pct


def cramers_v(era_values, labels):
    counts, _pct = era_crosstab(era_values, labels)
    table = counts.to_numpy(dtype=float)
    chi2, _p, _dof, _expected = chi2_contingency(table)
    n_total = float(table.sum())
    n_rows, n_cols = table.shape
    denominator = n_total * min(n_rows - 1, n_cols - 1)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denominator))


def era_lift_table(era_values, labels):
    """lift[c][e] = P(era e | cluster c) / P(era e)."""
    n_rows = len(era_values)
    era_share = {}
    for era in ERA_ORDER:
        era_share[era] = float(np.sum(era_values == era)) / n_rows
    cluster_ids = sorted(set(labels))
    lift = {}
    for cluster_id in cluster_ids:
        in_cluster = labels == cluster_id
        n_cluster = int(in_cluster.sum())
        lift[cluster_id] = {}
        for era in ERA_ORDER:
            if n_cluster == 0 or era_share[era] == 0:
                lift[cluster_id][era] = np.nan
                continue
            share_in_cluster = float(np.sum(era_values[in_cluster] == era)) / n_cluster
            lift[cluster_id][era] = share_in_cluster / era_share[era]
    return lift, cluster_ids


def era_lift_with_cis(era_values, labels, resamples):
    """Full-fit era lift + percentile CI over Hungarian-matched stability
    resamples (the same resamples the ARI summary used)."""
    full_lift, cluster_ids = era_lift_table(era_values, labels)
    lift_samples = {}
    for cluster_id in cluster_ids:
        lift_samples[cluster_id] = {}
        for era in ERA_ORDER:
            lift_samples[cluster_id][era] = []
    for idx, matched_labels, _ari in resamples:
        era_sub = era_values[idx]
        sample_lift, _sample_clusters = era_lift_table(era_sub, matched_labels)
        for cluster_id in cluster_ids:
            for era in ERA_ORDER:
                if cluster_id in sample_lift:
                    value = sample_lift[cluster_id][era]
                else:
                    value = np.nan
                lift_samples[cluster_id][era].append(value)
    rows = []
    for cluster_id in cluster_ids:
        n_cluster = int(np.sum(labels == cluster_id))
        for era in ERA_ORDER:
            samples = np.array(lift_samples[cluster_id][era], dtype=float)
            valid = samples[~np.isnan(samples)]
            n_valid = int(len(valid))
            if n_valid > 0:
                low = float(np.percentile(valid, 2.5))
                high = float(np.percentile(valid, 97.5))
            else:
                low = None
                high = None
            rows.append({"cluster": int(cluster_id), "era": era,
                         "n": n_cluster,
                         "lift": float(full_lift[cluster_id][era]),
                         "ci_low": low, "ci_high": high,
                         "n_resamples_valid": n_valid})
    return rows


def refit_labels(X, method, k, cov_type=None):
    if method == "kmeans":
        model = fit_kmeans(X, k)
        return model.labels_
    model = fit_gmm(X, k, cov_type)
    return model.predict(X)


def robustness_cells(X, reference_labels, method, k_star, cov_type=None,
                     k_min=2, k_max=None):
    """Fit at k*-1, k*, k*+1; max-overlap match to the reference solution;
    per reference cluster report recall (share of its members landing in the
    matched cell cluster). Survival rule, fixed ex ante: recall >= 0.7 in
    EVERY cell of the variant."""
    reference_ids = sorted(set(reference_labels))
    cells = []
    candidates = [k_star - 1, k_star, k_star + 1]
    for k in candidates:
        if k < k_min:
            continue
        if k_max is not None and k > k_max:
            continue
        cell_labels = refit_labels(X, method, k, cov_type)
        mapping = hungarian_map(reference_labels, cell_labels)
        matched = np.empty_like(cell_labels)
        n_rows = len(cell_labels)
        for position in range(n_rows):
            matched[position] = mapping[cell_labels[position]]
        recalls = {}
        for reference_id in reference_ids:
            in_reference = reference_labels == reference_id
            n_reference = int(in_reference.sum())
            if n_reference == 0:
                recalls[int(reference_id)] = {"recall": None, "n": 0}
                continue
            kept = int(np.sum(matched[in_reference] == reference_id))
            recalls[int(reference_id)] = {"recall": kept / n_reference,
                                          "n": n_reference}
        cells.append({"k": int(k), "recall_per_reference_cluster": recalls})
    survival = {}
    for reference_id in reference_ids:
        survived = True
        for cell in cells:
            recall_cell = cell["recall_per_reference_cluster"][int(reference_id)]
            recall = recall_cell["recall"]
            if recall is None or recall < 0.7:
                survived = False
        survival[int(reference_id)] = survived
    return {"cells": cells, "survival_rule": "recall >= 0.7 in every cell",
            "survives": survival}


def cross_variant_overlap(labels_a, labels_b):
    """Row-normalized overlap matrix between two labelings of the SAME rows."""
    counts = pd.crosstab(pd.Series(labels_a, name="a"),
                         pd.Series(labels_b, name="b"))
    row_totals = counts.sum(axis=1)
    shares = counts.div(row_totals, axis=0)
    return counts, shares


def trend_stats(change_values, labels, n_boot=PROFILE_BOOT_B,
                seed=PROFILE_BOOT_SEED):
    """Per-cluster median change + card-level bootstrap CI + coverage.

    change_values: per-row log price change, NaN = ineligible.
    """
    cluster_ids = sorted(set(labels))
    rows = []
    for cluster_id in cluster_ids:
        in_cluster = labels == cluster_id
        n_cluster = int(in_cluster.sum())
        values = change_values[in_cluster]
        cell = median_ci(values, n_boot=n_boot, seed=seed)
        n_eligible = cell["n"]
        if n_cluster > 0:
            coverage = n_eligible / n_cluster
        else:
            coverage = None
        rows.append({"cluster": int(cluster_id), "n_cluster": n_cluster,
                     "n_eligible": n_eligible, "coverage": coverage,
                     "median_change": cell["median"],
                     "ci_low": cell["ci_low"], "ci_high": cell["ci_high"]})
    return rows
