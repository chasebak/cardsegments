"""Feature screening: correlation structure, feature blocks, pruning, VIF,
price-dominance ranking, and the price-residualization transform.

The pruning rule is fixed ex ante: features are grouped into blocks by
average-linkage hierarchical clustering on distance 1 - |Spearman rho|, cut
at 0.2; each block keeps ONE representative - the feature with the highest
non-missing coverage, ties broken by position in the declared feature order.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

BLOCK_CUT = 0.2


def correlation_matrices(frame, feature_cols):
    """Pairwise-complete Spearman and Pearson correlation matrices."""
    values = frame[feature_cols]
    pearson = values.corr(method="pearson")
    spearman = values.corr(method="spearman")
    return spearman, pearson


def feature_blocks(spearman, cut=BLOCK_CUT):
    """Average-linkage blocks on 1 - |Spearman|, cut at `cut`.

    Returns a dict feature -> block id (1-based).
    """
    feature_names = list(spearman.columns)
    abs_corr = spearman.abs().to_numpy(dtype=float).copy()
    # the diagonal must be exactly 1 and the matrix symmetric for squareform
    np.fill_diagonal(abs_corr, 1.0)
    abs_corr = np.clip(abs_corr, 0.0, 1.0)
    abs_corr = (abs_corr + abs_corr.T) / 2.0
    distance = 1.0 - abs_corr
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    assignments = fcluster(tree, t=cut, criterion="distance")
    blocks = {}
    n_features = len(feature_names)
    for position in range(n_features):
        blocks[feature_names[position]] = int(assignments[position])
    return blocks


def prune_blocks(frame, feature_cols, blocks):
    """One representative per block: highest non-missing coverage, ties broken
    by position in feature_cols. Returns (kept list, drop table rows)."""
    block_ids = sorted(set(blocks.values()))
    kept = []
    drop_rows = []
    for block_id in block_ids:
        members = []
        for feature_name in feature_cols:
            if blocks[feature_name] == block_id:
                members.append(feature_name)
        best_feature = None
        best_coverage = -1
        for feature_name in members:
            coverage = int(frame[feature_name].notna().sum())
            if coverage > best_coverage:
                best_coverage = coverage
                best_feature = feature_name
        kept.append(best_feature)
        for feature_name in members:
            if feature_name != best_feature:
                drop_rows.append({"block": block_id, "dropped": feature_name,
                                  "kept": best_feature})
    ordered_kept = []
    for feature_name in feature_cols:
        if feature_name in kept:
            ordered_kept.append(feature_name)
    return ordered_kept, drop_rows


def vif_table(frame, feature_cols):
    """VIF per feature via numpy lstsq on standardized complete cases."""
    complete = frame.dropna(subset=feature_cols)
    raw = complete[feature_cols].to_numpy(dtype=float)
    means = raw.mean(axis=0)
    stds = raw.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    X = (raw - means) / stds
    n_rows, n_cols = X.shape
    rows = []
    for target_pos in range(n_cols):
        y = X[:, target_pos]
        other_positions = []
        for position in range(n_cols):
            if position != target_pos:
                other_positions.append(position)
        others = X[:, other_positions]
        design = np.column_stack([others, np.ones(n_rows)])
        coef, _res, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
        predicted = design @ coef
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            r2 = 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot
        if r2 >= 1.0:
            vif = float("inf")
        else:
            vif = 1.0 / (1.0 - r2)
        rows.append({"feature": feature_cols[target_pos], "r2": r2, "vif": vif,
                     "n": int(n_rows)})
    return pd.DataFrame(rows)


def r2_on_price(frame, feature_cols, price_col="f_log_raw_price"):
    """Univariate OLS of each feature on the price column, ranked by R^2.

    High R^2 = the feature mostly proxies price level.
    """
    rows = []
    for feature_name in feature_cols:
        if feature_name == price_col:
            continue
        pair = frame[[price_col, feature_name]].dropna()
        x = pair[price_col].to_numpy(dtype=float)
        y = pair[feature_name].to_numpy(dtype=float)
        n_pairs = len(pair)
        if n_pairs < 3:
            rows.append({"feature": feature_name, "r2_on_price": np.nan,
                         "pearson_r": np.nan, "slope": np.nan, "n": n_pairs})
            continue
        design = np.column_stack([x, np.ones(n_pairs)])
        coef, _res, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
        predicted = design @ coef
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            r2 = 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot
        r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"feature": feature_name, "r2_on_price": r2,
                     "pearson_r": r, "slope": float(coef[0]), "n": n_pairs})
    table = pd.DataFrame(rows)
    table = table.sort_values("r2_on_price", ascending=False).reset_index(drop=True)
    return table


def fit_price_residualization(frame, residualize_cols, price_col="f_log_raw_price"):
    """Per-feature OLS fit on the price column (complete pairs).

    Returns {feature: {"slope", "intercept", "n"}} so the same transform can
    be applied to new rows (fit once, apply anywhere).
    """
    fits = {}
    for feature_name in residualize_cols:
        pair = frame[[price_col, feature_name]].dropna()
        x = pair[price_col].to_numpy(dtype=float)
        y = pair[feature_name].to_numpy(dtype=float)
        design = np.column_stack([x, np.ones(len(pair))])
        coef, _res, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
        fits[feature_name] = {"slope": float(coef[0]),
                              "intercept": float(coef[1]),
                              "n": int(len(pair))}
    return fits


def apply_price_residualization(frame, fits, price_col="f_log_raw_price",
                                prefix="f_resid_"):
    """Residualize fitted columns on price; NaN stays NaN.

    Output column names: prefix + original name without its 'f_' prefix.
    """
    result = frame.copy()
    fitted_names = list(fits.keys())
    for feature_name in fitted_names:
        fit = fits[feature_name]
        expected = fit["intercept"] + fit["slope"] * result[price_col]
        residual = result[feature_name] - expected
        if feature_name.startswith("f_"):
            bare_name = feature_name[2:]
        else:
            bare_name = feature_name
        result[prefix + bare_name] = residual
    return result
