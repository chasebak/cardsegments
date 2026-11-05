"""Figures. Real-data figures are drawn ONLY from bundle aggregates (never
per-card rows); per-point scatters are allowed for synthetic data only and
are labeled as synthetic.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ERA_ORDER = ["vintage", "mid", "modern"]
ERA_COLORS = {"vintage": "#8c510a", "mid": "#5aae61", "modern": "#4393c3"}

SHORT_FEATURE_NAMES = {
    "f_log_raw_price": "log raw price",
    "f_log_multiplier": "log multiplier",
    "f_gem_rate_pooled": "gem rate",
    "f_log_pop_total": "log pop total",
    "f_log1p_sales_365": "log sales 365d",
    "f_log1p_sales_90": "log sales 90d",
    "f_tcg_vol_365": "raw volatility",
    "f_psa10_vol_365": "graded volatility",
    "f_momentum_90_pct": "momentum 90d %",
    "f_pop_delta_30d_rate": "pop growth rate",
    "f_log_sellers": "log sellers",
    "f_log_psa9_price": "log grade-9 price",
    "f_gem_rate": "gem rate (raw)",
}


def short_name(feature):
    if feature in SHORT_FEATURE_NAMES:
        return SHORT_FEATURE_NAMES[feature]
    return feature


def robust_z(cluster_median, global_median, global_iqr):
    """Robust z: (median_c - median_g) / (IQR_g / 1.349)."""
    if global_iqr is None or global_iqr == 0:
        return 0.0
    return (cluster_median - global_median) / (global_iqr / 1.349)


def profile_z_matrix(profile, feature_cols, min_cell_n=0):
    """Clusters x features robust-z matrix from a bundle profile block.

    Suppressed/small cells come back as NaN. Returns (matrix, cluster_rows).
    """
    clusters = profile["clusters"]
    global_stats = profile["global"]
    n_clusters = len(clusters)
    n_features = len(feature_cols)
    matrix = np.full((n_clusters, n_features), np.nan)
    for row_position in range(n_clusters):
        cluster = clusters[row_position]
        for col_position in range(n_features):
            feature = feature_cols[col_position]
            cell = cluster["features"].get(feature)
            if cell is None or cell.get("median") is None:
                continue
            if cell["n"] < min_cell_n:
                continue
            global_cell = global_stats[feature]
            matrix[row_position, col_position] = robust_z(
                cell["median"], global_cell["median"], global_cell.get("iqr"))
    return matrix, clusters


def plot_profile_heatmap(axis, matrix, cluster_rows, feature_cols,
                         cluster_names=None, title="segment profiles"):
    display = np.ma.masked_invalid(matrix)
    limit = 2.5
    image = axis.imshow(display, cmap="RdBu_r", vmin=-limit, vmax=limit,
                        aspect="auto")
    axis.set_xticks(range(len(feature_cols)))
    labels = []
    for feature in feature_cols:
        labels.append(short_name(feature))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    row_labels = []
    for row_position in range(len(cluster_rows)):
        cluster = cluster_rows[row_position]
        name = None
        if cluster_names:
            # a bundle reloaded from JSON carries string keys
            if cluster["cluster"] in cluster_names:
                name = cluster_names[cluster["cluster"]]
            elif str(cluster["cluster"]) in cluster_names:
                name = cluster_names[str(cluster["cluster"])]
        if name:
            text = "%s (n=%d)" % (name, cluster["n"])
        else:
            text = "cluster %d (n=%d)" % (cluster["cluster"], cluster["n"])
        row_labels.append(text)
    axis.set_yticks(range(len(cluster_rows)))
    axis.set_yticklabels(row_labels, fontsize=8)
    axis.set_title(title, fontsize=10)
    return image


def plot_era_bars(axis, era_rows, title="era composition per segment"):
    """era_rows: [{cluster, n, vintage, mid, modern}] with whole-percent shares."""
    n_clusters = len(era_rows)
    positions = np.arange(n_clusters)
    bottoms = np.zeros(n_clusters)
    for era in ERA_ORDER:
        values = []
        for row in era_rows:
            values.append(row.get(era, 0.0))
        values = np.array(values, dtype=float)
        axis.bar(positions, values, bottom=bottoms, label=era,
                 color=ERA_COLORS[era], width=0.7)
        bottoms = bottoms + values
    tick_labels = []
    for row in era_rows:
        tick_labels.append("c%d\nn=%d" % (row["cluster"], row["n"]))
    axis.set_xticks(positions)
    axis.set_xticklabels(tick_labels, fontsize=8)
    axis.set_ylabel("% of segment")
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=8)


def save_hero(profile, feature_cols, era_rows, validation_text, out_path,
              cluster_names=None, min_cell_n=50):
    """README hero: profile z-heatmap + era bars + ONE validation number."""
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                gridspec_kw={"width_ratios": [1.6, 1.0]})
    matrix, cluster_rows = profile_z_matrix(profile, feature_cols,
                                            min_cell_n=min_cell_n)
    image = plot_profile_heatmap(axes[0], matrix, cluster_rows, feature_cols,
                                 cluster_names=cluster_names,
                                 title="segment profiles (robust z vs global)")
    figure.colorbar(image, ax=axes[0], shrink=0.85)
    plot_era_bars(axes[1], era_rows)
    figure.suptitle(validation_text, fontsize=9, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    # dpi 100 keeps the README hero at 900 px wide (mobile-friendly)
    figure.savefig(out_path, dpi=100)
    plt.close(figure)
    return out_path


def save_null_control(scan_tables, out_path, data_word="real"):
    """scan_tables: {label: {"silhouette": [{k, silhouette}],
    "gap": [{k, gap, s_k}]}}. data_word names the non-null dataset in the
    titles ("synthetic" for the committed bundle)."""
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    styles = {"v1_real": {"color": "tab:blue", "marker": "o"},
              "real": {"color": "tab:blue", "marker": "o"},
              "v1_synthetic": {"color": "tab:blue", "marker": "o"},
              "mvn_null": {"color": "tab:orange", "marker": "s"},
              "shuffle_null": {"color": "tab:green", "marker": "^"}}
    dataset_labels = list(scan_tables.keys())
    for label in dataset_labels:
        style = styles.get(label, {"marker": "o"})
        sil_rows = scan_tables[label]["silhouette"]
        ks = []
        sils = []
        for row in sil_rows:
            ks.append(row["k"])
            sils.append(row["silhouette"])
        axes[0].plot(ks, sils, label=label, **style)
        gap_rows = scan_tables[label]["gap"]
        gks = []
        gaps = []
        errors = []
        for row in gap_rows:
            gks.append(row["k"])
            gaps.append(row["gap"])
            errors.append(row["s_k"])
        axes[1].errorbar(gks, gaps, yerr=errors, label=label, capsize=3, **style)
    axes[0].set_title("k-means silhouette: %s vs null controls" % data_word,
                      fontsize=10)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("silhouette")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("gap statistic (one-SE): %s vs null controls" % data_word,
                      fontsize=10)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("gap")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return out_path


def save_calibration(bins, base_rate, out_path, title_prefix=""):
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    predicted = []
    observed = []
    sizes = []
    for row in bins:
        predicted.append(row["mean_predicted"])
        observed.append(row["observed_rate"])
        sizes.append(row["n"])
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.plot(predicted, observed, marker="o", color="tab:blue")
    for position in range(len(bins)):
        axis.annotate("n=%d" % sizes[position],
                      (predicted[position], observed[position]),
                      fontsize=6, xytext=(4, 4), textcoords="offset points")
    axis.axhline(base_rate, color="tab:red", linewidth=0.8, alpha=0.6,
                 label="base rate %.2f" % base_rate)
    axis.set_xlabel("mean predicted probability")
    axis.set_ylabel("observed rate")
    axis.set_title(title_prefix + "calibration (pooled out-of-fold)", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return out_path


def save_threshold_cost(curves, out_path, title_prefix=""):
    figure, axis = plt.subplots(figsize=(5.4, 4.0))
    for curve in curves:
        thresholds = []
        costs = []
        for point in curve["points"]:
            thresholds.append(point["threshold"])
            costs.append(point["cost"])
        label = "m=%.2f h=%.2f (t*=%.2f)" % (curve["m"], curve["h"],
                                             curve["t_star"])
        axis.plot(thresholds, costs, label=label)
        axis.axvline(curve["t_star"], linestyle=":", alpha=0.5)
    axis.set_xlabel("decision threshold on P(liquid)")
    axis.set_ylabel("expected cost per card")
    axis.set_title(title_prefix + "threshold-cost curve, three cost settings",
                   fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return out_path


def save_pca_density(X, labels, out_path, synthetic=False):
    """PCA map: hexbin density + centroids + 2-sigma covariance ellipses.

    Per-point scatter is drawn ONLY when synthetic=True (and labeled so).
    """
    center = X.mean(axis=0)
    centered = X - center
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    figure, axis = plt.subplots(figsize=(5.6, 4.6))
    if synthetic:
        cluster_ids = sorted(set(labels))
        for cluster_id in cluster_ids:
            in_cluster = labels == cluster_id
            axis.scatter(coords[in_cluster, 0], coords[in_cluster, 1], s=4,
                         alpha=0.4, label="cluster %d" % cluster_id)
        axis.legend(fontsize=7)
        title = "PCA map (SYNTHETIC data; per-point scatter)"
    else:
        axis.hexbin(coords[:, 0], coords[:, 1], gridsize=35, cmap="Blues",
                    mincnt=1)
        title = "PCA density map"
    cluster_ids = sorted(set(labels))
    for cluster_id in cluster_ids:
        in_cluster = labels == cluster_id
        if int(in_cluster.sum()) < 3:
            continue
        sub = coords[in_cluster]
        centroid = sub.mean(axis=0)
        cov = np.cov(sub, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = float(np.degrees(np.arctan2(eigenvectors[1, 1],
                                            eigenvectors[0, 1])))
        width = 2 * 2.0 * float(np.sqrt(max(eigenvalues[1], 0.0)))
        height = 2 * 2.0 * float(np.sqrt(max(eigenvalues[0], 0.0)))
        ellipse = matplotlib.patches.Ellipse(
            centroid, width, height, angle=angle, fill=False,
            edgecolor="black", linewidth=1.0, alpha=0.8)
        axis.add_patch(ellipse)
        axis.plot(centroid[0], centroid[1], marker="x", color="black",
                  markersize=7)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------- page CLI

def main(argv=None):
    """python -m cardsegments.viz build [--check]: the Buy List Explorer page.

    The page data and their README / DECISIONS / METHOD blocks are built by
    cardsegments.explorer from the committed SYNTHETIC results files; this
    entry point only parses the command line. --check rebuilds in memory and
    exits 1 if any generated file is stale or a static page file is off.
    """
    import argparse

    from cardsegments import explorer

    parser = argparse.ArgumentParser(prog="python -m cardsegments.viz",
                                     description="Build the Buy List Explorer page data.")
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--check", action="store_true",
                        help="rebuild in memory and fail if docs/data or a block is stale")
    args = parser.parse_args(argv)
    if args.check:
        problems = explorer.check()
        if problems:
            for problem in problems:
                print(problem)
            return 1
        print("docs/data and the explorer blocks are fresh; tokens.css, .nojekyll and the "
              "Chart.js tag check out")
        return 0
    problems = explorer.write()
    if problems:
        for problem in problems:
            print(problem)
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
