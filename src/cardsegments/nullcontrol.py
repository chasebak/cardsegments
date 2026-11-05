"""Null-data controls: the identical selection pipeline on data with no
planted clusters.

(a) mvn_null: one multivariate normal matched to the input's complete-case
    mean/covariance (keeps the correlation structure, plants no clusters).
(b) shuffle_null: each column independently permuted (keeps marginals,
    destroys dependence).

The honest reading, learned on real data and kept here as the fixed
methodological rule: mixture criteria (BIC/ICL) and the silhouette respond
to non-Gaussian marginals, not only to cluster structure; the gap statistic
and bootstrap stability are the structure arbiters, silhouette/BIC are
descriptive.
"""

import numpy as np

from cardsegments.cluster import (
    gap_statistic,
    gmm_scan,
    silhouette_scan,
    standardize,
)

MVN_SEED = 42
SHUFFLE_SEED = 43

NO_STRUCTURE_RULE = (
    "no-structure verdict = the gap statistic (one-SE) chooses k=1; "
    "silhouette and BIC/ICL are reported but are not structure arbiters"
)


def build_nulls(raw_matrix, mvn_seed=MVN_SEED, shuffle_seed=SHUFFLE_SEED):
    """Return (mvn_sample, shuffled) on the ORIGINAL scale of raw_matrix."""
    n_rows = raw_matrix.shape[0]
    mean_vector = raw_matrix.mean(axis=0)
    cov_matrix = np.cov(raw_matrix, rowvar=False)
    rng_mvn = np.random.default_rng(mvn_seed)
    mvn_sample = rng_mvn.multivariate_normal(mean_vector, cov_matrix, size=n_rows)
    rng_shuffle = np.random.default_rng(shuffle_seed)
    shuffled = raw_matrix.copy()
    n_cols = shuffled.shape[1]
    for col in range(n_cols):
        permutation = rng_shuffle.permutation(n_rows)
        shuffled[:, col] = shuffled[permutation, col]
    return mvn_sample, shuffled


def scan_dataset(label, X, k_values, gap_b=50, verbose=False):
    """Selection criteria on one standardized matrix (no solution is chosen
    here; a null control only asks what each criterion WOULD pick)."""
    sil_frame, sil_k = silhouette_scan(X, k_values, verbose=verbose)
    gap_frame, gap_k = gap_statistic(X, k_values, n_refs=gap_b, verbose=verbose)
    gmm_frame, bic_choice, icl_choice = gmm_scan(X, k_values, verbose=verbose)
    summary = {
        "label": label,
        "silhouette_k": int(sil_k),
        "gap_one_se_k": int(gap_k),
        "bic_choice": {"cov": bic_choice[0], "k": int(bic_choice[1])},
        "icl_choice": {"cov": icl_choice[0], "k": int(icl_choice[1])},
        "max_silhouette": float(sil_frame["silhouette"].max()),
        "no_structure_by_gap": bool(gap_k == 1),
    }
    tables = {"silhouette": sil_frame, "gap": gap_frame, "gmm": gmm_frame}
    return summary, tables


def run_null_control(raw_matrix, k_values, gap_b=50, mvn_seed=MVN_SEED,
                     shuffle_seed=SHUFFLE_SEED, real_label="real",
                     verbose=False):
    """Scan the real matrix and both nulls with the identical pipeline.

    Returns ({label: summary}, {label: tables}).
    """
    mvn_sample, shuffled = build_nulls(raw_matrix, mvn_seed, shuffle_seed)
    X_real = standardize(raw_matrix)
    X_mvn = standardize(mvn_sample)
    X_shuffle = standardize(shuffled)
    summaries = {}
    tables = {}
    datasets = [(real_label, X_real), ("mvn_null", X_mvn),
                ("shuffle_null", X_shuffle)]
    for label, X in datasets:
        summary, dataset_tables = scan_dataset(label, X, k_values, gap_b=gap_b,
                                               verbose=verbose)
        summaries[label] = summary
        tables[label] = dataset_tables
    return summaries, tables
