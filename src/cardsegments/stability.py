"""Bootstrap stability of a clustering solution.

B resamples of 80% of rows without replacement (sampling seeds seed_base..
seed_base+B-1, refit seed fixed), full refit per resample, ARI of the refit
labels against the full-data labels on the sampled rows. Hungarian matching
aligns resample clusters to full clusters (ARI itself is permutation
invariant; the matched labels are what era-lift CIs and profile CIs use).

B defaults to 20 for tests/demo; the production stability run uses a larger
B set by the caller.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from cardsegments.cluster import fit_gmm, fit_kmeans

N_BOOT_DEFAULT = 20
BOOT_FRAC = 0.8
BOOT_SEED_BASE = 1000


def hungarian_map(reference_labels, other_labels):
    """Map other-cluster id -> reference-cluster id by max overlap."""
    reference_ids = sorted(set(reference_labels))
    other_ids = sorted(set(other_labels))
    overlap = np.zeros((len(other_ids), len(reference_ids)))
    for row_pos in range(len(other_ids)):
        for col_pos in range(len(reference_ids)):
            both = (other_labels == other_ids[row_pos]) & \
                   (reference_labels == reference_ids[col_pos])
            overlap[row_pos, col_pos] = int(both.sum())
    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {}
    n_assigned = len(row_ind)
    for position in range(n_assigned):
        mapping[other_ids[row_ind[position]]] = reference_ids[col_ind[position]]
    # unassigned ids (more other clusters than reference): map to best overlap
    for other_id in other_ids:
        if other_id not in mapping:
            row_pos = other_ids.index(other_id)
            best_col = int(np.argmax(overlap[row_pos]))
            mapping[other_id] = reference_ids[best_col]
    return mapping


def summarize_ari(ari_values):
    ari_arr = np.array(ari_values, dtype=float)
    summary = {"mean_ari": float(ari_arr.mean()),
               "std_ari": float(ari_arr.std(ddof=0)),
               "min_ari": float(ari_arr.min()),
               "p5_ari": float(np.percentile(ari_arr, 5.0))}
    return summary


def bootstrap_resample_labels(X, method, k, cov_type=None, n_boot=N_BOOT_DEFAULT,
                              frac=BOOT_FRAC, seed_base=BOOT_SEED_BASE):
    """Refit on n_boot x frac subsamples. Returns the full-data labels and a
    list of (index_array, hungarian-matched subsample labels, ari), plus the
    ARI summary (mean, std, min, 5th percentile)."""
    if method == "kmeans":
        full_model = fit_kmeans(X, k)
        full_labels = full_model.labels_
    else:
        full_model = fit_gmm(X, k, cov_type)
        full_labels = full_model.predict(X)
    n_rows = X.shape[0]
    sample_size = int(round(frac * n_rows))
    resamples = []
    ari_values = []
    for i in range(n_boot):
        rng = np.random.default_rng(seed_base + i)
        idx = rng.choice(n_rows, size=sample_size, replace=False)
        X_sub = X[idx]
        if method == "kmeans":
            sub_model = fit_kmeans(X_sub, k)
            sub_labels = sub_model.labels_
        else:
            sub_model = fit_gmm(X_sub, k, cov_type)
            sub_labels = sub_model.predict(X_sub)
        ari = float(adjusted_rand_score(full_labels[idx], sub_labels))
        ari_values.append(ari)
        mapping = hungarian_map(full_labels[idx], sub_labels)
        matched = np.empty_like(sub_labels)
        n_sub = len(sub_labels)
        for position in range(n_sub):
            matched[position] = mapping[sub_labels[position]]
        resamples.append((idx, matched, ari))
    summary = summarize_ari(ari_values)
    return full_labels, resamples, summary


def bootstrap_ari(X, method, k, cov_type=None, n_boot=N_BOOT_DEFAULT,
                  frac=BOOT_FRAC, seed_base=BOOT_SEED_BASE):
    """ARI summary only. k=1 has no ARI (NaN summary by convention)."""
    if k < 2:
        return {"mean_ari": float("nan"), "std_ari": float("nan"),
                "min_ari": float("nan"), "p5_ari": float("nan")}
    _full, _resamples, summary = bootstrap_resample_labels(
        X, method, k, cov_type, n_boot=n_boot, frac=frac, seed_base=seed_base)
    return summary


def stability_at_k_neighbors(X, method, k_star, cov_type=None,
                             n_boot=N_BOOT_DEFAULT, k_min=2, k_max=None):
    """ARI summaries at k*-1, k*, k*+1 (clipped to [k_min, k_max])."""
    rows = []
    candidates = [k_star - 1, k_star, k_star + 1]
    for k in candidates:
        if k < k_min:
            continue
        if k_max is not None and k > k_max:
            continue
        summary = bootstrap_ari(X, method, k, cov_type, n_boot=n_boot)
        row = {"k": int(k), "method": method, "cov_type": cov_type or "",
               **summary}
        rows.append(row)
    return rows
