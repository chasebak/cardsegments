"""Cluster-count selection with fixed, per-method criteria.

Fixed settings (chosen ex ante, never tuned on results):
  k-means: k in K_VALUES, n_init=10, seed 42; criteria = silhouette AND the
    gap statistic (PCA-aligned uniform reference box, B=50, one-SE rule,
    k=1 allowed).
  GMM: covariances {spherical, diag, full}, same k grid, n_init=10,
    reg_covar=1e-6, seed 42; criteria = BIC AND ICL jointly.
  Within-method disagreements are recorded and tiebroken by bootstrap ARI;
  the dataset's single chosen solution = the method/k with the higher mean
  bootstrap ARI. A gap-statistic choice of k=1 is a no-structure verdict:
  it stands, and the silhouette k becomes a profiling fallback (reported,
  never hidden).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
COV_TYPES = ["spherical", "diag", "full"]
FIT_SEED = 42
GAP_B = 50
GAP_SEED = 777


def standardize(matrix):
    """Z-score columns; constant columns are left centered (std treated as 1)."""
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds = stds.copy()
    stds[stds == 0] = 1.0
    return (matrix - means) / stds


def complete_case_matrix(frame, feature_cols):
    """Complete cases on feature_cols, z-scored. Returns (complete_frame, X)."""
    complete = frame.dropna(subset=feature_cols).reset_index(drop=True)
    raw = complete[feature_cols].to_numpy(dtype=float)
    X = standardize(raw)
    return complete, X


def fit_kmeans(X, k, fit_seed=FIT_SEED):
    model = KMeans(n_clusters=k, n_init=10, random_state=fit_seed)
    model.fit(X)
    return model


def fit_gmm(X, k, cov_type, fit_seed=FIT_SEED):
    model = GaussianMixture(n_components=k, covariance_type=cov_type,
                            n_init=10, reg_covar=1e-6, random_state=fit_seed)
    model.fit(X)
    return model


def fit_solution(X, solution, fit_seed=FIT_SEED):
    """Fit the chosen (method, k, cov_type) and return hard labels."""
    method = solution["method"]
    k = solution["k"]
    if method == "kmeans":
        model = fit_kmeans(X, k, fit_seed)
        labels = model.labels_
    else:
        model = fit_gmm(X, k, solution["cov_type"], fit_seed)
        labels = model.predict(X)
    return model, labels


def log_wk(X, k, fit_seed=FIT_SEED):
    """log of within-cluster sum of squares; k=1 = total SS to the mean."""
    if k == 1:
        center = X.mean(axis=0)
        diffs = X - center
        w = float(np.sum(diffs ** 2))
    else:
        model = fit_kmeans(X, k, fit_seed)
        w = float(model.inertia_)
    return np.log(w)


def make_pca_box_references(X, n_refs, rng):
    """Tibshirani et al. (2001) method (b): uniform box aligned with PCA axes."""
    center = X.mean(axis=0)
    centered = X - center
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vt.T
    lows = rotated.min(axis=0)
    highs = rotated.max(axis=0)
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    references = []
    for _b in range(n_refs):
        uniform = rng.uniform(size=(n_rows, n_cols))
        sample_rotated = lows + uniform * (highs - lows)
        sample = sample_rotated @ vt + center
        references.append(sample)
    return references


def gap_statistic(X, k_values, n_refs=GAP_B, gap_seed=GAP_SEED, verbose=False):
    """Gap(k), s_k and the one-SE chosen k (k=1 allowed).

    Returns (gap_frame indexed by k, chosen_k).
    """
    rng = np.random.default_rng(gap_seed)
    references = make_pca_box_references(X, n_refs, rng)
    records = []
    for k in k_values:
        log_w_data = log_wk(X, k)
        ref_log_ws = []
        for reference in references:
            ref_log_ws.append(log_wk(reference, k))
        ref_arr = np.array(ref_log_ws)
        gap = float(ref_arr.mean() - log_w_data)
        sk = float(ref_arr.std(ddof=0) * np.sqrt(1.0 + 1.0 / n_refs))
        records.append({"k": k, "log_w": log_w_data,
                        "ref_log_w_mean": float(ref_arr.mean()),
                        "gap": gap, "s_k": sk})
        if verbose:
            print("  GAP k=%d logW=%.4f gap=%.4f s=%.4f" % (k, log_w_data, gap, sk))
    gap_frame = pd.DataFrame(records).set_index("k")
    chosen = None
    ordered_ks = list(gap_frame.index)
    n_ks = len(ordered_ks)
    for position in range(n_ks - 1):
        k = ordered_ks[position]
        k_next = ordered_ks[position + 1]
        gap_k = gap_frame.loc[k, "gap"]
        gap_next = gap_frame.loc[k_next, "gap"]
        s_next = gap_frame.loc[k_next, "s_k"]
        if gap_k >= gap_next - s_next:
            chosen = k
            break
    if chosen is None:
        chosen = ordered_ks[-1]
        if verbose:
            print("  GAP one-SE never satisfied; falling back to max scanned k=%d" % chosen)
    return gap_frame, chosen


def silhouette_scan(X, k_values, verbose=False):
    """k-means silhouette over the k grid (k=1 has no silhouette)."""
    records = []
    for k in k_values:
        if k < 2:
            records.append({"k": k, "silhouette": np.nan})
            continue
        model = fit_kmeans(X, k)
        labels = model.labels_
        value = float(silhouette_score(X, labels))
        records.append({"k": k, "silhouette": value})
        if verbose:
            print("  SIL k=%d silhouette=%.4f" % (k, value))
    sil_frame = pd.DataFrame(records).set_index("k")
    valid = sil_frame["silhouette"].dropna()
    chosen = int(valid.idxmax())
    return sil_frame, chosen


def gmm_scan(X, k_values, cov_types=None, verbose=False):
    """BIC and ICL over cov types x k. ICL = BIC + 2 * posterior entropy."""
    if cov_types is None:
        cov_types = COV_TYPES
    records = []
    for cov_type in cov_types:
        for k in k_values:
            model = fit_gmm(X, k, cov_type)
            bic = float(model.bic(X))
            probabilities = model.predict_proba(X)
            clipped = np.clip(probabilities, 1e-12, 1.0)
            entropy = float(-np.sum(probabilities * np.log(clipped)))
            icl = bic + 2.0 * entropy
            records.append({"cov_type": cov_type, "k": k, "bic": bic,
                            "entropy": entropy, "icl": icl,
                            "converged": bool(model.converged_)})
            if verbose:
                print("  GMM cov=%-9s k=%2d bic=%12.1f icl=%12.1f conv=%s"
                      % (cov_type, k, bic, icl, model.converged_))
    gmm_frame = pd.DataFrame(records)
    bic_row = gmm_frame.loc[gmm_frame["bic"].idxmin()]
    icl_row = gmm_frame.loc[gmm_frame["icl"].idxmin()]
    bic_choice = (str(bic_row["cov_type"]), int(bic_row["k"]))
    icl_choice = (str(icl_row["cov_type"]), int(icl_row["k"]))
    return gmm_frame, bic_choice, icl_choice


def pick_kmeans_k(name, X, sil_k, gap_k, tiebreak_rows, n_boot=20, verbose=False):
    """Fixed rule: agree -> that k; gap k=1 -> no-structure verdict with the
    silhouette k as profiling fallback; otherwise bootstrap-ARI tiebreak."""
    # local import: stability imports fit functions from this module
    from cardsegments import stability

    if gap_k == sil_k:
        if verbose:
            print("[%s] kmeans: silhouette and gap agree on k=%d" % (name, gap_k))
        return gap_k, "agree"
    if verbose:
        print("[%s] kmeans DISAGREEMENT: silhouette k=%d vs gap k=%d" % (name, sil_k, gap_k))
    summary_sil = stability.bootstrap_ari(X, "kmeans", sil_k, n_boot=n_boot)
    summary_gap = stability.bootstrap_ari(X, "kmeans", gap_k, n_boot=n_boot)
    tiebreak_rows.append({"variant": name, "method": "kmeans", "k": sil_k,
                          "cov_type": "", "role": "tiebreak-silhouette",
                          **summary_sil})
    tiebreak_rows.append({"variant": name, "method": "kmeans", "k": gap_k,
                          "cov_type": "", "role": "tiebreak-gap",
                          **summary_gap})
    if gap_k == 1:
        # k=1 has no ARI; gap says "no structure" -- keep that verdict but
        # the profiled fallback is the silhouette k (reported, not hidden)
        return 1, "gap-k1-no-structure"
    if np.isnan(summary_sil["mean_ari"]) or \
            summary_gap["mean_ari"] >= summary_sil["mean_ari"]:
        return gap_k, "tiebreak-ari"
    return sil_k, "tiebreak-ari"


def pick_gmm_choice(name, X, bic_choice, icl_choice, tiebreak_rows, n_boot=20,
                    verbose=False):
    from cardsegments import stability

    if bic_choice == icl_choice:
        if verbose:
            print("[%s] gmm: BIC and ICL agree on cov=%s k=%d"
                  % (name, bic_choice[0], bic_choice[1]))
        return bic_choice, "agree"
    if verbose:
        print("[%s] gmm DISAGREEMENT: BIC %s k=%d vs ICL %s k=%d"
              % (name, bic_choice[0], bic_choice[1], icl_choice[0], icl_choice[1]))
    summary_bic = stability.bootstrap_ari(X, "gmm", bic_choice[1], bic_choice[0],
                                          n_boot=n_boot)
    summary_icl = stability.bootstrap_ari(X, "gmm", icl_choice[1], icl_choice[0],
                                          n_boot=n_boot)
    tiebreak_rows.append({"variant": name, "method": "gmm", "k": bic_choice[1],
                          "cov_type": bic_choice[0], "role": "tiebreak-bic",
                          **summary_bic})
    tiebreak_rows.append({"variant": name, "method": "gmm", "k": icl_choice[1],
                          "cov_type": icl_choice[0], "role": "tiebreak-icl",
                          **summary_icl})
    bic_nan = np.isnan(summary_bic["mean_ari"])
    icl_nan = np.isnan(summary_icl["mean_ari"])
    if bic_nan and icl_nan:
        return icl_choice, "tiebreak-ari-both-k1"
    if bic_nan:
        return icl_choice, "tiebreak-ari"
    if icl_nan:
        return bic_choice, "tiebreak-ari"
    if summary_icl["mean_ari"] >= summary_bic["mean_ari"]:
        return icl_choice, "tiebreak-ari"
    return bic_choice, "tiebreak-ari"


def run_selection(name, X, k_values=None, n_boot=20, gap_b=GAP_B, verbose=False):
    """The full fixed selection pipeline on one standardized matrix.

    Returns a selection report dict (disagreements recorded, stability the
    tiebreaker) plus the scan tables.
    """
    from cardsegments import stability

    if k_values is None:
        k_values = K_VALUES
    tiebreak_rows = []
    sil_frame, sil_k = silhouette_scan(X, k_values, verbose=verbose)
    gap_frame, gap_k = gap_statistic(X, k_values, n_refs=gap_b, verbose=verbose)
    gmm_frame, bic_choice, icl_choice = gmm_scan(X, k_values, verbose=verbose)

    km_k, km_how = pick_kmeans_k(name, X, sil_k, gap_k, tiebreak_rows,
                                 n_boot=n_boot, verbose=verbose)
    gmm_pick, gmm_how = pick_gmm_choice(name, X, bic_choice, icl_choice,
                                        tiebreak_rows, n_boot=n_boot, verbose=verbose)

    km_summary = stability.bootstrap_ari(X, "kmeans", km_k, n_boot=n_boot)
    gmm_summary = stability.bootstrap_ari(X, "gmm", gmm_pick[1], gmm_pick[0],
                                          n_boot=n_boot)
    if km_k == 1 and sil_k != 1:
        km_profile_fallback = sil_k
        km_fallback_summary = stability.bootstrap_ari(X, "kmeans", sil_k, n_boot=n_boot)
    else:
        km_profile_fallback = None
        km_fallback_summary = None

    km_ari = km_summary["mean_ari"]
    gmm_ari = gmm_summary["mean_ari"]
    if np.isnan(km_ari) and np.isnan(gmm_ari):
        solution = {"method": "none", "k": 1, "cov_type": ""}
    elif np.isnan(km_ari) or (not np.isnan(gmm_ari) and gmm_ari > km_ari):
        solution = {"method": "gmm", "k": int(gmm_pick[1]), "cov_type": gmm_pick[0]}
    else:
        solution = {"method": "kmeans", "k": int(km_k), "cov_type": ""}

    report = {
        "variant": name,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "kmeans": {"silhouette_k": int(sil_k), "gap_k": int(gap_k),
                   "chosen_k": int(km_k), "how": km_how,
                   "profile_fallback_k": km_profile_fallback,
                   "bootstrap": km_summary,
                   "fallback_bootstrap": km_fallback_summary},
        "gmm": {"bic_choice": {"cov": bic_choice[0], "k": int(bic_choice[1])},
                "icl_choice": {"cov": icl_choice[0], "k": int(icl_choice[1])},
                "chosen": {"cov": gmm_pick[0], "k": int(gmm_pick[1])},
                "how": gmm_how, "bootstrap": gmm_summary},
        "solution": solution,
        "weak_structure": bool(km_k == 1),
        "max_silhouette": float(sil_frame["silhouette"].max()),
        "tiebreaks": tiebreak_rows,
    }
    tables = {"silhouette": sil_frame, "gap": gap_frame, "gmm": gmm_frame}
    return report, tables
