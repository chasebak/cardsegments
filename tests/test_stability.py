"""Stability bootstrap and Hungarian matching."""

import numpy as np

from cardsegments import stability
from cardsegments.cluster import complete_case_matrix
from cardsegments.synth import generate_segmented


def test_hungarian_map_recovers_permutation():
    reference = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([2, 2, 0, 0, 1, 1])
    mapping = stability.hungarian_map(reference, permuted)
    assert mapping == {2: 0, 0: 1, 1: 2}


def test_hungarian_map_extra_cluster_folds_in():
    reference = np.array([0, 0, 0, 1, 1, 1])
    other = np.array([0, 0, 2, 1, 1, 1])
    mapping = stability.hungarian_map(reference, other)
    assert mapping[0] == 0
    assert mapping[1] == 1
    assert mapping[2] in (0, 1)


def test_bootstrap_ari_high_for_separated_clusters():
    frame, manifest, _truth = generate_segmented(
        n_rows=600, n_clusters=3, separation=3.0, seed=1, missingness=False)
    _complete, X = complete_case_matrix(frame, manifest["feature_cols"])
    summary = stability.bootstrap_ari(X, "kmeans", 3, n_boot=10)
    assert summary["mean_ari"] > 0.9
    assert summary["p5_ari"] <= summary["mean_ari"]
    assert summary["min_ari"] <= summary["p5_ari"] + 1e-12


def test_bootstrap_ari_k1_is_nan():
    X = np.random.default_rng(0).normal(size=(50, 3))
    summary = stability.bootstrap_ari(X, "kmeans", 1, n_boot=5)
    assert np.isnan(summary["mean_ari"])


def test_stability_at_k_neighbors_bounds():
    frame, manifest, _truth = generate_segmented(
        n_rows=400, n_clusters=3, separation=3.0, seed=2, missingness=False)
    _complete, X = complete_case_matrix(frame, manifest["feature_cols"])
    rows = stability.stability_at_k_neighbors(X, "kmeans", 3, n_boot=5,
                                              k_min=2, k_max=4)
    ks = []
    for row in rows:
        ks.append(row["k"])
    assert ks == [2, 3, 4]
