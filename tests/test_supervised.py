"""Supervised protocol: cost threshold, no_cluster coverage, calibration
bin floor, and the nested gate on planted value-add vs a no-signal control."""

import numpy as np
import pytest

from cardsegments import supervised
from cardsegments.synth import generate_supervised


def test_cost_threshold_formula():
    assert supervised.cost_threshold(0.10, 0.10) == pytest.approx(2.0 / 3.0)
    assert supervised.cost_threshold(0.10, 0.05) == pytest.approx(0.6)
    assert supervised.cost_threshold(0.10, 0.20) == pytest.approx(0.75)


def test_confusion_counts_sum_to_n():
    y = np.array([0, 0, 1, 1, 1, 0])
    probs = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.4])
    confusion = supervised.confusion_at_threshold(y, probs, 0.5)
    total = confusion["tp"] + confusion["fp"] + confusion["fn"] + confusion["tn"]
    assert total == len(y) == confusion["n"]
    assert confusion["tp"] == 2
    assert confusion["fp"] == 1
    assert confusion["fn"] == 1
    assert confusion["tn"] == 2


def test_calibration_bins_respect_min_n():
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=437)
    y = (rng.uniform(size=437) < probs).astype(int)
    bins = supervised.calibration_bins(y, probs, min_bin_n=50)
    total = 0
    for row in bins:
        assert row["n"] >= 50
        total += row["n"]
    assert total == 437


def test_calibration_bins_small_cohort_does_not_chain_merge():
    # Regression: with n=300, 10 quantile bins and min_bin_n=50 every raw
    # interval is 30 rows, and the old merge rule (test only the CURRENT
    # interval's width) chain-merged everything into ONE bin. Accumulation
    # closes a bin at 60 rows (two 30-row intervals), giving 5 bins.
    rng = np.random.default_rng(1)
    probs = rng.uniform(size=300)
    y = (rng.uniform(size=300) < probs).astype(int)
    bins = supervised.calibration_bins(y, probs, min_bin_n=50, n_bins=10)
    assert len(bins) == 5
    total = 0
    for row in bins:
        assert row["n"] >= 50
        total += row["n"]
    assert total == 300


def test_fold_cluster_onehots_cover_every_row():
    rng = np.random.default_rng(1)
    n_rows = 200
    block = rng.normal(size=(n_rows, 3))
    block[rng.uniform(size=n_rows) < 0.3, 0] = np.nan
    complete_mask = ~np.isnan(block).any(axis=1)
    train_idx = np.arange(0, 150)
    test_idx = np.arange(150, 200)
    train_oh, test_oh = supervised.fold_cluster_onehots(
        block, train_idx, test_idx, complete_mask,
        {"method": "kmeans", "k": 2})
    assert train_oh.shape == (150, 3)
    assert test_oh.shape == (50, 3)
    assert np.all(train_oh.sum(axis=1) == 1.0)
    assert np.all(test_oh.sum(axis=1) == 1.0)
    # incomplete rows land in the explicit no_cluster column (last)
    test_incomplete = ~complete_mask[test_idx]
    assert np.all(test_oh[test_incomplete, 2] == 1.0)


def test_group_bootstrap_resamples_whole_groups():
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    index_lists = supervised.group_index_lists(groups)
    rng = np.random.default_rng(0)
    idx = supervised.group_bootstrap_indices(groups, rng, index_lists)
    # three groups resampled with replacement: counts per group stay intact
    sampled_groups = groups[idx]
    unique_values, counts = np.unique(sampled_groups, return_counts=True)
    sizes = {"a": 2, "b": 3, "c": 1}
    for value, count in zip(unique_values, counts):
        assert count % sizes[str(value)] == 0


def run_protocol(cluster_signal, seed=11):
    frame, manifest, _truth = generate_supervised(
        n_rows=1500, n_clusters=3, separation=2.5, seed=seed,
        cluster_signal=cluster_signal)
    results = supervised.run_supervised(
        frame, manifest["feature_cols"], manifest["baseline_cols"],
        cluster_feature_cols=manifest["feature_cols"],
        cluster_spec={"method": "kmeans", "k": 3},
        n_seeds=2, b_group_boot=200, b_coef_boot=30,
        calibration_min_bin_n=50)
    return results


def test_nested_gate_finds_planted_value_add():
    results = run_protocol(cluster_signal=0.9)
    nested = results["nested_delta_auc"]
    assert nested["mean_over_seeds"] > 0.01
    assert nested["group_bootstrap"]["ci_low"] > 0
    assert nested["ships"] is True
    assert results["n_cluster_feature_complete"] < results["n_rows"]
    # PR-AUC is oriented to whichever class is the minority
    assert results["minority_class"] in ("liquid (y=1)", "sits (y=0)")


def test_nested_gate_null_when_no_planted_signal():
    results = run_protocol(cluster_signal=0.0)
    nested = results["nested_delta_auc"]
    assert nested["ships"] is False
