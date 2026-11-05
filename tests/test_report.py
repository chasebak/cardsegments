"""Disclosure suppression and findings rendering from a minimal bundle."""

import os

from cardsegments import report


def make_cell(n):
    return {"median": 1.0, "ci_low": 0.5, "ci_high": 1.5, "n": n}


def test_suppress_cell_below_floor():
    kept = report.suppress_cell(make_cell(50))
    assert kept["median"] == 1.0
    suppressed = report.suppress_cell(make_cell(49))
    assert suppressed["median"] is None
    assert suppressed["suppressed"] is True
    assert suppressed["n"] == 49  # the n itself stays visible


def test_apply_disclosure_suppresses_small_cells():
    bundle = {
        "variants": {
            "v1": {
                "profile": {
                    "clusters": [
                        {"cluster": 0, "n": 400,
                         "features": {"f_x": make_cell(30)}},
                        {"cluster": 1, "n": 20,
                         "features": {"f_x": make_cell(400)}},
                    ],
                    "global": {"f_x": make_cell(420)},
                    "clusters_total_n": 420,
                    "feature_cols": ["f_x"],
                },
                "era": {"cluster_shares_pct": [
                    {"cluster": 0, "n": 400, "vintage": 10, "mid": 20,
                     "modern": 70},
                    {"cluster": 1, "n": 20, "vintage": 50, "mid": 25,
                     "modern": 25},
                ]},
                "trends": {"d90": [
                    {"cluster": 0, "n_cluster": 400, "n_eligible": 12,
                     "coverage": 0.03, "median_change": 0.5,
                     "ci_low": 0.1, "ci_high": 0.9},
                ]},
            },
        },
    }
    cleaned = report.apply_disclosure(bundle)
    v1 = cleaned["variants"]["v1"]
    # small feature cell suppressed inside a big cluster
    assert v1["profile"]["clusters"][0]["features"]["f_x"]["median"] is None
    # every cell of a small cluster suppressed, even with big feature n
    assert v1["profile"]["clusters"][1]["features"]["f_x"]["median"] is None
    # era shares of a small cluster replaced by a suppressed stub
    assert v1["era"]["cluster_shares_pct"][1].get("suppressed") is True
    assert "vintage" not in v1["era"]["cluster_shares_pct"][1]
    # trend rows with few eligible cards suppressed
    assert v1["trends"]["d90"][0]["median_change"] is None
    # the original bundle is untouched (deep copy)
    assert bundle["variants"]["v1"]["profile"]["clusters"][0]["features"][
        "f_x"]["median"] == 1.0


def test_apply_disclosure_suppresses_small_lift_and_recall():
    bundle = {
        "variants": {
            "v1": {
                "profile": {
                    "clusters": [
                        {"cluster": 0, "n": 400,
                         "features": {"f_x": make_cell(400)}},
                        {"cluster": 8, "n": 4,
                         "features": {"f_x": make_cell(4)}},
                    ],
                    "global": {"f_x": make_cell(404)},
                    "clusters_total_n": 404,
                    "feature_cols": ["f_x"],
                },
                "era": {
                    "cluster_shares_pct": [
                        {"cluster": 0, "n": 400, "vintage": 10, "mid": 20,
                         "modern": 70},
                        {"cluster": 8, "n": 4, "vintage": 50, "mid": 25,
                         "modern": 25},
                    ],
                    # lift rows WITHOUT n, as older exports wrote them
                    "lift": [
                        {"cluster": 0, "era": "vintage", "lift": 1.8,
                         "ci_low": 1.2, "ci_high": 2.3,
                         "n_resamples_valid": 20},
                        {"cluster": 8, "era": "vintage", "lift": 3.37,
                         "ci_low": 0.0, "ci_high": 11.8,
                         "n_resamples_valid": 20},
                    ],
                },
            },
        },
        "robustness": {
            "v1": {
                # bare recall floats, as older exports wrote them
                "cells": [{"k": 2, "recall_per_reference_cluster":
                           {"0": 0.9, "8": 1.0}}],
                "survival_rule": "recall >= 0.7 in every cell",
                "survives": {"0": True, "8": True},
            },
        },
    }
    cleaned = report.apply_disclosure(bundle)
    lift_rows = cleaned["variants"]["v1"]["era"]["lift"]
    # every lift row carries its cluster n
    assert lift_rows[0]["n"] == 400
    assert lift_rows[0]["lift"] == 1.8
    # the small cluster's lift stats are nulled, the n stays visible
    assert lift_rows[1]["n"] == 4
    assert lift_rows[1]["lift"] is None
    assert lift_rows[1]["ci_low"] is None
    assert lift_rows[1]["ci_high"] is None
    assert lift_rows[1]["suppressed"] is True
    recalls = cleaned["robustness"]["v1"]["cells"][0][
        "recall_per_reference_cluster"]
    # every recall cell carries its reference-cluster n
    assert recalls["0"] == {"recall": 0.9, "n": 400}
    assert recalls["8"]["recall"] is None
    assert recalls["8"]["n"] == 4
    assert recalls["8"]["suppressed"] is True


def test_sig_and_pct_formatting():
    assert report.sig(0.45678) == "0.457"
    assert report.sig(None) == "suppressed"
    assert report.sig(float("nan")) == "n/a"
    assert report.whole_pct(41.7) == "42%"


def test_clean_nan_recurses():
    value = {"a": float("nan"), "b": [1.0, float("nan")], "c": {"d": 2.0}}
    cleaned = report.clean_nan(value)
    assert cleaned["a"] is None
    assert cleaned["b"][1] is None
    assert cleaned["c"]["d"] == 2.0


def test_cell_text_marks_suppressed():
    assert report.cell_text(None) == "suppressed (n=0)"
    text = report.cell_text({"median": None, "n": 12})
    assert text == "suppressed (n=12)"
    text = report.cell_text(make_cell(80))
    assert text.startswith("1 [0.5, 1.5]")
    assert "(n=80)" in text


def minimal_supervised_block():
    return {
        "n_rows": 1000, "n_groups": 40, "base_rate": 0.23,
        "n_cluster_feature_complete": 300,
        "pooled_auc": {
            "majority": {"auc": 0.5, "ci_low": 0.5, "ci_high": 0.5},
            "era_velocity": {"auc": 0.8, "ci_low": 0.78, "ci_high": 0.82},
            "era_velocity_cluster": {"auc": 0.81, "ci_low": 0.79,
                                     "ci_high": 0.83},
            "primary_logistic": {"auc": 0.82, "ci_low": 0.80,
                                 "ci_high": 0.84},
        },
        "nested_delta_auc": {
            "mean_over_seeds": 0.005,
            "group_bootstrap": {"mean": 0.005, "ci_low": -0.002,
                                "ci_high": 0.012, "n_boot_valid": 200,
                                "n_boot_skipped": 0},
            "fold_seed_spread": {"mean": 0.005, "min": -0.01, "p5": -0.005,
                                 "n_cells": 10},
            "ship_rule": "95% CI excludes 0 AND mean delta AUC >= +0.01",
            "ships": False,
        },
        "minority_class": "liquid (y=1)",
        "pr_auc_minority": 0.55,
        "confusion_at_t_star": {"threshold": 2.0 / 3.0, "tp": 50, "fp": 20,
                                "fn": 180, "tn": 750, "precision": 0.71,
                                "recall": 0.22, "specificity": 0.97,
                                "f1": 0.33, "n": 1000},
        "threshold_cost_curves": [
            {"m": 0.1, "h": 0.1, "t_star": 2.0 / 3.0,
             "points": [{"threshold": 0.2, "cost": 0.08},
                        {"threshold": 0.8, "cost": 0.05}]},
        ],
        "calibration_bins": [
            {"n": 500, "mean_predicted": 0.1, "observed_rate": 0.12},
            {"n": 500, "mean_predicted": 0.4, "observed_rate": 0.38},
        ],
        "coefficients": [
            {"feature": "f_log_raw_price", "coef": -0.2, "ci_low": -0.3,
             "ci_high": -0.1, "n_boot_valid": 30},
        ],
        "forward_regression": {"n_eligible": 30, "oof_r2": None,
                               "note": "fewer than 50 eligible rows"},
    }


def minimal_variant(weak):
    profile = {
        "clusters": [
            {"cluster": 0, "n": 400, "features": {"f_x": make_cell(400)}},
            {"cluster": 1, "n": 300, "features": {"f_x": make_cell(300)}},
        ],
        "global": {"f_x": {"median": 1.0, "ci_low": 0.9, "ci_high": 1.1,
                           "n": 700, "iqr": 0.8}},
        "clusters_total_n": 700,
        "feature_cols": ["f_x"],
    }
    if weak:
        gap_k = 1
        kmeans_chosen = 1
    else:
        gap_k = 2
        kmeans_chosen = 2
    selection = {
        "variant": "x", "n": 700, "p": 1,
        "kmeans": {"silhouette_k": 2, "gap_k": gap_k, "chosen_k": kmeans_chosen,
                   "how": "agree", "profile_fallback_k": None,
                   "bootstrap": {"mean_ari": 0.9, "std_ari": 0.02,
                                 "min_ari": 0.85, "p5_ari": 0.86},
                   "fallback_bootstrap": None},
        "gmm": {"bic_choice": {"cov": "full", "k": 3},
                "icl_choice": {"cov": "full", "k": 3},
                "chosen": {"cov": "full", "k": 3}, "how": "agree",
                "bootstrap": {"mean_ari": 0.6, "std_ari": 0.1,
                              "min_ari": 0.4, "p5_ari": 0.45}},
        "solution": {"method": "kmeans", "k": max(kmeans_chosen, 2),
                     "cov_type": ""},
        "weak_structure": weak,
        "max_silhouette": 0.2,
    }
    era_rows = [{"cluster": 0, "n": 400, "vintage": 10, "mid": 20,
                 "modern": 70},
                {"cluster": 1, "n": 300, "vintage": 40, "mid": 30,
                 "modern": 30}]
    lift_rows = [{"cluster": 0, "era": "vintage", "lift": 0.4,
                  "ci_low": 0.2, "ci_high": 0.7, "n_resamples_valid": 20},
                 {"cluster": 1, "era": "vintage", "lift": 1.8,
                  "ci_low": 1.2, "ci_high": 2.3, "n_resamples_valid": 20}]
    trend_rows = [{"cluster": 0, "n_cluster": 400, "n_eligible": 80,
                   "coverage": 0.2, "median_change": 0.01, "ci_low": -0.02,
                   "ci_high": 0.05}]
    return {
        "n_rows": 700, "n_complete": 700, "p": 1, "feature_cols": ["f_x"],
        "selection": selection,
        "profile": profile,
        "era": {"cluster_shares_pct": era_rows, "lift": lift_rows,
                "cramers_v": 0.3},
        "trends": {"protocol": "test protocol", "d90": trend_rows,
                   "d365": trend_rows},
        "cluster_names": {0: "alpha", 1: "beta"},
    }


def minimal_bundle(weak):
    scan = {"silhouette": [{"k": 2, "silhouette": 0.2}],
            "gap": [{"k": 1, "gap": 0.1, "s_k": 0.02},
                    {"k": 2, "gap": 0.12, "s_k": 0.02}]}
    robustness = {
        "cells": [{"k": 2, "recall_per_reference_cluster": {0: 0.9, 1: 0.8}}],
        "survival_rule": "recall >= 0.7 in every cell",
        "survives": {0: True, 1: False},
    }
    return {
        "bundle_version": 1,
        "anchor_date": "TEST",
        "stability_b": 20,
        "disclosure": {"min_cell_n": report.MIN_CELL_N},
        "variants": {"v1": minimal_variant(weak), "v4": minimal_variant(False)},
        "sensitivities": [
            {"name": "v2", "n": 700, "p": 2, "silhouette_k": 2, "gap_k": 1,
             "gmm_cov": "full", "gmm_k": 3,
             "solution": {"method": "gmm", "k": 3, "cov_type": "full"},
             "max_silhouette": 0.19},
        ],
        "robustness": {"v1": robustness, "v4": robustness},
        "null_control": {
            "summaries": {
                "real": {"silhouette_k": 2, "gap_one_se_k": 1,
                         "bic_choice": {"cov": "full", "k": 3},
                         "icl_choice": {"cov": "full", "k": 3},
                         "max_silhouette": 0.2, "no_structure_by_gap": True},
                "shuffle_null": {"silhouette_k": 2, "gap_one_se_k": 1,
                                 "bic_choice": {"cov": "diag", "k": 7},
                                 "icl_choice": {"cov": "diag", "k": 4},
                                 "max_silhouette": 0.28,
                                 "no_structure_by_gap": True},
            },
            "scan_tables": {"real": scan, "shuffle_null": scan},
            "methodological_conclusion": "test conclusion.",
        },
        "supervised": {
            "primary_anchor": "TEST-P",
            "sensitivity_anchor": "TEST-S",
            "primary": minimal_supervised_block(),
            "sensitivity": minimal_supervised_block(),
            "notes": ["note one", "note two"],
        },
        "era_argument": "test era argument.",
        "predictions": [
            {"id": 1, "summary": "test prediction", "verdict": "confirmed",
             "deciding_statistic": "test stat"},
        ],
        "predictions_registration": {"date": "TEST", "sha256": "0" * 64},
        "limitations": ["test limitation"],
        "headline": {"hero_validation_text": "test hero"},
        "nopop_extrapolation": {
            "note": "test extrapolation note",
            "clusters": [{"cluster": 0, "n": 120}, {"cluster": 1, "n": 60}],
        },
    }


def test_render_findings_weak_and_strong_branches(tmp_path):
    for weak in (True, False):
        out_dir = str(tmp_path / ("weak_%s" % weak))
        os.makedirs(out_dir, exist_ok=True)
        written = report.render_findings(minimal_bundle(weak), out_dir)
        findings_path = written[-1]
        with open(findings_path) as handle:
            text = handle.read()
        if weak:
            assert report.WEAK_STRUCTURE_LEAD in text
        else:
            assert report.WEAK_STRUCTURE_LEAD not in text
            assert "The selection pipeline chose" in text
        assert "Sensitivity variants" in text
        assert "note one" in text
        assert "test era argument." in text
        assert "**confirmed**" in text
        for name in ("hero.png", "null_control.png", "calibration.png",
                     "threshold_cost.png"):
            assert os.path.exists(os.path.join(out_dir, "figures", name))
