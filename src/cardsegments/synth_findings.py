"""Synthetic-only v2 of the findings (DECISIONS.md, 2026-11-04 — Findings
regenerated on synthetic data only).

The committed findings report used to carry aggregates computed on a
licensed price feed. Until the feed terms are read, every tracked output is
synthetic: this module runs the SAME method the private run used (screen ->
select k -> stability -> null controls -> profiles and era lifts ->
robustness cells -> supervised nested gate) on data from cardsegments.synth,
judges the pre-registered predictions in code, and writes

    reports/results_bundle.json   the aggregates bundle, labelled synthetic
    reports/findings.md           rendered by report.findings_text
    reports/figures/*.png         rendered by report.render_figures
    README.md / DECISIONS.md / docs/METHOD.md   generated blocks

    python -m cardsegments.synth_findings build    # heavy (~10 min), never in CI
    python -m cardsegments.synth_findings render   # re-render from the committed bundle
    python -m cardsegments.synth_findings retext   # re-derive text fields, no recompute
    python -m cardsegments.synth_findings --check  # fast; CI and the drift test

Every number in the generated text is read from the committed bundle; this
module's report functions only format. "Why this metric" text lives in
WHY_THIS_METRIC, copied into the bundle so README and findings.md share it.
Failure mode: the bundle shows what the method does when the truth is
planted; it says nothing about the real market, and a generator whose
structure is too easy would make the method look better than it is (the
null controls and the weak-separation sensitivity are the counterweight).
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cardsegments import ladder, nullcontrol, profiles, report, screen, stability, synth
from cardsegments.cluster import K_VALUES, complete_case_matrix, run_selection
from cardsegments.supervised import SHIP_MIN_DELTA, run_supervised

BUNDLE_REL = "reports/results_bundle.json"
FINDINGS_REL = "reports/findings.md"
FIGURE_NAMES = ["hero.png", "null_control.png", "calibration.png", "threshold_cost.png"]
PREDICTIONS_REL = "results/PREDICTIONS_synthetic_findings_2026-11-04.md"
PREDICTIONS_DATE = "2026-11-04"
SHA_PREFIX_LEN = 16
PRICE_COL = "f_log_raw_price"
NUISANCE_COLS = ["f_gem_rate_pooled", "f_tcg_vol_365"]
RESID_PREFIX = "f_resid_"
V1_REAL_LABEL = "v1_synthetic"

# Pre-registered thresholds (results/PREDICTIONS_synthetic_findings_2026-11-04.md).
PLANTED_ARI_MIN = 0.90
STABILITY_ARI_MIN = 0.90
NUISANCE_RHO_MAX = 0.10

FROZEN_SETTINGS = {
    "seg_n_rows": 2000,
    "seg_k": 4,
    "seg_separation": 2.0,
    "seg_seed": 1105,
    "weak_separation": 1.0,
    "k_values": list(K_VALUES),
    "gap_b": 50,
    "stability_b": 200,
    "signals": [0.0, 0.9],
    "supervised_n_rows": synth.ILLUSTRATIVE_N_ROWS,
    "supervised_groups_per_cluster": synth.ILLUSTRATIVE_GROUPS_PER_CLUSTER,
    "supervised_seed": synth.ILLUSTRATIVE_SEED,
    "cluster_k": 3,
    "n_seeds": 10,
    "b_group_boot": 1000,
    "b_coef_boot": 200,
}

# Tiny settings for the CI smoke test: same code path, seconds instead of minutes.
SMOKE_SETTINGS = {
    "seg_n_rows": 300,
    "seg_k": 3,
    "seg_separation": 3.0,
    "seg_seed": 5,
    "weak_separation": 1.0,
    "k_values": [1, 2, 3, 4],
    "gap_b": 5,
    "stability_b": 5,
    "signals": [0.0, 0.9],
    "supervised_n_rows": 600,
    "supervised_groups_per_cluster": 10,
    "supervised_seed": 7,
    "cluster_k": 3,
    "n_seeds": 1,
    "b_group_boot": 50,
    "b_coef_boot": 5,
}

WHY_THIS_METRIC = {
    "structure": {
        "question": "Does the selection pipeline find the segments that are really there, "
                    "and only those?",
        "kind": "Simulated: gap statistic, silhouette, BIC/ICL and bootstrap ARI on "
                "SYNTHETIC cards with planted segments; truth is the planted labels "
                "(adjusted Rand index against them).",
        "lies": "An easy generator flatters any method; the matched-Gaussian and "
                "column-shuffled nulls and the weak-separation sensitivity must read as "
                "no structure (tests/test_cluster_gates.py, prediction 4 and 5).",
    },
    "null_control": {
        "question": "Would the pipeline invent segments in data that has none?",
        "kind": "Simulated control: the identical pipeline on a matched-Gaussian sample "
                "and on column-shuffled data; truth is 'no structure' by construction.",
        "lies": "Mixture criteria react to non-Gaussian marginals, so BIC/ICL can pick "
                "k > 1 on a null; the gap statistic and bootstrap stability are the "
                "arbiters, and the gates in tests/test_cluster_gates.py fail if the gap "
                "statistic finds structure in a null.",
    },
    "nested_gate": {
        "question": "Does knowing a card's segment help predict whether it sells, beyond "
                    "era and trailing sales?",
        "kind": "Modelled: pooled out-of-fold ΔAUC of era + velocity + cluster over era + "
                "velocity on SYNTHETIC cards, paired group bootstrap; truth is the planted "
                "segment effect (0 or 0.9).",
        "lies": "Leakage or a gate that always passes would show a lift at signal 0; the "
                "signal-0 case must FAIL and the signal-0.9 case must PASS (predictions 8 "
                "and 9), and the clusters are re-fit inside every training fold.",
    },
}

ASSUMPTIONS = [
    {"knob": "segmentation data", "value": "n = {:,}, planted k = {}, separation {:.1f}, "
                                           "seed {}".format(FROZEN_SETTINGS["seg_n_rows"],
                                                            FROZEN_SETTINGS["seg_k"],
                                                            FROZEN_SETTINGS["seg_separation"],
                                                            FROZEN_SETTINGS["seg_seed"]),
     "source": "illustrative (round values, never the private cohort)",
     "if_wrong": "an easier or harder planted structure moves every recovery number"},
    {"knob": "weak-separation sensitivity", "value": "separation %.1f"
                                                     % FROZEN_SETTINGS["weak_separation"],
     "source": "design choice", "if_wrong": "a different cut-off changes where the "
                                            "gap statistic starts calling a continuum"},
    {"knob": "planted missingness", "value": "gem rate missing 25% in the low-pop cluster, "
                                             "3% elsewhere",
     "source": "design choice (synth.py)", "if_wrong": "complete-case n and V1 profiles move"},
    {"knob": "V4 residualization", "value": "every kept axis except price, OLS on log raw "
                                            "price",
     "source": "design choice (mirrors the private V4)",
     "if_wrong": "a different list changes what V4 can separate"},
    {"knob": "k grid", "value": "1-10 (k = 1 allowed)", "source": "design choice",
     "if_wrong": "a narrower grid could hide the planted k"},
    {"knob": "gap statistic", "value": "B = %d PCA-box references, one-SE rule"
                                       % FROZEN_SETTINGS["gap_b"],
     "source": "published default (Tibshirani et al. 2001)",
     "if_wrong": "fewer references make the one-SE choice noisier"},
    {"knob": "stability bootstrap", "value": "B = %d, 80%% resamples"
                                             % FROZEN_SETTINGS["stability_b"],
     "source": "design choice", "if_wrong": "ARI summaries carry more Monte Carlo error"},
    {"knob": "supervised data", "value": "the illustrative preset at cluster_signal 0 and 0.9",
     "source": "illustrative (DECISIONS.md, 2026-11-04 — Illustrative preset values)",
     "if_wrong": "the gate's verdict scales with the planted signal"},
    {"knob": "in-fold clustering", "value": "k-means k = %d on the six features"
                                            % FROZEN_SETTINGS["cluster_k"],
     "source": "design choice (planted k of the preset)",
     "if_wrong": "a wrong k blurs the cluster one-hots and weakens the gate"},
    {"knob": "nested-gate ship rule", "value": "95%% CI excludes 0 AND mean ΔAUC >= +%.2f"
                                               % SHIP_MIN_DELTA,
     "source": "pre-registered (unchanged)", "if_wrong": "a lower bar ships smaller lifts"},
    {"knob": "CV and bootstrap", "value": "StratifiedGroupKFold 5 x %d seeds; group "
                                          "bootstrap B = %d"
                                          % (FROZEN_SETTINGS["n_seeds"],
                                             FROZEN_SETTINGS["b_group_boot"]),
     "source": "design choice", "if_wrong": "CI endpoints carry Monte Carlo error"},
    {"knob": "price trends", "value": "not computed", "source": "design choice",
     "if_wrong": "the generator has no price history; random trend numbers would look "
                 "like findings"},
    {"knob": "disclosure floor", "value": "n >= %d per published cell" % report.MIN_CELL_N,
     "source": "design choice (privacy)", "if_wrong": "small cells could identify cards"},
]

README_BLOCK = ("<!-- findings-headline:begin (generated by python -m "
                "cardsegments.synth_findings build; do not edit) -->",
                "<!-- findings-headline:end -->")
DECISIONS_BLOCK = ("<!-- findings-decisions:begin (generated by python -m "
                   "cardsegments.synth_findings build; do not edit) -->",
                   "<!-- findings-decisions:end -->")
METHOD_BLOCK = ("<!-- findings-assumptions:begin (generated by python -m "
                "cardsegments.synth_findings build; do not edit) -->",
                "<!-- findings-assumptions:end -->")

PROVENANCE_NOTE = (
    "Every number in this report comes from data generated by `cardsegments.synth` "
    "(planted segments, planted outcomes), so it shows what the method does when the "
    "truth is known. It says nothing about the real market. The private licensed-data "
    "run is not published until the price-feed terms are read (see DECISIONS.md, "
    "2026-11-04 — Findings regenerated on synthetic data only).")

NULL_CONCLUSION = (
    "mixture criteria respond to non-Gaussian marginals, not only to cluster structure; "
    "the gap statistic and bootstrap stability are therefore the structure arbiters, "
    "and silhouette/BIC are descriptive.")


# ---------------------------------------------------------------- method stages

def screened_features(frame, feature_cols):
    """screen.py's fixed block rule: one representative per |rho| >= 0.8 block."""
    spearman, _pearson = screen.correlation_matrices(frame, feature_cols)
    blocks = screen.feature_blocks(spearman)
    kept, drops = screen.prune_blocks(frame, feature_cols, blocks)
    return kept, drops


def v4_frame(complete, kept):
    """Price kept as one axis; every other kept axis residualized on it.

    Returns (frame with the residual columns added, V4 feature list)."""
    residualize_cols = []
    for column in kept:
        if column != PRICE_COL:
            residualize_cols.append(column)
    fits = screen.fit_price_residualization(complete, residualize_cols)
    residualized = screen.apply_price_residualization(complete, fits, prefix=RESID_PREFIX)
    v4_cols = [PRICE_COL]
    for column in residualize_cols:
        v4_cols.append(RESID_PREFIX + column[2:])
    return residualized, v4_cols


def scan_table_rows(tables):
    """Silhouette and gap scan tables as plain rows for the figure."""
    sil_rows = []
    silhouette_items = list(tables["silhouette"]["silhouette"].items())
    for k, value in silhouette_items:
        if value != value:
            silhouette_value = None
        else:
            silhouette_value = float(value)
        sil_rows.append({"k": int(k), "silhouette": silhouette_value})
    gap_rows = []
    gap_index = list(tables["gap"].index)
    for k in gap_index:
        gap_rows.append({"k": int(k), "gap": float(tables["gap"].loc[k, "gap"]),
                         "s_k": float(tables["gap"].loc[k, "s_k"])})
    return {"silhouette": sil_rows, "gap": gap_rows}


def era_share_rows(era_values, labels):
    """Within-cluster era shares as whole-percent rows (report disclosure)."""
    _counts, pct = profiles.era_crosstab(era_values, labels)
    rows = []
    cluster_ids = list(pct.index)
    era_names = list(pct.columns)
    for cluster_id in cluster_ids:
        n_cluster = int(np.sum(labels == cluster_id))
        row = {"cluster": int(cluster_id), "n": n_cluster}
        for era in era_names:
            row[era] = float(round(pct.loc[cluster_id, era]))
        rows.append(row)
    return rows


def treat_variant(name, complete, X, profile_cols, settings):
    """Selection, stability, profiles, era lifts and robustness for one matrix.

    Returns (bundle block, full-data labels). The profile is on the ORIGINAL
    feature scale (profile_cols), as in the private run."""
    selection, tables = run_selection(name, X, k_values=settings["k_values"],
                                      n_boot=settings["stability_b"],
                                      gap_b=settings["gap_b"])
    selection.pop("tiebreaks", None)
    solution = selection["solution"]
    cov_type = solution.get("cov_type") or None
    full_labels, resamples, ari_summary = stability.bootstrap_resample_labels(
        X, solution["method"], solution["k"], cov_type, n_boot=settings["stability_b"])
    profile_global = profiles.global_profile(complete, profile_cols)
    profile_clusters = profiles.cluster_profile_table(complete, full_labels, profile_cols)
    total_n = 0
    for cluster in profile_clusters:
        total_n += cluster["n"]
    era_values = complete["era_label"].to_numpy()
    lift_rows = profiles.era_lift_with_cis(era_values, full_labels, resamples)
    cramers = profiles.cramers_v(era_values, full_labels)
    robust_k = max(solution["k"], 2)
    robustness = profiles.robustness_cells(X, full_labels, solution["method"], robust_k,
                                           cov_type, k_max=max(settings["k_values"]))
    block = {
        "n_rows": int(len(complete)),
        "n_complete": int(len(complete)),
        "p": int(X.shape[1]),
        "feature_cols": list(profile_cols),
        "model_feature_cols": [],
        "selection": selection,
        "stability": ari_summary,
        "scan_tables": scan_table_rows(tables),
        "profile": {"global": profile_global, "clusters": profile_clusters,
                    "clusters_total_n": total_n, "feature_cols": list(profile_cols)},
        "era": {"cluster_shares_pct": era_share_rows(era_values, full_labels),
                "lift": lift_rows, "cramers_v": cramers},
        "trends": {
            "protocol": "Not computed on synthetic data: the generator draws no price "
                        "history, so there is no trailing price change to summarize. The "
                        "protocol (median 90 d / 365 d log change per segment, >= 5 sales "
                        "in each endpoint window) runs only on the private data.",
            "d90": None, "d365": None},
        "cluster_names": {},
        "robustness": robustness,
    }
    return block, full_labels


def sensitivity_row(name, X, settings):
    """Selection-only summary of one sensitivity matrix."""
    selection, _tables = run_selection(name, X, k_values=settings["k_values"],
                                       n_boot=settings["stability_b"],
                                       gap_b=settings["gap_b"])
    return {
        "name": name, "n": int(X.shape[0]), "p": int(X.shape[1]),
        "silhouette_k": selection["kmeans"]["silhouette_k"],
        "gap_k": selection["kmeans"]["gap_k"],
        "gmm_cov": selection["gmm"]["chosen"]["cov"],
        "gmm_k": selection["gmm"]["chosen"]["k"],
        "solution": selection["solution"],
        "weak_structure": selection["weak_structure"],
        "max_silhouette": selection["max_silhouette"],
    }


def planted_names(found_labels, planted_labels, survives):
    """Name each surviving cluster by the planted segment it overlaps most.

    Rule (fixed): name = "planted segment P (P's share of this cluster)".
    Clusters failing a robustness cell get no name (report.py prints them as
    unnamed). Failure mode: two found clusters can map to one planted
    segment; both names then say so, which is the honest reading."""
    names = {}
    found_ids = sorted(set(found_labels.tolist()))
    for cluster_id in found_ids:
        survived = survives.get(int(cluster_id), survives.get(str(cluster_id)))
        if not survived:
            continue
        in_cluster = found_labels == cluster_id
        members = planted_labels[in_cluster]
        counts = np.bincount(members)
        best = int(np.argmax(counts))
        share = float(counts[best]) / float(len(members))
        names[int(cluster_id)] = "planted segment %d (%d%% of members)" % (
            best, int(round(100.0 * share)))
    return names


def gem_multiplier_stats(frame):
    """Spearman rho(gem rate, log multiplier), overall and within era."""
    pair = frame[["f_gem_rate_pooled", "f_log_multiplier", "era_label"]].dropna()
    rho_overall = float(pair["f_gem_rate_pooled"].corr(pair["f_log_multiplier"],
                                                       method="spearman"))
    within = {}
    for era in synth.ERA_ORDER:
        era_pair = pair[pair["era_label"] == era]
        rho = float(era_pair["f_gem_rate_pooled"].corr(era_pair["f_log_multiplier"],
                                                       method="spearman"))
        within[era] = {"rho": rho, "n": int(len(era_pair))}
    return {"rho_overall": rho_overall, "n_overall": int(len(pair)), "within_era": within}


def price_dominance_stats(frame, complete, kept):
    """Top R^2-on-price feature and |corr(PC1, log raw price)| on V1."""
    r2_table = screen.r2_on_price(frame, kept)
    top = r2_table.iloc[0]
    raw = complete[kept].to_numpy(dtype=float)
    centered = raw - raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    standardized = centered / scale
    _u, _s, vt = np.linalg.svd(standardized, full_matrices=False)
    pc1 = standardized @ vt[0]
    price = complete[PRICE_COL].to_numpy(dtype=float)
    pc1_corr = float(abs(np.corrcoef(pc1, price)[0, 1]))
    return {"top_r2_feature": str(top["feature"]), "top_r2_on_price": float(top["r2_on_price"]),
            "pc1_abs_corr_with_price": pc1_corr}


def era_argument_text(v4_block):
    """One code-built paragraph on the era lifts (no hand-typed numbers)."""
    best = {}
    for row in v4_block["era"]["lift"]:
        if row.get("ci_low") is None or row.get("lift") is None:
            continue
        if row["ci_low"] <= 1.0:
            continue
        current = best.get(row["era"])
        if current is None or row["lift"] > current["lift"]:
            best[row["era"]] = row
    parts = []
    for era in synth.ERA_ORDER:
        row = best.get(era)
        if row is None:
            parts.append("no cluster is %s-enriched (no lift CI above 1)" % era)
            continue
        parts.append("cluster %d is the strongest %s-enriched cluster (lift %s, CI [%s, %s])"
                     % (row["cluster"], era, report.sig(row["lift"]),
                        report.sig(row["ci_low"]), report.sig(row["ci_high"])))
    return ("SYNTHETIC: era labels were planted as a different era mixture per segment "
            "(synth.ERA_MIXES) and never entered the features. On V4, " + "; ".join(parts)
            + ". Recovering a planted mixture checks the lift and CI code; it is not a claim "
              "about how real eras sort into segments.")


def supervised_case(signal, settings):
    """run_supervised on the illustrative preset (or the smoke-size version)."""
    name = "synthetic_signal_%s" % str(signal).replace(".", "p")
    frame, manifest, _truth = synth.generate_supervised(
        n_rows=settings["supervised_n_rows"], n_clusters=3, separation=2.5,
        seed=settings["supervised_seed"], cluster_signal=signal, name=name,
        outcome_mode="threshold",
        groups_per_cluster=settings["supervised_groups_per_cluster"],
        target_base_rate=synth.THRESHOLD_TARGET_BASE_RATE)
    results = run_supervised(
        frame, manifest["feature_cols"], manifest["baseline_cols"],
        cluster_feature_cols=manifest["feature_cols"],
        cluster_spec={"method": "kmeans", "k": settings["cluster_k"]},
        n_seeds=settings["n_seeds"], b_group_boot=settings["b_group_boot"],
        b_coef_boot=settings["b_coef_boot"], calibration_min_bin_n=report.MIN_CELL_N)
    results["cluster_signal"] = float(signal)
    results["y_liquid_threshold"] = synth.LIQUID_MIN_SOLD
    results["outcome_note"] = (
        "Outcome (SYNTHETIC): y_liquid = 1 when the generator's drawn sold count is at "
        "least %d; the count is Poisson with a planted segment effect of %s (no calendar "
        "window)." % (synth.LIQUID_MIN_SOLD, report.sig(float(signal))))
    return results


# ---------------------------------------------------------------- predictions

def predictions_sha(root):
    path = os.path.join(root, PREDICTIONS_REL)
    with open(path, "rb") as handle:
        data = handle.read()
    return hashlib.sha256(data).hexdigest()[:SHA_PREFIX_LEN]


def verdict_word(holds):
    if holds is None:
        return "not decidable"
    if holds:
        return "confirmed"
    return "refuted"


def find_sensitivity(bundle, name):
    sensitivities = bundle.get("sensitivities", [])
    for row in sensitivities:
        if row["name"] == name:
            return row
    return None


def judge_predictions(bundle):
    """Verdict per pre-registered prediction, decided by its written rule.

    Each rule reads bundle numbers only, so a hand-made bundle can test it
    (tests/test_synth_findings.py flips each deciding number to show the
    verdict flips too). Failure mode: a rule is only as good as the number
    it reads; a wrong bundle key raises KeyError instead of passing.
    """
    rows = []
    v1 = bundle["variants"]["v1"]
    v4 = bundle["variants"]["v4"]
    planted = bundle["planted_truth"]

    gap_k = v1["selection"]["kmeans"]["gap_k"]
    rows.append({"id": 1, "summary": "V1 gap statistic recovers the planted k",
                 "holds": gap_k == planted["k"],
                 "deciding_statistic": "gap one-SE k=%d, planted k=%d" % (gap_k, planted["k"])})

    ari = planted["v1_ari"]
    rows.append({"id": 2, "summary": "V1 solution recovers the planted labels (ARI >= %.2f)"
                                     % PLANTED_ARI_MIN,
                 "holds": ari >= PLANTED_ARI_MIN,
                 "deciding_statistic": "planted-truth ARI %s" % report.sig(ari)})

    stability_mean = v1["stability"]["mean_ari"]
    rows.append({"id": 3, "summary": "V1 solution is stable (bootstrap mean ARI >= %.2f)"
                                     % STABILITY_ARI_MIN,
                 "holds": stability_mean >= STABILITY_ARI_MIN,
                 "deciding_statistic": "bootstrap mean ARI %s (B=%d)"
                                       % (report.sig(stability_mean), bundle["stability_b"])})

    summaries = bundle["null_control"]["summaries"]
    mvn_k = summaries["mvn_null"]["gap_one_se_k"]
    shuffle_k = summaries["shuffle_null"]["gap_one_se_k"]
    rows.append({"id": 4, "summary": "both null controls find no structure (gap k=1)",
                 "holds": mvn_k == 1 and shuffle_k == 1,
                 "deciding_statistic": "gap one-SE k: matched-Gaussian %d, column-shuffled %d"
                                       % (mvn_k, shuffle_k)})

    weak = find_sensitivity(bundle, "v2_weak_separation")
    if weak is None:
        weak_holds = None
        weak_text = "sensitivity missing"
    else:
        weak_holds = weak["gap_k"] == 1
        weak_text = "gap one-SE k=%d at separation %s" % (
            weak["gap_k"], report.sig(bundle["settings"]["weak_separation"]))
    rows.append({"id": 5, "summary": "weak separation reads as a continuum (gap k=1)",
                 "holds": weak_holds, "deciding_statistic": weak_text})

    enriched = {"vintage": [], "modern": []}
    for row in v4["era"]["lift"]:
        if row["era"] not in enriched:
            continue
        if row.get("ci_low") is not None and row["ci_low"] > 1.0:
            enriched[row["era"]].append(row)
    era_parts = []
    era_names = ["vintage", "modern"]
    for era in era_names:
        era_rows = enriched[era]
        if era_rows:
            first = era_rows[0]
            era_parts.append("%s: cluster %d lift %s CI [%s, %s]"
                             % (era, first["cluster"], report.sig(first["lift"]),
                                report.sig(first["ci_low"]), report.sig(first["ci_high"])))
        else:
            era_parts.append("%s: no lift CI above 1" % era)
    era_holds = bool(enriched["vintage"]) and bool(enriched["modern"])
    rows.append({"id": 6, "summary": "era composition recovered without era features (V4)",
                 "holds": era_holds, "deciding_statistic": "; ".join(era_parts)})

    rho = bundle["gem_multiplier"]["rho_overall"]
    rows.append({"id": 7, "summary": "gem rate is a nuisance axis "
                                     "(abs Spearman rho with multiplier < %.2f)"
                                     % NUISANCE_RHO_MAX,
                 "holds": abs(rho) < NUISANCE_RHO_MAX,
                 "deciding_statistic": "Spearman rho %s (n=%d)"
                                       % (report.sig(rho), bundle["gem_multiplier"]["n_overall"])})

    null_case = bundle["supervised"]["primary"]
    planted_case = bundle["supervised"]["sensitivity"]
    rows.append({"id": 8, "summary": "honest null: the nested gate FAILS at signal 0",
                 "holds": not null_case["nested_delta_auc"]["ships"],
                 "deciding_statistic": gate_text(null_case)})
    rows.append({"id": 9, "summary": "the nested gate PASSES at signal 0.9",
                 "holds": bool(planted_case["nested_delta_auc"]["ships"]),
                 "deciding_statistic": gate_text(planted_case)})

    r2_values = [null_case["forward_regression"].get("oof_r2"),
                 planted_case["forward_regression"].get("oof_r2")]
    if r2_values[0] is None or r2_values[1] is None:
        r2_holds = None
    else:
        r2_holds = r2_values[0] <= 0 and r2_values[1] <= 0
    rows.append({"id": 10, "summary": "planted-noise forward price: out-of-fold R^2 <= 0",
                 "holds": r2_holds,
                 "deciding_statistic": "R^2 %s (signal 0), %s (signal 0.9)"
                                       % (report.sig(r2_values[0]), report.sig(r2_values[1]))})

    judged = []
    for row in rows:
        judged.append({"id": row["id"], "summary": row["summary"],
                       "verdict": verdict_word(row["holds"]),
                       "deciding_statistic": row["deciding_statistic"]})
    return judged


def gate_text(case):
    nested = case["nested_delta_auc"]
    boot = nested["group_bootstrap"]
    return "mean dAUC %s, CI [%s, %s], ships=%s" % (
        report.sig(nested["mean_over_seeds"]), report.sig(boot["ci_low"]),
        report.sig(boot["ci_high"]), nested["ships"])


# ---------------------------------------------------------------- the bundle

def supervised_notes(null_case, planted_case):
    """Design notes, numbers read from the two supervised blocks."""
    missing = null_case["n_rows"] - null_case["n_cluster_feature_complete"]
    return [
        "Realized base rates %s (signal 0) and %s (signal 0.9) against the planted %s; the "
        "positive liquid class is the minority, so PR-AUC is oriented to it."
        % (report.sig(null_case["base_rate"]), report.sig(planted_case["base_rate"]),
           report.sig(synth.THRESHOLD_TARGET_BASE_RATE)),
        "The generator removes the gem-rate cluster feature from about 30%% of rows (%d of "
        "%d at signal 0); they carry the explicit no_cluster category, so the nested "
        "comparison runs on the full cohort." % (missing, null_case["n_rows"]),
        "b_log1p_sold_365d equals f_log1p_sales_365 by construction in the generator, so "
        "the velocity baseline and that feature column are the same information.",
        "fwd_log_price_change is planted as pure noise on about 20%% of rows (%d eligible at "
        "signal 0); its R^2 is a null control, not a finding."
        % null_case["forward_regression"]["n_eligible"],
    ]


LIMITATIONS = [
    "Synthetic data only: the generator's distributions are design choices (synth.py "
    "docstring); nothing here measures the real market.",
    "The planted structure is Gaussian on the log scale with a known k; real markets need "
    "not be, which is why the null controls and the weak-separation sensitivity are shown.",
    "No price history is generated, so per-segment trends are not computed.",
    "The in-fold clustering freezes k at the preset's planted k (3) - a mild selection "
    "leak, disclosed in Design decisions.",
    "The private licensed-data run and its numbers are not in this repo (see DECISIONS.md, "
    "2026-11-04 — Findings regenerated on synthetic data only).",
]


def hero_text(planted, null_case, planted_case):
    """The hero figure's one-line title; short enough to fit 900 px."""
    null_gate = null_case["nested_delta_auc"]
    planted_gate = planted_case["nested_delta_auc"]
    return ("SYNTHETIC: planted k=%d (V1 ARI %s); nested gate dAUC %s at signal 0 (%s), "
            "%s at 0.9 (%s)"
            % (planted["k"], report.sig(planted["v1_ari"]),
               report.sig(null_gate["mean_over_seeds"]), gate_word(null_gate),
               report.sig(planted_gate["mean_over_seeds"]), gate_word(planted_gate)))


def gate_word(nested):
    if nested["ships"]:
        return "passes"
    return "null"


def refresh_text(bundle):
    """Re-derive every code-built text field from the bundle's own numbers.

    Used by the `retext` stage when wording rules change without any number
    changing; build_bundle produces the same fields on a fresh run."""
    null_case = bundle["supervised"]["primary"]
    planted_case = bundle["supervised"]["sensitivity"]
    bundle["headline"]["hero_validation_text"] = hero_text(
        bundle["planted_truth"], null_case, planted_case)
    bundle["era_argument"] = era_argument_text(bundle["variants"]["v4"])
    bundle["supervised"]["notes"] = supervised_notes(null_case, planted_case)
    bundle["provenance"]["note"] = PROVENANCE_NOTE
    bundle["limitations"] = list(LIMITATIONS)
    bundle["predictions"] = judge_predictions(bundle)
    return bundle


def build_bundle(settings, root, verbose=False):
    """Run every stage on synthetic data; return the bundle (not yet written)."""
    frame, manifest, truth = synth.generate_segmented(
        n_rows=settings["seg_n_rows"], n_clusters=settings["seg_k"],
        separation=settings["seg_separation"], seed=settings["seg_seed"], missingness=True)
    feature_cols = manifest["feature_cols"]
    kept, _drops = screened_features(frame, feature_cols)
    complete_mask = frame[kept].notna().all(axis=1).to_numpy()
    complete, X1 = complete_case_matrix(frame, kept)
    planted_labels = np.asarray(truth["labels"])[complete_mask]
    if verbose:
        print("[findings] V1: n=%d complete, p=%d" % (len(complete), len(kept)))

    v1_block, v1_labels = treat_variant("v1", complete, X1, kept, settings)
    v1_block["model_feature_cols"] = list(kept)
    v1_ari = float(adjusted_rand_score(planted_labels, v1_labels))

    residualized, v4_cols = v4_frame(complete, kept)
    _v4_complete, X4 = complete_case_matrix(residualized, v4_cols)
    v4_block, v4_labels = treat_variant("v4", complete, X4, kept, settings)
    v4_block["model_feature_cols"] = list(v4_cols)
    v4_block["cluster_names"] = planted_names(v4_labels, planted_labels,
                                              v4_block["robustness"]["survives"])
    v1_block["cluster_names"] = planted_names(v1_labels, planted_labels,
                                              v1_block["robustness"]["survives"])
    if verbose:
        print("[findings] V1 and V4 treated")

    sensitivities = []
    weak_frame, weak_manifest, _weak_truth = synth.generate_segmented(
        n_rows=settings["seg_n_rows"], n_clusters=settings["seg_k"],
        separation=settings["weak_separation"], seed=settings["seg_seed"], missingness=True)
    _weak_complete, X_weak = complete_case_matrix(weak_frame, kept)
    sensitivities.append(sensitivity_row("v2_weak_separation", X_weak, settings))
    selected_frame, _sel_manifest, _sel_truth = synth.generate_segmented(
        n_rows=settings["seg_n_rows"], n_clusters=settings["seg_k"],
        separation=settings["seg_separation"], seed=settings["seg_seed"], missingness=True,
        selection_effect=True)
    _selected_complete, X_selected = complete_case_matrix(selected_frame, kept)
    sensitivities.append(sensitivity_row("v3_selection_effect", X_selected, settings))
    core_cols = []
    for column in kept:
        if column not in NUISANCE_COLS:
            core_cols.append(column)
    _core_complete, X_core = complete_case_matrix(frame, core_cols)
    sensitivities.append(sensitivity_row("v1_core", X_core, settings))
    if verbose:
        print("[findings] sensitivities done")

    raw = complete[kept].to_numpy(dtype=float)
    null_summaries, null_tables = nullcontrol.run_null_control(
        raw, settings["k_values"], gap_b=settings["gap_b"], real_label=V1_REAL_LABEL)
    null_scan_tables = {}
    null_labels = list(null_tables.keys())
    for label in null_labels:
        null_scan_tables[label] = scan_table_rows(null_tables[label])
    if verbose:
        print("[findings] null control done")

    signals = settings["signals"]
    null_case = supervised_case(signals[0], settings)
    if verbose:
        print("[findings] supervised signal %s done" % signals[0])
    planted_case = supervised_case(signals[1], settings)
    if verbose:
        print("[findings] supervised signal %s done" % signals[1])

    planted = {"k": int(settings["seg_k"]), "separation": float(settings["seg_separation"]),
               "v1_ari": v1_ari, "n": int(len(complete))}
    robustness = {"v1": v1_block.pop("robustness"), "v4": v4_block.pop("robustness")}
    bundle = {
        "bundle_version": 2,
        "provenance": {"label": report.SYNTHETIC_LABEL, "generator": "cardsegments.synth",
                       "note": PROVENANCE_NOTE},
        "anchor_date": "SYNTHETIC",
        "settings": dict(settings),
        "stability_b": int(settings["stability_b"]),
        "disclosure": {"min_cell_n": report.MIN_CELL_N},
        "planted_truth": planted,
        "variants": {"v1": v1_block, "v4": v4_block},
        "sensitivities": sensitivities,
        "price_dominance": price_dominance_stats(frame, complete, kept),
        "gem_multiplier": gem_multiplier_stats(frame),
        "robustness": robustness,
        "null_control": {"summaries": null_summaries, "scan_tables": null_scan_tables,
                         "methodological_conclusion": NULL_CONCLUSION},
        "supervised": {
            "primary_anchor": "SYNTHETIC",
            "sensitivity_anchor": "SYNTHETIC",
            "primary_title": "SYNTHETIC illustrative preset, cluster_signal 0 (no planted "
                             "segment effect)",
            "sensitivity_title": "SYNTHETIC illustrative preset, cluster_signal 0.9 "
                                 "(planted segment effect)",
            "primary": null_case,
            "sensitivity": planted_case,
            "notes": supervised_notes(null_case, planted_case),
        },
        "era_argument": era_argument_text(v4_block),
        "limitations": list(LIMITATIONS),
        "headline": {"hero_validation_text": hero_text(planted, null_case, planted_case)},
        "why_this_metric": WHY_THIS_METRIC,
    }
    # disclosure and rounding first, so the verdicts are judged on exactly the
    # numbers the committed bundle carries (bundle_problems re-judges them)
    bundle = report.apply_disclosure(bundle)
    bundle = report.clean_nan(ladder.clean_for_json(bundle))
    bundle["predictions"] = judge_predictions(bundle)
    bundle["predictions_registration"] = {"date": PREDICTIONS_DATE,
                                          "sha256": predictions_sha(root),
                                          "file": PREDICTIONS_REL}
    return bundle


def write_bundle(bundle, path):
    text = json.dumps(bundle, indent=1, sort_keys=True)
    with open(path, "w") as handle:
        handle.write(text + "\n")
    return path


# ---------------------------------------------------------------- generated text

def count_verdicts(bundle):
    counts = {"confirmed": 0, "refuted": 0, "not decidable": 0}
    for row in bundle["predictions"]:
        counts[row["verdict"]] += 1
    return counts


def chosen_text(selection):
    text, boot = report.chosen_solution_bootstrap(selection)
    return text, boot


def readme_block(bundle):
    """The README's synthetic findings section, every number from the bundle."""
    v1 = bundle["variants"]["v1"]
    v4 = bundle["variants"]["v4"]
    planted = bundle["planted_truth"]
    v1_selection = v1["selection"]
    v1_solution_text, _v1_boot = chosen_text(v1_selection)
    v4_solution_text, v4_boot = chosen_text(v4["selection"])
    summaries = bundle["null_control"]["summaries"]
    weak = find_sensitivity(bundle, "v2_weak_separation")
    null_case = bundle["supervised"]["primary"]
    planted_case = bundle["supervised"]["sensitivity"]
    counts = count_verdicts(bundle)
    lines = []
    lines.append("## Headline findings (SYNTHETIC v2, from the committed bundle)")
    lines.append("")
    lines.append("Every number below is read from [`reports/results_bundle.json`]"
                 "(reports/results_bundle.json), computed on generated data with planted "
                 "truth; the full report is [`reports/findings.md`](reports/findings.md). "
                 "None of it is a claim about the real market.")
    lines.append("")
    lines.append("- **%s** On V1 (n=%d, %d features, "
                 "planted k=%d at separation %s) the gap statistic (one-SE) picks k=%d and "
                 "the chosen solution (%s) matches the planted labels with ARI %s; its "
                 "bootstrap mean ARI is %s (B=%d). The price-residualized V4 picks %s "
                 "(bootstrap mean ARI %s)."
                 % (structure_lead(bundle), v1_selection["n"], v1_selection["p"], planted["k"],
                    report.sig(planted["separation"]), v1_selection["kmeans"]["gap_k"],
                    v1_solution_text, report.sig(planted["v1_ari"]),
                    report.sig(v1["stability"]["mean_ari"]), bundle["stability_b"],
                    v4_solution_text, report.sig(v4_boot["mean_ari"])))
    lines.append("- **It does not invent segments.** The gap statistic picks k=%d on the "
                 "matched-Gaussian null and k=%d on the column-shuffled null; BIC picks "
                 "%s k=%d and %s k=%d there, which is why mixture criteria stay "
                 "descriptive. At separation %s the gap statistic picks k=%d."
                 % (summaries["mvn_null"]["gap_one_se_k"],
                    summaries["shuffle_null"]["gap_one_se_k"],
                    summaries["mvn_null"]["bic_choice"]["cov"],
                    summaries["mvn_null"]["bic_choice"]["k"],
                    summaries["shuffle_null"]["bic_choice"]["cov"],
                    summaries["shuffle_null"]["bic_choice"]["k"],
                    report.sig(bundle["settings"]["weak_separation"]), weak["gap_k"]))
    lines.append("- **The honest null, on synthetic data.** With no planted segment effect "
                 "(cluster_signal 0, n=%d, base rate %s) the nested gate (era + velocity + "
                 "cluster over era + velocity) gives %s: %s. With the planted effect "
                 "(cluster_signal 0.9) it gives %s: %s."
                 % (null_case["n_rows"], report.sig(null_case["base_rate"]),
                    ship_word(null_case), gate_text(null_case), ship_word(planted_case),
                    gate_text(planted_case)))
    lines.append("- Pre-registered predictions ([`%s`](%s), sha256 prefix `%s`): %d "
                 "confirmed, %d refuted, %d not decidable (verdicts by code, listed in "
                 "the findings report)."
                 % (PREDICTIONS_REL, PREDICTIONS_REL,
                    bundle["predictions_registration"]["sha256"], counts["confirmed"],
                    counts["refuted"], counts["not decidable"]))
    lines.append("")
    lines.append(report.why_md(bundle["why_this_metric"]["nested_gate"]))
    return "\n".join(lines)


def structure_lead(bundle):
    """Bold lead of the structure bullet, chosen by rule from the bundle."""
    v1_selection = bundle["variants"]["v1"]["selection"]
    planted = bundle["planted_truth"]
    gap_hits = v1_selection["kmeans"]["gap_k"] == planted["k"]
    chosen_hits = v1_selection["solution"]["k"] == planted["k"]
    if gap_hits and planted["v1_ari"] >= PLANTED_ARI_MIN:
        return "The pipeline recovers the planted segments."
    if chosen_hits:
        return ("The gap statistic misses the planted k; the stability tiebreak still "
                "lands on it (a loss on prediction 1, shipped as printed).")
    return "The pipeline misses the planted segments (shipped as printed)."


def ship_word(case):
    if case["nested_delta_auc"]["ships"]:
        return "PASSES"
    return "FAILS (null)"


def decisions_block(bundle):
    """Dated decision entries; numbers from the bundle."""
    null_case = bundle["supervised"]["primary"]
    planted_case = bundle["supervised"]["sensitivity"]
    summaries = bundle["null_control"]["summaries"]
    lines = []
    lines.append("### 2026-11-04 — Findings regenerated on synthetic data only")
    lines.append("Decided to replace the licensed-feed findings, bundle and figures with a "
                 "synthetic v2 (the real-data text moved to a local private folder outside "
                 "the repo) because the feed terms are unread and any tracked aggregate is "
                 "published by a push; measurement: the same method on the illustrative "
                 "preset still tells the honest-null story, the nested gate %s at signal 0 "
                 "(%s) and %s at signal 0.9 (%s). Alternative (keep the real bundle with a "
                 "'private' label) rejected: a label does not stop a push; alternative (wait "
                 "for the terms) would hold the whole repo for one report."
                 % (ship_word(null_case), gate_text(null_case), ship_word(planted_case),
                    gate_text(planted_case)))
    lines.append("")
    lines.append("### 2026-11-04 — Signal 0 is the primary supervised case")
    lines.append("Decided to print cluster_signal 0 first and 0.9 second because the null is "
                 "the case the method must be honest about; measurement: base rate %s vs %s, "
                 "so the two cases differ in the planted segment effect, not in size. "
                 "Alternative (show only the planted case) would hide that the gate's verdict "
                 "depends on it."
                 % (report.sig(null_case["base_rate"]), report.sig(planted_case["base_rate"])))
    lines.append("")
    lines.append("### 2026-11-04 — No synthetic price trends")
    lines.append("Decided to leave the per-segment trend tables out of the synthetic report "
                 "because the generator has no price history; alternative (the demo's random "
                 "changes with bootstrap CIs) lost because it prints numbers that read like "
                 "market findings and measure nothing.")
    lines.append("")
    v1_selection = bundle["variants"]["v1"]["selection"]
    planted = bundle["planted_truth"]
    lines.append("### 2026-11-04 — Gap statistic stays the structure arbiter")
    lines.append("Decided to keep the gap statistic and bootstrap stability as the arbiters "
                 "and silhouette/BIC as descriptive because on the synthetic nulls BIC picks "
                 "%s k=%d (matched Gaussian) and %s k=%d (column-shuffled) while the gap "
                 "statistic picks k=%d and k=%d; alternative (BIC as arbiter) would call "
                 "structure in a null whenever it picks k > 1. The cost, measured on the "
                 "planted V1 data: the gap statistic picks k=%d against a planted k=%d at "
                 "separation %s (prediction 1: %s), so it is conservative; the rule is kept "
                 "and the miss ships as printed rather than retuning the arbiter after "
                 "seeing it."
                 % (summaries["mvn_null"]["bic_choice"]["cov"],
                    summaries["mvn_null"]["bic_choice"]["k"],
                    summaries["shuffle_null"]["bic_choice"]["cov"],
                    summaries["shuffle_null"]["bic_choice"]["k"],
                    summaries["mvn_null"]["gap_one_se_k"],
                    summaries["shuffle_null"]["gap_one_se_k"],
                    v1_selection["kmeans"]["gap_k"], planted["k"],
                    report.sig(planted["separation"]),
                    prediction_verdict(bundle, 1)))
    lines.append("")
    lines.append("### 2026-11-04 — Real-cohort identifiers checked locally, not in a public test")
    lines.append("Decided that the check 'no tracked file carries the private cohort's n or "
                 "base rate' runs in the local deny-list check (outside the repo, before "
                 "every push) because a public test would have to contain those values; "
                 "alternative (a public test with hashed values) rejected: a short number's "
                 "hash is easy to reverse by trying every value.")
    return "\n".join(lines)


def prediction_verdict(bundle, prediction_id):
    for row in bundle["predictions"]:
        if row["id"] == prediction_id:
            return row["verdict"]
    raise KeyError(prediction_id)


def assumptions_table():
    lines = ["| knob | value | source | what changes if wrong |", "|---|---|---|---|"]
    for row in ASSUMPTIONS:
        lines.append("| %s | %s | %s | %s |" % (row["knob"], row["value"], row["source"],
                                                 row["if_wrong"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- files

def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def load_bundle(root=None):
    if root is None:
        root = repo_root()
    with open(os.path.join(root, BUNDLE_REL)) as handle:
        return json.load(handle)


def read_text(root, rel_path):
    with open(os.path.join(root, rel_path)) as handle:
        return handle.read()


def expected_outputs(root=None):
    """{relative path: full expected text} for every file this module generates."""
    from cardsegments.ladder_report import replace_block

    if root is None:
        root = repo_root()
    bundle = load_bundle(root)
    outputs = {}
    outputs[FINDINGS_REL] = report.findings_text(bundle)
    outputs["README.md"] = replace_block(read_text(root, "README.md"), README_BLOCK,
                                         readme_block(bundle))
    outputs["DECISIONS.md"] = replace_block(read_text(root, "DECISIONS.md"), DECISIONS_BLOCK,
                                            decisions_block(bundle))
    outputs["docs/METHOD.md"] = replace_block(read_text(root, "docs/METHOD.md"), METHOD_BLOCK,
                                              assumptions_table())
    return outputs


def bundle_problems(root=None):
    """Reasons the committed bundle is not a publishable synthetic v2 (empty = fine)."""
    if root is None:
        root = repo_root()
    bundle = load_bundle(root)
    problems = []
    if not report.is_synthetic(bundle):
        problems.append("bundle provenance label is not 'synthetic'")
    if bundle.get("why_this_metric") != WHY_THIS_METRIC:
        problems.append("bundle why_this_metric differs from synth_findings.WHY_THIS_METRIC")
    registration = bundle.get("predictions_registration", {})
    if registration.get("sha256") != predictions_sha(root):
        problems.append("predictions file changed after registration")
    rejudged = judge_predictions(bundle)
    if rejudged != bundle.get("predictions"):
        problems.append("prediction verdicts differ from judge_predictions")
    refreshed = refresh_text(json.loads(json.dumps(bundle)))
    if refreshed != bundle:
        problems.append("a code-built text field differs from refresh_text")
    figures_dir = os.path.join(root, "reports", "figures")
    for name in FIGURE_NAMES:
        if not os.path.exists(os.path.join(figures_dir, name)):
            problems.append("missing figure " + name)
    return problems


def stale_files(root=None):
    if root is None:
        root = repo_root()
    outputs = expected_outputs(root)
    stale = []
    paths = sorted(outputs.keys())
    for rel_path in paths:
        full_path = os.path.join(root, rel_path)
        if not os.path.exists(full_path):
            stale.append(rel_path)
            continue
        with open(full_path) as handle:
            current = handle.read()
        if current != outputs[rel_path]:
            stale.append(rel_path)
    return stale


def render_reports(root):
    """findings.md + figures from the committed bundle, then the doc blocks."""
    bundle = load_bundle(root)
    reports_dir = os.path.join(root, "reports")
    written = render_into(bundle, reports_dir)
    outputs = expected_outputs(root)
    paths = sorted(outputs.keys())
    for rel_path in paths:
        if rel_path == FINDINGS_REL:
            continue
        with open(os.path.join(root, rel_path), "w") as handle:
            handle.write(outputs[rel_path])
        written.append(rel_path)
    return written


def render_into(bundle, out_dir):
    """render_findings, refusing any bundle not labelled synthetic (in any
    directory, not only the repo's reports/)."""
    if not report.is_synthetic(bundle):
        raise ValueError("refusing to render an unlabelled bundle")
    return report.render_findings(bundle, out_dir)


def build(root=None, settings=None, verbose=True):
    if root is None:
        root = repo_root()
    if settings is None:
        settings = FROZEN_SETTINGS
    bundle = build_bundle(settings, root, verbose=verbose)
    bundle_path = os.path.join(root, BUNDLE_REL)
    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
    write_bundle(bundle, bundle_path)
    written = render_reports(root)
    for path in written:
        print("wrote", path)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic-only v2 findings (heavy build).")
    parser.add_argument("stage", nargs="?", default=None,
                        choices=["build", "render", "retext"])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the bundle or a generated file is stale")
    args = parser.parse_args(argv)
    if args.check:
        problems = bundle_problems()
        stale = stale_files()
        if stale:
            problems.append("stale: " + ", ".join(stale))
        if problems:
            for problem in problems:
                print(problem)
            sys.exit(1)
        print("synthetic findings are up to date")
        return
    if args.stage == "build":
        build()
        return
    if args.stage == "retext":
        root = repo_root()
        bundle = refresh_text(load_bundle(root))
        write_bundle(bundle, os.path.join(root, BUNDLE_REL))
        print("re-derived text fields in", BUNDLE_REL)
        return
    if args.stage == "render":
        written = render_reports(repo_root())
        for path in written:
            print("wrote", path)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
