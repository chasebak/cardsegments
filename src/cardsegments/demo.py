"""End-to-end demo on SYNTHETIC data: generate -> adapter -> screen ->
cluster selection -> stability -> null control -> profiles/era lifts ->
supervised validation -> figures -> a rendered findings report.

Writes ONLY to ./out/. Demo-sized settings (smaller gap B, 3 CV seeds) are
printed next to each stage; the full-size settings live in the modules.
"""

import json
import os
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cardsegments import nullcontrol, profiles, report, screen, stability, viz
from cardsegments.adapters import byo_csv
from cardsegments.cluster import complete_case_matrix, run_selection
from cardsegments.supervised import run_supervised
from cardsegments.synth import generate_segmented, generate_supervised, write_dataset

DEMO_K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
DEMO_GAP_B = 25
DEMO_NULL_GAP_B = 15
DEMO_STABILITY_B = 20
DEMO_SEEDS = 3


def stage(label, start):
    elapsed = time.time() - start
    print("[demo] %-34s %6.1f s" % (label, elapsed))
    return time.time()


def main():
    overall_start = time.time()
    out_dir = os.path.join(os.getcwd(), "out")
    figures_note = os.path.join(out_dir, "figures")
    os.makedirs(figures_note, exist_ok=True)
    clock = time.time()

    # 1. generate + round-trip through the adapter (the only data seam)
    frame, manifest, truth = generate_segmented(
        n_rows=900, n_clusters=3, separation=2.5, seed=7, missingness=True)
    csv_path, manifest_path = write_dataset(frame, manifest, out_dir)
    frame, manifest = byo_csv.load_dataset(csv_path, manifest_path)
    feature_cols = manifest["feature_cols"]
    print("[demo] synthetic dataset: n=%d, planted k=%d, separation=%.1f"
          % (len(frame), truth["n_clusters"], truth["separation"]))
    clock = stage("generate + adapter round-trip", clock)

    # 2. screening
    spearman, _pearson = screen.correlation_matrices(frame, feature_cols)
    blocks = screen.feature_blocks(spearman)
    kept, drops = screen.prune_blocks(frame, feature_cols, blocks)
    vif = screen.vif_table(frame, kept)
    price_rank = screen.r2_on_price(frame, feature_cols)
    print("[demo] feature blocks -> kept %d of %d features (%d dropped)"
          % (len(kept), len(feature_cols), len(drops)))
    print("[demo] max VIF = %.2f" % float(vif["vif"].max()))
    print("[demo] most price-proxying feature: %s (R^2=%.3f)"
          % (price_rank.iloc[0]["feature"], price_rank.iloc[0]["r2_on_price"]))
    clock = stage("screening (blocks, VIF, R^2)", clock)

    # 3. selection on complete cases
    complete, X = complete_case_matrix(frame, kept)
    selection, tables = run_selection("demo", X, k_values=DEMO_K_VALUES,
                                      n_boot=DEMO_STABILITY_B, gap_b=DEMO_GAP_B)
    solution = selection["solution"]
    print("[demo] selection: kmeans k=%d (%s), gmm %s k=%d -> solution %s k=%d"
          % (selection["kmeans"]["chosen_k"], selection["kmeans"]["how"],
             selection["gmm"]["chosen"]["cov"], selection["gmm"]["chosen"]["k"],
             solution["method"], solution["k"]))
    clock = stage("k selection (gap B=%d)" % DEMO_GAP_B, clock)

    # 4. stability + planted-truth check
    full_labels, resamples, ari_summary = stability.bootstrap_resample_labels(
        X, solution["method"], solution["k"], solution.get("cov_type") or None,
        n_boot=DEMO_STABILITY_B)
    complete_mask = frame[kept].notna().all(axis=1).to_numpy()
    planted = truth["labels"][complete_mask]
    truth_ari = adjusted_rand_score(planted, full_labels)
    print("[demo] planted-truth ARI = %.3f" % truth_ari)
    print("[demo] bootstrap ARI (B=%d): mean=%.3f min=%.3f p5=%.3f"
          % (DEMO_STABILITY_B, ari_summary["mean_ari"],
             ari_summary["min_ari"], ari_summary["p5_ari"]))
    clock = stage("stability bootstrap", clock)

    # 5. null control (identical pipeline, no planted structure)
    raw = complete[kept].to_numpy(dtype=float)
    null_summaries, null_tables = nullcontrol.run_null_control(
        raw, DEMO_K_VALUES, gap_b=DEMO_NULL_GAP_B, real_label="real")
    null_labels = ["real", "mvn_null", "shuffle_null"]
    for label in null_labels:
        summary = null_summaries[label]
        print("[demo] null control %-13s gap k=%d  max sil=%.3f"
              % (label, summary["gap_one_se_k"], summary["max_silhouette"]))
    clock = stage("null control", clock)

    # 6. profiles + era lifts on the chosen solution
    profile_global = profiles.global_profile(complete, kept)
    profile_clusters = profiles.cluster_profile_table(complete, full_labels, kept)
    era_values = complete["era_label"].to_numpy()
    lift_rows = profiles.era_lift_with_cis(era_values, full_labels, resamples)
    cramers = profiles.cramers_v(era_values, full_labels)
    _counts, pct = profiles.era_crosstab(era_values, full_labels)
    print("[demo] Cramer's V (cluster x era) = %.3f" % cramers)
    clock = stage("profiles + era lifts", clock)

    # 7. robustness cells
    robustness = profiles.robustness_cells(
        X, full_labels, solution["method"],
        max(solution["k"], 2), solution.get("cov_type") or None,
        k_max=max(DEMO_K_VALUES))
    clock = stage("robustness cells", clock)

    # 8. supervised validation on the planted-outcome variant
    sup_frame, sup_manifest, _sup_truth = generate_supervised(
        n_rows=2000, n_clusters=3, separation=2.5, seed=11)
    sup_csv, sup_manifest_path = write_dataset(sup_frame, sup_manifest, out_dir)
    sup_frame, sup_manifest = byo_csv.load_dataset(sup_csv, sup_manifest_path)
    supervised_results = run_supervised(
        sup_frame, sup_manifest["feature_cols"], sup_manifest["baseline_cols"],
        cluster_feature_cols=sup_manifest["feature_cols"],
        cluster_spec={"method": "kmeans", "k": 3},
        n_seeds=DEMO_SEEDS, b_group_boot=300, b_coef_boot=60,
        calibration_min_bin_n=50)
    nested = supervised_results["nested_delta_auc"]
    print("[demo] supervised (synthetic, %d seeds): full logistic (L2, C=1) AUC=%.3f, "
          "nested dAUC=%.3f CI=[%.3f, %.3f], ships=%s"
          % (DEMO_SEEDS,
             supervised_results["pooled_auc"]["primary_logistic"]["auc"],
             nested["mean_over_seeds"], nested["group_bootstrap"]["ci_low"],
             nested["group_bootstrap"]["ci_high"], nested["ships"]))
    clock = stage("supervised validation", clock)

    # 9. figures + a rendered report from a demo bundle
    viz.save_pca_density(X, full_labels,
                         os.path.join(figures_note, "pca_synthetic.png"),
                         synthetic=True)
    bundle = build_demo_bundle(kept, selection, tables, ari_summary,
                               profile_global, profile_clusters, pct,
                               lift_rows, cramers, robustness,
                               null_summaries, null_tables, supervised_results,
                               complete, full_labels)
    bundle = report.clean_nan(bundle)
    bundle_path = os.path.join(out_dir, "demo_bundle.json")
    with open(bundle_path, "w") as handle:
        json.dump(bundle, handle, indent=1)
    written = report.render_findings(bundle, out_dir)
    print("[demo] wrote %s and %d report files under out/"
          % (os.path.basename(bundle_path), len(written)))
    clock = stage("figures + rendered report", clock)

    total = time.time() - overall_start
    print("[demo] TOTAL %.1f s (budget: 90 s). All outputs under ./out/ "
          "- synthetic data only." % total)


def era_share_rows(pct, labels):
    rows = []
    cluster_ids = list(pct.index)
    for cluster_id in cluster_ids:
        n_cluster = int(np.sum(labels == cluster_id))
        row = {"cluster": int(cluster_id), "n": n_cluster}
        era_names = list(pct.columns)
        for era in era_names:
            row[era] = float(round(pct.loc[cluster_id, era]))
        rows.append(row)
    return rows


def scan_table_rows(tables):
    sil_rows = []
    for k, value in tables["silhouette"]["silhouette"].items():
        sil_rows.append({"k": int(k), "silhouette": None if value != value
                         else float(value)})
    gap_rows = []
    for k in tables["gap"].index:
        gap_rows.append({"k": int(k), "gap": float(tables["gap"].loc[k, "gap"]),
                         "s_k": float(tables["gap"].loc[k, "s_k"])})
    return {"silhouette": sil_rows, "gap": gap_rows}


def build_demo_bundle(kept, selection, tables, ari_summary, profile_global,
                      profile_clusters, pct, lift_rows, cramers, robustness,
                      null_summaries, null_tables, supervised_results,
                      complete, labels):
    """A bundle in the same schema the real run exports - synthetic data, so
    nothing here is a real-market claim. Exercises report.render_findings."""
    total_n = 0
    for cluster in profile_clusters:
        total_n += cluster["n"]
    profile_block = {"global": profile_global, "clusters": profile_clusters,
                     "clusters_total_n": total_n, "feature_cols": kept}
    era_rows = era_share_rows(pct, labels)
    null_scan_tables = {}
    null_names = list(null_tables.keys())
    for label in null_names:
        null_scan_tables[label] = scan_table_rows(null_tables[label])
    rng = np.random.default_rng(5)
    synthetic_changes = rng.normal(0.0, 0.1, size=len(labels))
    trend_rows = profiles.trend_stats(synthetic_changes, labels)
    variant_block = {
        "n_rows": int(len(complete)), "n_complete": int(len(complete)),
        "p": len(kept), "feature_cols": kept,
        "selection": selection,
        "scan_tables": scan_table_rows(tables),
        "profile": profile_block,
        "era": {"cluster_shares_pct": era_rows, "lift": lift_rows,
                "cramers_v": cramers},
        "trends": {"protocol": "SYNTHETIC demo trends: random changes, "
                               "card-level bootstrap CI.",
                   "d90": trend_rows, "d365": trend_rows},
        "cluster_names": {},
    }
    bundle = {
        "bundle_version": 1,
        "anchor_date": "SYNTHETIC-DEMO",
        "stability_b": DEMO_STABILITY_B,
        "disclosure": {"min_cell_n": report.MIN_CELL_N},
        "variants": {"v1": variant_block, "v4": variant_block},
        "robustness": {"v1": robustness, "v4": robustness},
        "null_control": {
            "summaries": null_summaries,
            "scan_tables": null_scan_tables,
            "methodological_conclusion":
                "demo rendering of the fixed rule: the gap statistic and "
                "bootstrap stability are the structure arbiters; "
                "silhouette/BIC are descriptive.",
        },
        "supervised": {
            "primary_anchor": "SYNTHETIC-DEMO",
            "sensitivity_anchor": "SYNTHETIC-DEMO",
            "primary": supervised_results,
            "sensitivity": supervised_results,
        },
        "era_argument": "(synthetic demo - no era argument is made)",
        "predictions": [
            {"id": 1, "summary": "synthetic demo prediction",
             "verdict": "confirmed",
             "deciding_statistic": "planted-truth ARI %.3f"
                                   % ari_summary["mean_ari"]},
        ],
        "predictions_registration": {"date": "SYNTHETIC-DEMO",
                                     "sha256": "0" * 64},
        "limitations": ["synthetic demo output - every number on this page "
                        "comes from generated data."],
        "headline": {
            "hero_validation_text":
                "SYNTHETIC demo: bootstrap mean ARI %.2f at the chosen k"
                % ari_summary["mean_ari"],
        },
    }
    return bundle


if __name__ == "__main__":
    main()
