"""Findings report + figures, regenerated ENTIRELY from an aggregates
bundle. The committed bundle (reports/results_bundle.json) is SYNTHETIC
(written by `python -m cardsegments.synth_findings build`); a private
licensed-data bundle renders through the same code but never into the
repo's reports/ directory (render_findings refuses, see
check_publishable). Disclosure rules are enforced here, in code:
  - any per-group cell with n < MIN_CELL_N is suppressed;
  - medians/IQR only (the bundle never carries min/max of real data);
  - era shares as whole percents;
  - 2-3 significant figures.
"""

import copy
import json
import os

from cardsegments import viz
from cardsegments.supervised import MODEL_LABELS

MIN_CELL_N = 50

NARRATIVE_RULE = (
    "every characterizing sentence cites a profile number whose CI excludes "
    "the global value; only segments surviving ALL robustness cells get "
    "names; fragile structure goes to a k-sensitivity appendix"
)

WEAK_STRUCTURE_LEAD = (
    "segments are a useful quantization of a continuum, not well-separated "
    "clusters"
)


SYNTHETIC_LABEL = "synthetic"


def is_synthetic(bundle):
    """True when the bundle says it was computed on generated data."""
    provenance = bundle.get("provenance") or {}
    return provenance.get("label") == SYNTHETIC_LABEL


def repo_reports_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "reports")


def check_publishable(bundle, out_dir):
    """Refuse to write a non-synthetic bundle's report into the repo tree.

    Rule (DECISIONS.md, 2026-11-04 — Findings regenerated on synthetic
    data only): the tracked reports/ directory holds synthetic output only
    until the feed terms clear. Failure mode: a private bundle rendered
    into any OTHER directory inside the repo is not caught here; the
    deny-list check before a push is the second layer.
    """
    target = os.path.realpath(out_dir)
    tracked = os.path.realpath(repo_reports_dir())
    if target == tracked and not is_synthetic(bundle):
        raise ValueError("refusing to render a bundle without provenance "
                         "label 'synthetic' into the repo's reports/ "
                         "directory (see DECISIONS.md, 2026-11-04)")


def why_md(box):
    """The 3-line "why this metric" box (question / kind / what would lie)."""
    lines = ["> **Why this metric.** " + box["question"],
             "> *Kind of number:* " + box["kind"],
             "> *What would make it lie:* " + box["lies"]]
    return "\n".join(lines)


def why_lines(bundle, key):
    """[box markdown, ""] when the bundle carries that box, else []."""
    boxes = bundle.get("why_this_metric") or {}
    box = boxes.get(key)
    if box is None:
        return []
    return [why_md(box), ""]


def clean_nan(value):
    """Recursively turn float NaN into None so exported JSON stays strict."""
    if isinstance(value, dict):
        cleaned = {}
        keys = list(value.keys())
        for key in keys:
            cleaned[key] = clean_nan(value[key])
        return cleaned
    if isinstance(value, (list, tuple)):
        cleaned_list = []
        for item in value:
            cleaned_list.append(clean_nan(item))
        return cleaned_list
    if isinstance(value, float) and value != value:
        return None
    return value


def sig(value, digits=3):
    """2-3 significant figures for prose/tables; None stays visible as such."""
    if value is None:
        return "suppressed"
    if value != value:  # NaN
        return "n/a"
    if value == 0:
        return "0"
    formatted = ("%." + str(digits) + "g") % value
    return formatted


def whole_pct(value):
    if value is None:
        return "suppressed"
    return "%d%%" % round(value)


def load_bundle(path):
    with open(path) as handle:
        return json.load(handle)


def suppress_cell(cell, min_n=MIN_CELL_N):
    """Null out the statistics of a {median, ci_low, ci_high, n} cell when
    n < min_n; the n itself stays visible."""
    if cell is None:
        return None
    result = dict(cell)
    n = result.get("n", 0)
    if n is not None and n < min_n:
        result["median"] = None
        result["ci_low"] = None
        result["ci_high"] = None
        result["suppressed"] = True
    return result


def cluster_n_lookup(variant):
    """Cluster id (as str) -> cluster n, read off the variant's profile."""
    lookup = {}
    if not variant:
        return lookup
    profile = variant.get("profile")
    if not profile:
        return lookup
    for cluster in profile["clusters"]:
        lookup[str(cluster["cluster"])] = cluster["n"]
    return lookup


def apply_disclosure(bundle, min_n=MIN_CELL_N):
    """Deep-copied bundle with every small cell suppressed. Called both when
    the bundle is exported and again at render time (belt and braces)."""
    cleaned = copy.deepcopy(bundle)
    variant_names = list(cleaned.get("variants", {}).keys())
    for variant_name in variant_names:
        variant = cleaned["variants"][variant_name]
        profile = variant.get("profile")
        if profile:
            for cluster in profile["clusters"]:
                cluster_small = cluster["n"] < min_n
                feature_names = list(cluster["features"].keys())
                for feature_name in feature_names:
                    cell = suppress_cell(cluster["features"][feature_name], min_n)
                    if cluster_small and cell is not None:
                        cell["median"] = None
                        cell["ci_low"] = None
                        cell["ci_high"] = None
                        cell["suppressed"] = True
                    cluster["features"][feature_name] = cell
        era = variant.get("era")
        if era:
            kept_rows = []
            for row in era.get("cluster_shares_pct", []):
                if row["n"] < min_n:
                    kept_rows.append({"cluster": row["cluster"], "n": row["n"],
                                      "suppressed": True})
                else:
                    kept_rows.append(row)
            era["cluster_shares_pct"] = kept_rows
            # era lifts: every row carries its cluster n; small clusters
            # get their lift and CI nulled, with the n left visible
            n_lookup = cluster_n_lookup(variant)
            lift_rows = era.get("lift", [])
            for row in lift_rows:
                row_n = row.get("n")
                if row_n is None:
                    row_n = n_lookup.get(str(row["cluster"]))
                row["n"] = row_n
                if row_n is not None and row_n < min_n:
                    row["lift"] = None
                    row["ci_low"] = None
                    row["ci_high"] = None
                    row["suppressed"] = True
        trends = variant.get("trends")
        if trends:
            window_names = list(trends.keys())
            for window_name in window_names:
                if not isinstance(trends[window_name], list):
                    continue
                for row in trends[window_name]:
                    if row.get("n_eligible", 0) < min_n:
                        row["median_change"] = None
                        row["ci_low"] = None
                        row["ci_high"] = None
                        row["suppressed"] = True
    robustness = cleaned.get("robustness")
    if robustness:
        robustness_names = list(robustness.keys())
        for variant_name in robustness_names:
            variant = cleaned.get("variants", {}).get(variant_name)
            n_lookup = cluster_n_lookup(variant)
            cells = robustness[variant_name].get("cells", [])
            for cell in cells:
                recalls = cell["recall_per_reference_cluster"]
                reference_ids = list(recalls.keys())
                for reference_id in reference_ids:
                    entry = recalls[reference_id]
                    # older exports carried a bare recall float; normalize
                    # so every recall cell carries its reference-cluster n
                    if not isinstance(entry, dict):
                        entry = {"recall": entry,
                                 "n": n_lookup.get(str(reference_id))}
                    elif entry.get("n") is None:
                        entry["n"] = n_lookup.get(str(reference_id))
                    n_reference = entry.get("n")
                    if n_reference is not None and n_reference < min_n:
                        entry["recall"] = None
                        entry["suppressed"] = True
                    recalls[reference_id] = entry
    nopop = cleaned.get("nopop_extrapolation")
    if nopop:
        for cluster in nopop.get("clusters", []):
            cluster_small = cluster["n"] < min_n
            feature_names = list(cluster.get("features", {}).keys())
            for feature_name in feature_names:
                cell = suppress_cell(cluster["features"][feature_name], min_n)
                if cluster_small and cell is not None:
                    cell["median"] = None
                    cell["ci_low"] = None
                    cell["ci_high"] = None
                    cell["suppressed"] = True
                cluster["features"][feature_name] = cell
    return cleaned


def cell_text(cell):
    if cell is None or cell.get("median") is None:
        n = 0
        if cell is not None:
            n = cell.get("n", 0)
        return "suppressed (n=%d)" % n
    return "%s [%s, %s] (n=%d)" % (sig(cell["median"]), sig(cell["ci_low"]),
                                   sig(cell["ci_high"]), cell["n"])


def profile_table_md(profile, feature_cols):
    lines = []
    header_cells = ["cluster", "n"]
    for feature in feature_cols:
        header_cells.append(viz.short_name(feature))
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "---|" * len(header_cells))
    for cluster in profile["clusters"]:
        row_cells = [str(cluster["cluster"]), str(cluster["n"])]
        for feature in feature_cols:
            row_cells.append(cell_text(cluster["features"].get(feature)))
        lines.append("| " + " | ".join(row_cells) + " |")
    global_cells = ["global", str(profile["clusters_total_n"])]
    for feature in feature_cols:
        cell = profile["global"].get(feature)
        if cell is None or cell.get("median") is None:
            global_cells.append("n/a")
        else:
            global_cells.append("%s (IQR %s)" % (sig(cell["median"]),
                                                 sig(cell.get("iqr"))))
    lines.append("| " + " | ".join(global_cells) + " |")
    return "\n".join(lines)


def era_lift_md(lift_rows):
    lines = ["| cluster | n | era | lift | 95% CI | era-enriched |",
             "|---|---|---|---|---|---|"]
    for row in lift_rows:
        row_n = row.get("n")
        if row_n is None:
            n_text = "n/a"
        else:
            n_text = str(row_n)
        if row.get("suppressed") or row.get("lift") is None:
            lines.append("| %d | %s | %s | suppressed | | |"
                         % (row["cluster"], n_text, row["era"]))
            continue
        if row["ci_low"] is None:
            ci_text = "n/a"
            enriched = ""
        else:
            ci_text = "[%s, %s]" % (sig(row["ci_low"]), sig(row["ci_high"]))
            if row["ci_low"] > 1.0:
                enriched = "YES"
            else:
                enriched = ""
        lines.append("| %d | %s | %s | %s | %s | %s |"
                     % (row["cluster"], n_text, row["era"], sig(row["lift"]),
                        ci_text, enriched))
    return "\n".join(lines)


def trends_md(trend_rows):
    lines = ["| cluster | n | eligible | coverage | median log change | 95% CI |",
             "|---|---|---|---|---|---|"]
    for row in trend_rows:
        if row.get("median_change") is None:
            stat = "suppressed"
            ci_text = ""
        else:
            stat = sig(row["median_change"])
            ci_text = "[%s, %s]" % (sig(row["ci_low"]), sig(row["ci_high"]))
        if row.get("coverage") is None:
            coverage_text = "n/a"
        else:
            coverage_text = whole_pct(row["coverage"] * 100.0)
        lines.append("| %d | %d | %d | %s | %s | %s |"
                     % (row["cluster"], row["n_cluster"], row["n_eligible"],
                        coverage_text, stat, ci_text))
    return "\n".join(lines)


def ordered_model_names(pooled_auc):
    """Models in MODEL_LABELS order (simple to complex), then any others.

    Why: the bundle JSON is written with sorted keys, which would put
    'majority' after 'gbc'; the table reads baseline-first."""
    names = []
    known = list(MODEL_LABELS.keys())
    for name in known:
        if name in pooled_auc:
            names.append(name)
    present = sorted(pooled_auc.keys())
    for name in present:
        if name not in names:
            names.append(name)
    return names


SYNTHETIC_FORWARD_NOTE = (
    "The generator draws this target as pure noise on a random ~20% of rows, so "
    "out-of-fold R^2 <= 0 is the expected null control, not a claim.")
REAL_FORWARD_NOTE = (
    "The eligibility screen (enough sales on both sides of the window) selects "
    "LIQUID cards, so this subset is not the cohort; and with log raw price at T0 "
    "in X, part of any fit is regression to the mean. Reported for completeness, "
    "not as a claim.")


def supervised_md(block, title, synthetic=False):
    lines = ["### %s" % title, ""]
    lines.append("Cohort n=%d (%d groups), base rate %s. Cluster features "
                 "complete for %d rows; all remaining rows carry the explicit "
                 "`no_cluster` category, so the nested comparison runs on the "
                 "full cohort."
                 % (block["n_rows"], block["n_groups"], sig(block["base_rate"]),
                    block["n_cluster_feature_complete"]))
    threshold = block.get("y_liquid_threshold")
    window = block.get("outcome_window")
    if threshold is not None and window:
        lines.append("")
        lines.append("Outcome: y_liquid = at least %d distinct sold events "
                     "in the (%s, %s] window (the threshold is prorated to "
                     "the window length across anchors)."
                     % (threshold, window[0], window[1]))
    outcome_note = block.get("outcome_note")
    if outcome_note:
        lines.append("")
        lines.append(outcome_note)
    lines.append("")
    lines.append("| model | pooled OOF ROC-AUC | 95% group-bootstrap CI |")
    lines.append("|---|---|---|")
    model_names = ordered_model_names(block["pooled_auc"])
    for name in model_names:
        row = block["pooled_auc"][name]
        label = MODEL_LABELS.get(name, name)
        lines.append("| %s | %s | [%s, %s] |"
                     % (label, sig(row["auc"]), sig(row["ci_low"]),
                        sig(row["ci_high"])))
    nested = block["nested_delta_auc"]
    lines.append("")
    lines.append("**Nested gate** (era+velocity+cluster over era+velocity): "
                 "mean dAUC over seeds = %s; paired group-bootstrap 95%% CI "
                 "[%s, %s]; fold-level spread mean %s, min %s, 5th pct %s "
                 "(%d fold-seed cells)."
                 % (sig(nested["mean_over_seeds"]),
                    sig(nested["group_bootstrap"]["ci_low"]),
                    sig(nested["group_bootstrap"]["ci_high"]),
                    sig(nested["fold_seed_spread"]["mean"]),
                    sig(nested["fold_seed_spread"]["min"]),
                    sig(nested["fold_seed_spread"]["p5"]),
                    nested["fold_seed_spread"]["n_cells"]))
    if nested["ships"]:
        verdict = ("The ship rule (%s) PASSES: cluster membership adds "
                   "predictive signal beyond era and trailing velocity."
                   % nested["ship_rule"])
    else:
        verdict = ("The ship rule (%s) FAILS: this is a NULL result - "
                   "in-fold cluster membership does not add predictive signal "
                   "beyond era and trailing velocity, and the segmentation's "
                   "value here is descriptive, not predictive."
                   % nested["ship_rule"])
    lines.append("")
    lines.append("**Verdict:** " + verdict)
    confusion = block["confusion_at_t_star"]
    lines.append("")
    lines.append("**Decisions at the cost threshold** t* = %s (C_FP = m+h, "
                 "C_FN = m, illustrative defaults m = h = 0.10 → t* = 2/3). "
                 "In buyer's terms: a false positive is capital parked in a "
                 "card that sits; a false negative is passing on a card that "
                 "would have sold. The threshold is high because parking "
                 "capital costs more than missing one flip. The defaults are "
                 "illustrative; the threshold-cost curve is drawn for three "
                 "(m, h) settings so no conclusion hinges on them."
                 % sig(confusion["threshold"]))
    lines.append("")
    lines.append("| | predicted liquid | predicted sits |")
    lines.append("|---|---|---|")
    lines.append("| actually liquid | TP=%d | FN=%d |"
                 % (confusion["tp"], confusion["fn"]))
    lines.append("| actually sits | FP=%d | TN=%d |"
                 % (confusion["fp"], confusion["tn"]))
    lines.append("")
    lines.append("precision %s, recall %s, specificity %s, F1 %s (n=%d). "
                 "PR-AUC oriented to the minority class (%s): %s."
                 % (sig(confusion["precision"]), sig(confusion["recall"]),
                    sig(confusion["specificity"]), sig(confusion["f1"]),
                    confusion["n"], block["minority_class"],
                    sig(block["pr_auc_minority"])))
    fwd = block["forward_regression"]
    lines.append("")
    if fwd.get("oof_r2") is None:
        fwd_text = "not fit (n_eligible=%d)" % fwd["n_eligible"]
    else:
        fwd_text = "out-of-fold R^2 = %s on n=%d eligible cards" \
            % (sig(fwd["oof_r2"]), fwd["n_eligible"])
    if synthetic:
        forward_note = SYNTHETIC_FORWARD_NOTE
    else:
        forward_note = REAL_FORWARD_NOTE
    lines.append("**Forward-price regression (secondary):** %s. %s"
                 % (fwd_text, forward_note))
    coefficients = block.get("coefficients")
    if coefficients:
        lines.append("")
        lines.append("Full logistic (L2, C=1; key `primary_logistic`) "
                     "coefficients (standardized scale; group-bootstrap 95% CI):")
        lines.append("")
        lines.append("| feature | coef | 95% CI |")
        lines.append("|---|---|---|")
        for row in coefficients:
            lines.append("| %s | %s | [%s, %s] |"
                         % (viz.short_name(row["feature"]), sig(row["coef"]),
                            sig(row["ci_low"]), sig(row["ci_high"])))
    return "\n".join(lines)


def chosen_solution_bootstrap(selection):
    """("kmeans k=4" or "gmm full k=9", that method's bootstrap summary)."""
    solution = selection["solution"]
    if solution["method"] == "gmm":
        text = "gmm %s k=%d" % (solution.get("cov_type", ""), solution["k"])
        return text, selection["gmm"]["bootstrap"]
    text = "%s k=%d" % (solution["method"], solution["k"])
    return text, selection["kmeans"]["bootstrap"]


def predictions_md(prediction_rows, registration):
    lines = []
    lines.append("Predictions were registered on %s (sha256 %s) before any "
                 "pilot inspection. Verdicts below are decided by the named "
                 "statistic, not by narrative."
                 % (registration["date"], registration["sha256"][:16] + "..."))
    lines.append("")
    lines.append("| # | prediction (summary) | verdict | deciding statistic |")
    lines.append("|---|---|---|---|")
    for row in prediction_rows:
        lines.append("| %d | %s | **%s** | %s |"
                     % (row["id"], row["summary"], row["verdict"],
                        row["deciding_statistic"]))
    return "\n".join(lines)


def render_figures(bundle, figures_dir):
    """The four report figures from a disclosure-applied bundle."""
    v4 = bundle["variants"]["v4"]
    headline = bundle["headline"]
    supervised_primary = bundle["supervised"]["primary"]
    if is_synthetic(bundle):
        data_word = "synthetic"
        prefix = "SYNTHETIC: "
    else:
        data_word = "real"
        prefix = ""
    written = []
    hero_path = os.path.join(figures_dir, "hero.png")
    viz.save_hero(v4["profile"], v4["profile"]["feature_cols"],
                  v4["era"]["cluster_shares_pct"],
                  headline["hero_validation_text"], hero_path,
                  cluster_names=v4.get("cluster_names"))
    written.append(hero_path)
    null_path = os.path.join(figures_dir, "null_control.png")
    viz.save_null_control(bundle["null_control"]["scan_tables"], null_path,
                          data_word=data_word)
    written.append(null_path)
    calibration_path = os.path.join(figures_dir, "calibration.png")
    viz.save_calibration(supervised_primary["calibration_bins"],
                         supervised_primary["base_rate"], calibration_path,
                         title_prefix=prefix)
    written.append(calibration_path)
    cost_path = os.path.join(figures_dir, "threshold_cost.png")
    viz.save_threshold_cost(supervised_primary["threshold_cost_curves"],
                            cost_path, title_prefix=prefix)
    written.append(cost_path)
    return written


def render_findings(bundle, out_dir, figures_dirname="figures"):
    """Write findings.md + all figures from the bundle. Returns file list.

    Refuses a non-synthetic bundle when out_dir is the repo's reports/
    (check_publishable)."""
    check_publishable(bundle, out_dir)
    figures_dir = os.path.join(out_dir, figures_dirname)
    os.makedirs(figures_dir, exist_ok=True)
    cleaned = apply_disclosure(bundle)
    written = render_figures(cleaned, figures_dir)
    text = findings_text(bundle, figures_dirname)
    findings_path = os.path.join(out_dir, "findings.md")
    with open(findings_path, "w") as handle:
        handle.write(text)
    written.append(findings_path)
    return written


def findings_text(bundle, figures_dirname="figures"):
    """findings.md as one string (no files written); disclosure re-applied."""
    bundle = apply_disclosure(bundle)
    v1 = bundle["variants"]["v1"]
    v4 = bundle["variants"]["v4"]
    synthetic = is_synthetic(bundle)
    if synthetic:
        data_word = "synthetic"
    else:
        data_word = "real"

    lines = []
    if synthetic:
        provenance = bundle["provenance"]
        lines.append("# Findings (SYNTHETIC v2) - graded-card market "
                     "segmentation method")
        lines.append("")
        lines.append("*This file and every figure under figures/ are "
                     "regenerated by `python -m cardsegments.report` from "
                     "`reports/results_bundle.json`, which `python -m "
                     "cardsegments.synth_findings build` computes. "
                     "Suppression rule: any per-group cell with n < %d is "
                     "suppressed in code.*" % MIN_CELL_N)
        lines.append("")
        lines.append("**SYNTHETIC DATA ONLY.** " + provenance["note"])
        lines.append("")
    else:
        lines.append("# Findings - graded-card market segmentation")
        lines.append("")
        lines.append("*This file and every figure under figures/ are "
                     "regenerated by `python -m cardsegments.report` from "
                     "`reports/results_bundle.json`. Suppression rule: any "
                     "per-group cell with n < %d is suppressed in code.*"
                     % MIN_CELL_N)
        lines.append("")
        lines.append("Snapshot as of %s; the underlying per-card dataset is "
                     "licensed and not redistributable, so this report is not "
                     "independently reproducible - the repo reproduces the "
                     "full method on synthetic data." % bundle["anchor_date"])
        lines.append("")
    lines.append("![hero](%s/hero.png)" % figures_dirname)
    lines.append("")

    # --- structure verdict (the weak-structure branch is decided by the
    # pre-registered rule: gap one-SE k=1 on the primary matrix)
    lines.append("## Structure verdict (V1, primary feature set)")
    lines.append("")
    v1_selection = v1["selection"]
    v1_weak = bool(v1_selection["kmeans"]["gap_k"] == 1)
    shuffle_sil = bundle["null_control"]["summaries"]["shuffle_null"][
        "max_silhouette"]
    if v1_weak:
        if v1_selection["max_silhouette"] < shuffle_sil:
            null_comparison = "BELOW the column-shuffled null's %s" \
                % sig(shuffle_sil)
        else:
            null_comparison = "vs the column-shuffled null's %s" \
                % sig(shuffle_sil)
        lines.append("**%s.** The gap statistic (one-SE) chooses k=1 on the "
                     "primary matrix (n=%d); the best k-means silhouette on "
                     "%s V1 is %s, %s. The GMM criteria prefer cov=%s "
                     "k=%d (bootstrap mean ARI %s) - read as a descriptive "
                     "quantization of a continuum, not as well-separated "
                     "clusters."
                     % (WEAK_STRUCTURE_LEAD, v1_selection["n"], data_word,
                        sig(v1_selection["max_silhouette"]), null_comparison,
                        v1_selection["gmm"]["chosen"]["cov"],
                        v1_selection["gmm"]["chosen"]["k"],
                        sig(v1_selection["gmm"]["bootstrap"]["mean_ari"])))
    else:
        v1_solution = v1_selection["solution"]
        lines.append("The selection pipeline chose %s k=%d on the primary "
                     "matrix (n=%d): silhouette k=%d, gap one-SE k=%d, "
                     "GMM %s k=%d; max silhouette %s (column-shuffled null: "
                     "%s)."
                     % (v1_solution["method"], v1_solution["k"],
                        v1_selection["n"],
                        v1_selection["kmeans"]["silhouette_k"],
                        v1_selection["kmeans"]["gap_k"],
                        v1_selection["gmm"]["chosen"]["cov"],
                        v1_selection["gmm"]["chosen"]["k"],
                        sig(v1_selection["max_silhouette"]),
                        sig(shuffle_sil)))
    lines.append("")
    planted = bundle.get("planted_truth")
    if planted:
        lines.append("Planted truth: k=%d segments; adjusted Rand index of the "
                     "chosen V1 solution against the planted labels = %s "
                     "(n=%d)." % (planted["k"], sig(planted["v1_ari"]),
                                  planted["n"]))
        lines.append("")
    if synthetic:
        v4_lead = ("The price-residualized companion V4 (price kept as one "
                   "axis, the other axes residualized on it)")
    else:
        v4_lead = "The price-residualized companion V4 is where repeatable " \
                  "structure lives"
    v4_solution_text, v4_boot = chosen_solution_bootstrap(v4["selection"])
    lines.append("%s: %s with bootstrap mean ARI %s "
                 "(min %s over %d resamples)."
                 % (v4_lead, v4_solution_text,
                    sig(v4_boot["mean_ari"]), sig(v4_boot["min_ari"]),
                    bundle["stability_b"]))
    lines.append("")
    lines.extend(why_lines(bundle, "structure"))
    sensitivities = bundle.get("sensitivities")
    if sensitivities:
        lines.append("Sensitivity variants (selection only):")
        lines.append("")
        lines.append("| variant | n | p | silhouette k | gap one-SE k | "
                     "GMM pick | chosen solution | max silhouette |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in sensitivities:
            solution = row["solution"]
            if solution["method"] == "gmm":
                solution_text = "gmm %s k=%d" % (solution["cov_type"],
                                                 solution["k"])
            else:
                solution_text = "%s k=%d" % (solution["method"], solution["k"])
            lines.append("| %s | %d | %d | %d | %d | %s k=%d | %s | %s |"
                         % (row["name"], row["n"], row["p"],
                            row["silhouette_k"], row["gap_k"],
                            row["gmm_cov"], row["gmm_k"], solution_text,
                            sig(row["max_silhouette"])))
        lines.append("")
    lines.append("**Narrative rule (fixed before writing):** \"%s\""
                 % NARRATIVE_RULE)
    lines.append("")

    # --- segments
    lines.append("## Segments (V4, price-residualized companion)")
    lines.append("")
    lines.append("Profile medians below are on the ORIGINAL feature scale "
                 "(per-feature median, card-level bootstrap 95% CI, cell n). "
                 "Only clusters surviving all robustness cells carry names.")
    lines.append("")
    name_rows = []
    cluster_names = v4.get("cluster_names", {})
    survives = bundle["robustness"]["v4"]["survives"]
    profile_clusters = v4["profile"]["clusters"]
    for cluster in profile_clusters:
        cluster_id = cluster["cluster"]
        survived = survives.get(str(cluster_id), survives.get(cluster_id))
        name = cluster_names.get(str(cluster_id), cluster_names.get(cluster_id))
        if survived and name:
            name_rows.append("- cluster %s: **%s**" % (cluster_id, name))
        elif survived:
            name_rows.append("- cluster %s: (survives every robustness cell; "
                             "no naming rule fired)" % cluster_id)
        else:
            name_rows.append("- cluster %s: (unnamed - failed a robustness "
                             "cell)" % cluster_id)
    lines.extend(name_rows)
    lines.append("")
    lines.append(profile_table_md(v4["profile"], v4["profile"]["feature_cols"]))
    lines.append("")

    # --- era composition
    lines.append("## Era composition")
    lines.append("")
    lines.append("Era labels were NEVER features; composition below is "
                 "recovery, not input. Cramer's V (cluster x era) = %s on V4, "
                 "%s on V1. Lift = P(era | cluster) / P(era); a CI excluding "
                 "1 marks the cluster era-enriched."
                 % (sig(v4["era"]["cramers_v"]), sig(v1["era"]["cramers_v"])))
    lines.append("")
    lines.append("V4 era lifts (CI over %d Hungarian-matched stability "
                 "resamples):" % bundle["stability_b"])
    lines.append("")
    lines.append(era_lift_md(v4["era"]["lift"]))
    lines.append("")
    lines.append(bundle["era_argument"])
    lines.append("")

    # --- trends
    lines.append("## Trends per segment")
    lines.append("")
    lines.append(v4["trends"]["protocol"])
    lines.append("")
    if v4["trends"].get("d90") is not None:
        lines.append("90-day window:")
        lines.append("")
        lines.append(trends_md(v4["trends"]["d90"]))
        lines.append("")
    if v4["trends"].get("d365") is not None:
        lines.append("365-day window:")
        lines.append("")
        lines.append(trends_md(v4["trends"]["d365"]))
        lines.append("")

    # --- robustness
    lines.append("## Robustness and stability")
    lines.append("")
    solution_parts = []
    stability_names = ["v4", "v1"]
    for variant_name in stability_names:
        selection = bundle["variants"][variant_name]["selection"]
        solution_text, boot = chosen_solution_bootstrap(selection)
        solution_parts.append("%s %s mean %s / min %s"
                              % (variant_name.upper(), solution_text,
                                 sig(boot["mean_ari"]), sig(boot["min_ari"])))
    lines.append("Bootstrap ARI (B=%d, 80%% resamples, Hungarian-matched) of "
                 "each variant's chosen solution: %s. Robustness cells (k*-1, "
                 "k*, k*+1 per variant; survival = %s):"
                 % (bundle["stability_b"], "; ".join(solution_parts),
                    bundle["robustness"]["v4"]["survival_rule"]))
    lines.append("")
    robustness_names = ["v1", "v4"]
    for variant_name in robustness_names:
        block = bundle["robustness"][variant_name]
        lines.append("%s cells:" % variant_name.upper())
        for cell in block["cells"]:
            recall_parts = []
            recall_ids = list(cell["recall_per_reference_cluster"].keys())
            for reference_id in recall_ids:
                entry = cell["recall_per_reference_cluster"][reference_id]
                if isinstance(entry, dict):
                    recall = entry.get("recall")
                    n_reference = entry.get("n")
                else:
                    recall = entry
                    n_reference = None
                if n_reference is None:
                    n_text = "n/a"
                else:
                    n_text = str(n_reference)
                recall_parts.append("c%s=%s (n=%s)"
                                    % (reference_id, sig(recall, 2), n_text))
            lines.append("- k=%d: recall %s" % (cell["k"],
                                                ", ".join(recall_parts)))
        lines.append("")
    lines.append("Fragile structure (any cluster failing a cell) is excluded "
                 "from naming above; the k-sensitivity detail stays in this "
                 "appendix rather than the narrative.")
    lines.append("")

    # --- null control
    lines.append("## Null-data control")
    lines.append("")
    summaries = bundle["null_control"]["summaries"]
    summary_labels = list(summaries.keys())
    lines.append("| dataset | silhouette k | gap one-SE k | BIC pick | "
                 "ICL pick | max silhouette |")
    lines.append("|---|---|---|---|---|---|")
    for label in summary_labels:
        summary = summaries[label]
        lines.append("| %s | %d | %d | %s k=%d | %s k=%d | %s |"
                     % (label, summary["silhouette_k"], summary["gap_one_se_k"],
                        summary["bic_choice"]["cov"], summary["bic_choice"]["k"],
                        summary["icl_choice"]["cov"], summary["icl_choice"]["k"],
                        sig(summary["max_silhouette"])))
    lines.append("")
    lines.append("![null control](%s/null_control.png)" % figures_dirname)
    lines.append("")
    lines.append("**Read honestly:** " +
                 bundle["null_control"]["methodological_conclusion"])
    lines.append("")
    lines.extend(why_lines(bundle, "null_control"))

    # --- supervised
    if synthetic:
        lines.append("## Supervised validation (SYNTHETIC illustrative preset, "
                     "leak-audited)")
    else:
        lines.append("## Supervised validation (T0-anchored, leak-audited)")
    lines.append("")
    supervised_block = bundle["supervised"]
    primary_title = supervised_block.get(
        "primary_title", "Primary anchor T0 = %s"
        % supervised_block["primary_anchor"])
    sensitivity_title = supervised_block.get(
        "sensitivity_title", "Sensitivity anchor T0 = %s"
        % supervised_block["sensitivity_anchor"])
    lines.append(supervised_md(supervised_block["primary"], primary_title,
                               synthetic=synthetic))
    lines.append("")
    lines.append(supervised_md(supervised_block["sensitivity"],
                               sensitivity_title, synthetic=synthetic))
    lines.append("")
    lines.extend(why_lines(bundle, "nested_gate"))
    supervised_notes = bundle["supervised"].get("notes", [])
    if supervised_notes:
        lines.append("**Supervised-design notes:**")
        lines.append("")
        for note in supervised_notes:
            lines.append("- " + note)
        lines.append("")
    lines.append("![calibration](%s/calibration.png)" % figures_dirname)
    lines.append("![threshold cost](%s/threshold_cost.png)" % figures_dirname)
    lines.append("")

    # --- no-pop extrapolation
    nopop = bundle.get("nopop_extrapolation")
    if nopop:
        lines.append("## No-population extrapolation (EXTRAPOLATION - "
                     "assignment only)")
        lines.append("")
        lines.append(nopop["note"])
        lines.append("")
        lines.append("| cluster | n assigned |")
        lines.append("|---|---|")
        for cluster in nopop["clusters"]:
            lines.append("| %d | %d |" % (cluster["cluster"], cluster["n"]))
        lines.append("")

    # --- predictions
    lines.append("## Registered predictions, judged")
    lines.append("")
    lines.append(predictions_md(bundle["predictions"],
                                bundle["predictions_registration"]))
    lines.append("")

    # --- limitations
    lines.append("## Limitations")
    lines.append("")
    for item in bundle["limitations"]:
        lines.append("- " + item)
    lines.append("")
    return "\n".join(lines)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    reports_dir = os.path.join(repo_root, "reports")
    bundle_path = os.path.join(reports_dir, "results_bundle.json")
    bundle = load_bundle(bundle_path)
    # render_findings refuses a non-synthetic bundle here (DECISIONS.md,
    # 2026-11-04 — Findings regenerated on synthetic data only)
    written = render_findings(bundle, reports_dir)
    for path in written:
        print("wrote", path)


if __name__ == "__main__":
    main()
