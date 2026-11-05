"""Synthetic-only v2 of the findings (DECISIONS.md, 2026-11-04 — Findings
regenerated on synthetic data only).

What is guarded here:
- the committed bundle is labelled synthetic, its predictions file is pinned
  and its verdicts are what judge_predictions says;
- report.py refuses to render a bundle without the synthetic label into the
  repo's reports/ directory (the lab script used to write there);
- each prediction's verdict follows its written rule: flipping the deciding
  number flips the verdict (a rule that cannot flip proves nothing);
- the whole build runs end to end at smoke size and its output re-checks
  clean (the frozen-size build is ~10 minutes and never runs in CI).
"""

import copy
import json
import os
import shutil

import pytest

from cardsegments import report, synth_findings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def committed_bundle():
    return synth_findings.load_bundle(REPO_ROOT)


def verdict_of(bundle, prediction_id):
    rows = synth_findings.judge_predictions(bundle)
    for row in rows:
        if row["id"] == prediction_id:
            return row["verdict"]
    raise KeyError(prediction_id)


def test_committed_bundle_is_synthetic_and_current():
    assert synth_findings.bundle_problems(REPO_ROOT) == []


def test_committed_bundle_carries_no_real_run_sections():
    bundle = committed_bundle()
    assert bundle["provenance"]["label"] == "synthetic"
    assert bundle["anchor_date"] == "SYNTHETIC"
    assert "nopop_extrapolation" not in bundle
    assert bundle["supervised"]["primary_title"].startswith("SYNTHETIC")
    assert bundle["supervised"]["sensitivity_title"].startswith("SYNTHETIC")
    assert bundle["headline"]["hero_validation_text"].startswith("SYNTHETIC")
    with open(os.path.join(REPO_ROOT, synth_findings.FINDINGS_REL)) as handle:
        first_line = handle.readline()
    assert "SYNTHETIC" in first_line


def test_no_list_as_long_as_the_data_rows():
    bundle = committed_bundle()
    row_counts = {bundle["planted_truth"]["n"],
                  bundle["supervised"]["primary"]["n_rows"],
                  bundle["supervised"]["sensitivity"]["n_rows"]}
    too_long = []

    def walk(value, path):
        if isinstance(value, dict):
            keys = list(value.keys())
            for key in keys:
                walk(value[key], path + "/" + str(key))
        elif isinstance(value, list):
            if len(value) in row_counts:
                too_long.append(path)
            numbered = list(enumerate(value))
            for position, item in numbered:
                walk(item, path + "[%d]" % position)

    walk(bundle, "")
    assert too_long == []


def test_report_refuses_unlabelled_bundle_in_repo_reports():
    bundle = committed_bundle()
    reports_dir = report.repo_reports_dir()
    report.check_publishable(bundle, reports_dir)  # synthetic: allowed
    unlabelled = copy.deepcopy(bundle)
    del unlabelled["provenance"]
    with pytest.raises(ValueError):
        report.check_publishable(unlabelled, reports_dir)
    relabelled = copy.deepcopy(bundle)
    relabelled["provenance"]["label"] = "private"
    with pytest.raises(ValueError):
        report.check_publishable(relabelled, reports_dir)


def test_render_into_refuses_unlabelled_bundle_anywhere(tmp_path):
    bundle = committed_bundle()
    unlabelled = copy.deepcopy(bundle)
    unlabelled["provenance"] = {}
    with pytest.raises(ValueError):
        synth_findings.render_into(unlabelled, str(tmp_path))


def test_findings_text_is_labelled_by_provenance():
    bundle = committed_bundle()
    synthetic_text = report.findings_text(bundle)
    assert synthetic_text.startswith("# Findings (SYNTHETIC v2)")
    assert "SYNTHETIC DATA ONLY." in synthetic_text
    unlabelled = copy.deepcopy(bundle)
    del unlabelled["provenance"]
    plain_text = report.findings_text(unlabelled)
    assert "SYNTHETIC DATA ONLY." not in plain_text


def test_why_boxes_have_three_parts_and_reach_the_report():
    bundle = committed_bundle()
    text = report.findings_text(bundle)
    keys = list(synth_findings.WHY_THIS_METRIC.keys())
    for key in keys:
        box = synth_findings.WHY_THIS_METRIC[key]
        assert sorted(box.keys()) == ["kind", "lies", "question"]
        assert report.why_md(box) in text


# Each case: (prediction id, function that pushes its deciding number to the
# other side of the rule). Applied to the committed bundle, the verdict must
# change; that is the control that the rule reads the number it names.
def flip_gap_k(bundle):
    selection = bundle["variants"]["v1"]["selection"]["kmeans"]
    planted_k = bundle["planted_truth"]["k"]
    if selection["gap_k"] == planted_k:
        selection["gap_k"] = planted_k + 1
    else:
        selection["gap_k"] = planted_k


def flip_planted_ari(bundle):
    planted = bundle["planted_truth"]
    if planted["v1_ari"] >= synth_findings.PLANTED_ARI_MIN:
        planted["v1_ari"] = synth_findings.PLANTED_ARI_MIN - 0.01
    else:
        planted["v1_ari"] = synth_findings.PLANTED_ARI_MIN


def flip_stability(bundle):
    summary = bundle["variants"]["v1"]["stability"]
    if summary["mean_ari"] >= synth_findings.STABILITY_ARI_MIN:
        summary["mean_ari"] = synth_findings.STABILITY_ARI_MIN - 0.01
    else:
        summary["mean_ari"] = synth_findings.STABILITY_ARI_MIN


def flip_null_gap(bundle):
    summary = bundle["null_control"]["summaries"]["shuffle_null"]
    if summary["gap_one_se_k"] == 1:
        summary["gap_one_se_k"] = 2
        return
    summary["gap_one_se_k"] = 1
    bundle["null_control"]["summaries"]["mvn_null"]["gap_one_se_k"] = 1


def flip_weak_gap(bundle):
    row = synth_findings.find_sensitivity(bundle, "v2_weak_separation")
    if row["gap_k"] == 1:
        row["gap_k"] = 3
    else:
        row["gap_k"] = 1


def flip_era(bundle):
    lift_rows = bundle["variants"]["v4"]["era"]["lift"]
    vintage_enriched = False
    for row in lift_rows:
        if row["era"] == "vintage" and row.get("ci_low") is not None and row["ci_low"] > 1:
            vintage_enriched = True
    for row in lift_rows:
        if row["era"] != "vintage":
            continue
        if vintage_enriched:
            if row.get("ci_low") is not None:
                row["ci_low"] = 0.5
        else:
            row["ci_low"] = 1.5
            row["ci_high"] = 2.0
            row["lift"] = 1.7
    if not vintage_enriched:
        for row in lift_rows:
            if row["era"] == "modern":
                row["ci_low"] = 1.5
                row["ci_high"] = 2.0
                row["lift"] = 1.7


def flip_rho(bundle):
    block = bundle["gem_multiplier"]
    if abs(block["rho_overall"]) < synth_findings.NUISANCE_RHO_MAX:
        block["rho_overall"] = -0.5
    else:
        block["rho_overall"] = 0.0


def flip_null_gate(bundle):
    nested = bundle["supervised"]["primary"]["nested_delta_auc"]
    nested["ships"] = not nested["ships"]


def flip_planted_gate(bundle):
    nested = bundle["supervised"]["sensitivity"]["nested_delta_auc"]
    nested["ships"] = not nested["ships"]


def flip_forward_r2(bundle):
    fwd = bundle["supervised"]["primary"]["forward_regression"]
    if fwd.get("oof_r2") is not None and fwd["oof_r2"] <= 0:
        fwd["oof_r2"] = 0.2
    else:
        fwd["oof_r2"] = -0.1
        bundle["supervised"]["sensitivity"]["forward_regression"]["oof_r2"] = -0.1


FLIPS = [(1, flip_gap_k), (2, flip_planted_ari), (3, flip_stability), (4, flip_null_gap),
         (5, flip_weak_gap), (6, flip_era), (7, flip_rho), (8, flip_null_gate),
         (9, flip_planted_gate), (10, flip_forward_r2)]


@pytest.mark.parametrize("prediction_id,flip", FLIPS)
def test_each_verdict_flips_with_its_deciding_number(prediction_id, flip):
    bundle = committed_bundle()
    before = verdict_of(bundle, prediction_id)
    tampered = copy.deepcopy(bundle)
    flip(tampered)
    after = verdict_of(tampered, prediction_id)
    assert before != after


def test_verdict_words_are_the_three_allowed():
    rows = committed_bundle()["predictions"]
    assert len(rows) == 10
    for row in rows:
        assert row["verdict"] in ("confirmed", "refuted", "not decidable")


def copy_generated(target):
    rel_paths = ["README.md", "DECISIONS.md", "docs/METHOD.md", synth_findings.BUNDLE_REL,
                 synth_findings.FINDINGS_REL, synth_findings.PREDICTIONS_REL]
    for name in synth_findings.FIGURE_NAMES:
        rel_paths.append("reports/figures/" + name)
    for rel_path in rel_paths:
        destination = os.path.join(target, rel_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, rel_path), destination)


def test_changed_predictions_file_is_caught(tmp_path):
    copy_generated(str(tmp_path))
    assert synth_findings.bundle_problems(str(tmp_path)) == []
    path = os.path.join(str(tmp_path), synth_findings.PREDICTIONS_REL)
    with open(path, "a") as handle:
        handle.write("\n11. A prediction added after the run.\n")
    problems = synth_findings.bundle_problems(str(tmp_path))
    assert "predictions file changed after registration" in problems


def test_edited_verdict_is_caught(tmp_path):
    copy_generated(str(tmp_path))
    bundle_path = os.path.join(str(tmp_path), synth_findings.BUNDLE_REL)
    with open(bundle_path) as handle:
        bundle = json.load(handle)
    first = bundle["predictions"][0]
    if first["verdict"] == "confirmed":
        first["verdict"] = "refuted"
    else:
        first["verdict"] = "confirmed"
    synth_findings.write_bundle(bundle, bundle_path)
    problems = synth_findings.bundle_problems(str(tmp_path))
    assert "prediction verdicts differ from judge_predictions" in problems


def test_smoke_build_runs_end_to_end(tmp_path):
    root = str(tmp_path)
    copy_generated(root)
    bundle = synth_findings.build(root=root, settings=synth_findings.SMOKE_SETTINGS,
                                  verbose=False)
    assert report.is_synthetic(bundle)
    assert synth_findings.stale_files(root) == []
    problems = synth_findings.bundle_problems(root)
    assert problems == []
    assert len(bundle["predictions"]) == 10
