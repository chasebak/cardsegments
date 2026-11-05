"""Model ladders: fold integrity, leakage controls, oracle bounds, the beats
rule, the headline KPI, and the pre-registration pin.

Two kinds of test sit here:
- LIVE tests run the ladder on a small threshold-mode synthetic frame
  (n = 1,500, 30 set groups, 2 CV seeds), a few seconds in CI;
- FROZEN tests read the committed results/ladder_frozen.json (the heavy run
  never runs in CI) and check its numbers obey the registered rules.
Every check has a control that must fail, so a check that cannot fail is
caught.
"""

import hashlib
import json
import math
import os
import re
import warnings

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

from cardsegments import buylist, ladder, ladder_predictions, ladder_report
from cardsegments.synth import generate_supervised

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
LIVE_SEEDS = (0, 1)
LIVE_ORACLE_TOLERANCE = 0.02


def live_frame(cluster_signal=0.9, seed=3):
    return generate_supervised(n_rows=1500, n_clusters=3, separation=2.5, seed=seed,
                               cluster_signal=cluster_signal, outcome_mode="threshold")


@pytest.fixture(scope="module")
def live_run():
    frame, manifest, truth = live_frame()
    raw = ladder.cross_validate_ladder(frame, manifest["feature_cols"],
                                       manifest["baseline_cols"], LIVE_SEEDS, truth=truth)
    return frame, manifest, truth, raw


@pytest.fixture(scope="module")
def frozen():
    with open(os.path.join(RESULTS, ladder.FROZEN_FILENAME)) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def trial():
    with open(os.path.join(RESULTS, ladder.TRIAL_FILENAME)) as handle:
        return json.load(handle)


# ---------------------------------------------------------------- folds

def test_every_rung_sees_identical_folds(live_run):
    _frame, _manifest, _truth, raw = live_run
    checks = ladder.fold_checks(raw)
    assert checks["identical_folds_all_rungs"] is True
    assert checks["every_row_scored_once_per_seed"] is True
    reference = raw["fold_seen_class"]["velocity"]
    for name in raw["fold_seen_reg"]:
        assert np.array_equal(raw["fold_seen_reg"][name], reference)


def test_fold_fingerprint_control_differs_across_seeds(live_run):
    frame, _manifest, _truth, _raw = live_run
    y = frame["y_liquid"].to_numpy()
    groups = frame["group_id"].to_numpy()
    first = ladder.fold_fingerprint(ladder.outer_folds(y, groups, 5, 0), len(y))
    second = ladder.fold_fingerprint(ladder.outer_folds(y, groups, 5, 1), len(y))
    assert not np.array_equal(first, second)


def test_outer_folds_are_group_disjoint(live_run):
    frame, _manifest, _truth, raw = live_run
    assert raw["shared_group_pairs"] == 0
    y = frame["y_liquid"].to_numpy()
    groups = frame["group_id"].to_numpy()
    for seed in LIVE_SEEDS:
        folds = ladder.outer_folds(y, groups, 5, seed)
        assert ladder.shared_groups_count(folds, groups) == 0


def test_group_disjoint_check_catches_plain_kfold(live_run):
    frame, _manifest, _truth, _raw = live_run
    groups = frame["group_id"].to_numpy()
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    split_iter = splitter.split(np.zeros((len(groups), 1)))
    folds = []
    for train_idx, test_idx in split_iter:
        folds.append((train_idx, test_idx))
    assert ladder.shared_groups_count(folds, groups) > 0


def test_no_rung_makes_an_internal_split():
    """No rung has inner folds: none tunes, none early-stops on an ungrouped split."""
    all_rungs = ladder.CLASSIFICATION_RUNGS + ladder.REGRESSION_RUNGS
    assert ladder.internal_split_violations(all_rungs) == []


def test_internal_split_check_catches_early_stopping_and_cv_estimators():
    def make_early():
        return GradientBoostingClassifier(n_iter_no_change=5)

    def make_cv():
        return LogisticRegressionCV()

    bad_rungs = [{"name": "early", "make": make_early}, {"name": "cv", "make": make_cv}]
    assert ladder.internal_split_violations(bad_rungs) == ["early", "cv"]


# ---------------------------------------------------------------- determinism

def tables_without_seconds(raw):
    class_rows = ladder.summarize_classification(raw, 50, 7)
    reg_rows = ladder.summarize_regression(raw, 0.01, 50, 7)
    for row in class_rows:
        row["fit_seconds"] = None
    for row in reg_rows:
        row["fit_seconds"] = None
    text = ladder_report.classification_table(class_rows) + "\n" \
        + ladder_report.regression_table(reg_rows)
    payload = json.dumps(ladder.clean_for_json({"c": class_rows, "r": reg_rows}),
                         sort_keys=True)
    return text, payload


def test_same_seed_gives_byte_identical_table():
    frame, manifest, truth = live_frame(seed=5)
    raw_a = ladder.cross_validate_ladder(frame, manifest["feature_cols"],
                                         manifest["baseline_cols"], (4,), truth=truth)
    raw_b = ladder.cross_validate_ladder(frame, manifest["feature_cols"],
                                         manifest["baseline_cols"], (4,), truth=truth)
    text_a, payload_a = tables_without_seconds(raw_a)
    text_b, payload_b = tables_without_seconds(raw_b)
    assert text_a.encode() == text_b.encode()
    assert payload_a.encode() == payload_b.encode()


# ---------------------------------------------------------------- beats rule

def test_beats_rule_follows_ci_and_bar():
    bar = 0.01
    assert ladder.beats_baseline(0.02, 0.005, 0.03, bar) is True
    assert ladder.beats_baseline(0.01, 0.0001, 0.02, bar) is True  # on the bar
    assert ladder.beats_baseline(0.02, -0.001, 0.04, bar) is False  # CI holds 0
    assert ladder.beats_baseline(0.005, 0.001, 0.009, bar) is False  # under the bar
    assert ladder.beats_baseline(-0.02, -0.03, -0.01, bar) is False  # worse, CI < 0
    assert ladder.beats_baseline(0.02, 0.0, 0.03, bar) is False  # CI touches 0
    assert ladder.beats_baseline(None, None, None, bar) is False


# ---------------------------------------------------------------- leakage

def leak_rung():
    return {"name": "leak_control", "label": "LEAK CONTROL (label as a feature)",
            "design": "features_plus_label", "make": ladder.make_l2_logistic,
            "constant": False}


def leak_regressor():
    return {"name": "leak_ols", "label": "LEAK CONTROL", "design": "features_plus_label",
            "make": ladder.make_ols, "constant": False, "count_model": False}


@pytest.fixture(scope="module")
def shuffled_with_leak():
    frame, manifest, _truth = live_frame(cluster_signal=0.9, seed=8)
    class_rungs = [ladder.rung_by_name(ladder.CLASSIFICATION_RUNGS, "majority"),
                   ladder.rung_by_name(ladder.CLASSIFICATION_RUNGS, "velocity"),
                   ladder.rung_by_name(ladder.CLASSIFICATION_RUNGS, "logistic_l2_c1"),
                   leak_rung()]
    reg_rungs = [ladder.rung_by_name(ladder.REGRESSION_RUNGS, "mean"),
                 ladder.rung_by_name(ladder.REGRESSION_RUNGS, "velocity_ols"),
                 ladder.rung_by_name(ladder.REGRESSION_RUNGS, "ols"),
                 leak_regressor()]
    raw = ladder.cross_validate_ladder(frame, manifest["feature_cols"],
                                       manifest["baseline_cols"], LIVE_SEEDS, shuffle=True,
                                       classification_rungs=class_rungs,
                                       regression_rungs=reg_rungs)
    return raw


def test_shuffled_labels_land_in_band_and_planted_leak_does_not(shuffled_with_leak):
    summary = ladder.shuffled_summary(shuffled_with_leak)
    # constant rungs are excluded from the band by construction
    assert "majority" not in summary["auc_by_rung"]
    assert "mean" not in summary["r2_by_rung"]
    small_n_band = {"auc_low": 0.42, "auc_high": 0.58, "r2_high": 0.03}
    check = ladder.band_check(summary, small_n_band)
    assert check["auc_outside_band"] == ["leak_control"]
    assert check["r2_above_bound"] == ["leak_ols"]
    assert check["passes"] is False
    assert summary["auc_by_rung"]["leak_control"]["mean"] > 0.95


def test_shuffle_keeps_the_real_label_folds(shuffled_with_leak):
    assert ladder.fold_checks(shuffled_with_leak)["identical_folds_all_rungs"] is True
    assert shuffled_with_leak["shared_group_pairs"] == 0


def test_frozen_shuffled_control_inside_registered_band(frozen):
    band = frozen["band"]
    for key in frozen["signals"]:
        leakage = frozen["signals"][key]["leakage"]
        assert leakage["band_check"]["passes"] is True
        assert "majority" not in leakage["auc_by_rung"]
        assert "mean" not in leakage["r2_by_rung"]
        for name in leakage["auc_by_rung"]:
            value = leakage["auc_by_rung"][name]["mean"]
            assert band["auc_low"] <= value <= band["auc_high"], (key, name)
        for name in leakage["r2_by_rung"]:
            assert leakage["r2_by_rung"][name]["mean"] <= band["r2_high"], (key, name)


def test_band_was_set_from_trial_seeds(frozen, trial):
    assert frozen["band"] == trial["band"]
    assert frozen["r2_gate"] == trial["r2_gate"]
    assert set(trial["cv_seeds"]).isdisjoint(set(frozen["cv_seeds"]))
    assert trial["data_seed"] != frozen["preset"]["data_seed"]


def test_band_formula_hand_fixture():
    residual_set = {"0.0": {"shuffled_auc_residuals": [0.01, -0.01, 0.02, -0.02],
                            "shuffled_r2_residuals": [0.001, -0.001],
                            "shuffled_n_auc_rungs": 2, "shuffled_n_r2_rungs": 1,
                            "velocity_ols_r2_fold_sd": 0.0121}}
    band, gate = ladder.band_from_trial(residual_set)
    # pooled sd = sqrt((1e-4 + 1e-4 + 4e-4 + 4e-4) / (4 - 2)) = sqrt(5e-4)
    expected_sd = math.sqrt(5e-4)
    assert band["single_fold_auc_sd"] == pytest.approx(expected_sd)
    assert band["auc_half_width"] == pytest.approx(math.ceil(3 * expected_sd * 1000) / 1000)
    assert band["auc_low"] == pytest.approx(0.5 - band["auc_half_width"])
    assert gate == pytest.approx(0.013)


@pytest.fixture(scope="module")
def leak_committed():
    with open(os.path.join(RESULTS, ladder.LEAK_FILENAME)) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def leak_recomputed(frozen):
    """One run of the moderate-leak control (about 2 s)."""
    return json.loads(json.dumps(ladder.clean_for_json(ladder.leak_detection(frozen["band"]))))


def test_leak_detection_file_is_fresh(leak_recomputed, leak_committed):
    result = buylist.recompute_problems(leak_recomputed, leak_committed)
    assert result["problems"] == [], result


def test_band_misses_a_moderate_leak_and_catches_a_stronger_one(leak_committed, frozen):
    """The registered band's detection limit, recorded: a leak at shuffled
    fold-mean AUC about 0.53 passes the band, one about 0.54 or above is
    caught, and the missed leak is still far above every honest rung."""
    band = frozen["band"]
    assert leak_committed["band"]["auc_high"] == band["auc_high"]
    missed = leak_committed["largest_missed"]
    caught = leak_committed["smallest_caught"]
    assert missed["caught"] is False
    assert 0.52 < missed["fold_mean_auc"] <= band["auc_high"]
    assert caught["caught"] is True
    assert band["auc_high"] < caught["fold_mean_auc"] < 0.56
    _lowest, highest = ladder.honest_shuffled_auc_range(frozen)
    assert missed["fold_mean_auc"] > highest + 0.02


def test_control_leak_strength_follows_the_noise(leak_committed):
    """Control: less noise = a stronger leak, so fold-mean AUC falls as the
    noise sd rises, and the strongest leak (sd 2) is caught."""
    rows = leak_committed["rows"]
    previous = None
    for row in rows:
        if previous is not None:
            assert row["noise_sd"] > previous["noise_sd"]
            assert row["fold_mean_auc"] < previous["fold_mean_auc"]
        previous = row
    assert rows[0]["caught"] is True
    assert rows[-1]["caught"] is False


def test_honest_shuffled_range_hand_fixture():
    fixture = {"signals": {"0.0": {"leakage": {"auc_by_rung": {"a": {"mean": 0.49},
                                                               "b": {"mean": 0.51}}}},
                           "0.9": {"leakage": {"auc_by_rung": {"a": {"mean": 0.505}}}}}}
    assert ladder.honest_shuffled_auc_range(fixture) == (0.49, 0.51)


# ---------------------------------------------------------------- oracle

def test_live_no_model_beats_the_oracle(live_run):
    _frame, _manifest, _truth, raw = live_run
    class_rows = ladder.summarize_classification(raw, 50, 1)
    reg_rows = ladder.summarize_regression(raw, 0.01, 50, 1)
    assert ladder.oracle_bound_violations(class_rows, reg_rows, LIVE_ORACLE_TOLERANCE) == []


def test_oracle_check_catches_a_rung_above_truth():
    class_rows = [{"name": "oracle", "constant": False, "auc_folds": {"mean": 0.80}},
                  {"name": "cheat", "constant": False, "auc_folds": {"mean": 0.85}},
                  {"name": "fine", "constant": False, "auc_folds": {"mean": 0.805}}]
    reg_rows = [{"name": "oracle", "r2_folds": {"mean": 0.5},
                 "deviance_folds": {"mean": 1.0}},
                {"name": "dev_cheat", "r2_folds": {"mean": 0.4},
                 "deviance_folds": {"mean": 0.9}},
                {"name": "r2_cheat", "r2_folds": {"mean": 0.6}, "deviance_folds": None}]
    violations = ladder.oracle_bound_violations(class_rows, reg_rows, 0.01)
    assert violations == ["cheat:auc", "dev_cheat:deviance", "r2_cheat:r2"]


def test_frozen_no_model_beats_the_oracle(frozen):
    for key in frozen["signals"]:
        block = frozen["signals"][key]
        assert block["oracle_violations"] == []
        recomputed = ladder.oracle_bound_violations(block["classification"],
                                                    block["regression"])
        assert recomputed == []


def test_frozen_poisson_glm_within_003_of_oracle_at_signal_zero(frozen):
    gap = frozen["signals"]["0.0"]["glm_oracle_deviance_gap"]
    assert gap is not None
    assert -ladder.ORACLE_AUC_TOLERANCE <= gap <= ladder.GLM_ORACLE_DEVIANCE_TOLERANCE


# ---------------------------------------------------------------- rung facts

def test_frozen_glm_gap_at_planted_signal_is_printed_not_bounded(frozen):
    """The 0.03 GLM-oracle bound was registered for signal 0 only; at 0.9 the
    gap is far above it, and LADDER.md / DECISIONS.md must say so."""
    tolerance = ladder.GLM_ORACLE_DEVIANCE_TOLERANCE
    planted_gap = frozen["signals"]["0.9"]["glm_oracle_deviance_gap"]
    assert planted_gap > tolerance
    with open(os.path.join(RESULTS, "LADDER.md")) as handle:
        ladder_text = handle.read()
    assert ladder_report.glm_gap_sentence(frozen["signals"]["0.9"], "0.9") in ladder_text
    assert "not bounded here" in ladder_report.glm_gap_sentence(frozen["signals"]["0.9"], "0.9")
    assert "bounded at" in ladder_report.glm_gap_sentence(frozen["signals"]["0.0"], "0.0")
    assert "checked at signal 0 only" in ladder_report.WHY_THIS_METRIC["count_ladder"]["lies"]


def answer_rows(top_auc):
    rows = []
    entries = [("majority", 0.5, True), ("velocity", 0.87, False),
               ("logistic_l2_c1", 0.966, False), ("gbc", top_auc, False),
               ("oracle", 0.99, False)]
    for name, auc, constant in entries:
        rows.append({"name": name, "constant": constant, "auc_folds": {"mean": auc},
                     "beats_velocity": name in ("logistic_l2_c1", "gbc")})
    return rows


def test_answer_says_the_simpler_model_is_kept_under_the_bar():
    text = ladder_report.classification_answer(answer_rows(0.9744), "0.9")
    assert "gbc leads logistic_l2_c1 by +0.0084 fold-mean AUC, under the +0.01 bar, so the " \
           "simpler model is kept." in text


def test_control_answer_has_no_margin_sentence_over_the_bar_or_without_a_simpler_rung():
    over_bar = ladder_report.classification_answer(answer_rows(0.9800), "0.9")
    assert "simpler model is kept" not in over_bar
    rows = answer_rows(0.80)
    no_simpler = ladder_report.simpler_rung_margin_text(rows, rows[1])
    assert no_simpler == ""


def test_json_line_steps_into_the_first_list_element():
    text = json.dumps({"a": [{"b": 1, "c": 2}, {"b": 3}], "d": {"b": 4}}, indent=1,
                      sort_keys=True)
    lines = text.split("\n")
    assert lines[ladder_report.json_line(text, ["a", 0, "c"]) - 1].strip() == '"c": 2'
    assert lines[ladder_report.json_line(text, ["d", "b"]) - 1].strip() == '"b": 4'
    with pytest.raises(ValueError):
        ladder_report.json_line(text, ["a", 1, "b"])


def test_readme_kpi_citations_point_at_the_cited_keys():
    """Every `results/...json:N` in the README ladder section lands on a key
    line; each KPI row cites both signals."""
    with open(os.path.join(REPO_ROOT, "README.md")) as handle:
        readme = handle.read()
    section = ladder_report.extract_block(readme, ladder_report.README_SECTION)
    citations = re.findall(r"`(results/[a-z_]+\.json):(\d+)`", section)
    assert len(citations) >= 11
    for rel_path, line_text in citations:
        with open(os.path.join(REPO_ROOT, rel_path)) as handle:
            lines = handle.read().split("\n")
        assert lines[int(line_text) - 1].lstrip().startswith('"'), (rel_path, line_text)
    rows = section.split("\n")
    for row in rows:
        if row.startswith("| Lift") or row.startswith("| Calibration") or \
                row.startswith("| Decision cost") or row.startswith("| Count-regression"):
            assert "0.9 `results/" in row and "; 0 `results/" in row, row[:40]


def test_unpenalized_rung_is_really_unpenalized():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, 3))
    logit = 2.0 * features[:, 0] - 1.0 * features[:, 1]
    labels = (rng.uniform(size=120) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    rung = ladder.rung_by_name(ladder.CLASSIFICATION_RUNGS, "logistic_unpenalized")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probs = ladder.fit_predict_classifier(rung, features, features, labels, 120)
    assert caught == []
    huge_c = LogisticRegression(C=1e10, max_iter=5000).fit(features, labels)
    assert np.allclose(probs, huge_c.predict_proba(features)[:, 1], atol=1e-5)
    l2_rung = ladder.rung_by_name(ladder.CLASSIFICATION_RUNGS, "logistic_l2_c1")
    l2_probs = ladder.fit_predict_classifier(l2_rung, features, features, labels, 120)
    assert np.max(np.abs(l2_probs - probs)) > 1e-3  # the L2 rung IS penalized


def test_expected_log1p_poisson_hand_fixture():
    values = ladder.expected_log1p_poisson(np.array([0.0, 1.0, 0.5]))
    assert values[0] == 0.0
    for position, mu in ((1, 1.0), (2, 0.5)):
        total = 0.0
        counts = range(60)
        for count in counts:
            pmf = math.exp(-mu) * mu ** count / math.factorial(count)
            total += math.log1p(count) * pmf
        assert values[position] == pytest.approx(total, abs=1e-12)


def test_parked_capital_rate_hand_fixture():
    y = np.array([1, 0, 1, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.1, 0.95])
    assert ladder.parked_capital_rate(y, probs, 2.0 / 3.0) == pytest.approx(0.5)
    assert ladder.parked_capital_rate(y, probs, 0.99) is None


def test_paired_bootstrap_is_paired():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(20), 10).astype(str)
    y = rng.integers(0, 2, size=200)
    probs = rng.uniform(size=200)
    same = ladder.paired_group_bootstrap_delta(y, probs, probs, groups, ladder.auc_or_none,
                                               100, 1)
    assert same["ci_low"] == 0.0
    assert same["ci_high"] == 0.0
    better = np.where(y == 1, probs + 0.5, probs)
    lifted = ladder.paired_group_bootstrap_delta(y, better, probs, groups,
                                                 ladder.auc_or_none, 100, 1)
    assert lifted["ci_low"] > 0.0


# ---------------------------------------------------------------- frozen KPI

def test_frozen_parked_rates_match_their_confusions(frozen):
    for key in frozen["signals"]:
        kpi = frozen["signals"][key]["kpi"]
        assert kpi["t_star"] == pytest.approx(2.0 / 3.0)
        for side in ("model", "velocity"):
            confusion = kpi[side + "_confusion"]
            total = confusion["tp"] + confusion["fp"] + confusion["fn"] + confusion["tn"]
            assert total == confusion["n"] == frozen["preset"]["n_rows"]
            flagged = confusion["tp"] + confusion["fp"]
            assert kpi[side + "_parked"] == pytest.approx(confusion["fp"] / flagged)
        expected_delta = kpi["velocity_parked"] - kpi["model_parked"]
        assert kpi["delta_parked"]["value"] == pytest.approx(expected_delta)


def test_frozen_fold_integrity(frozen):
    for key in frozen["signals"]:
        for block_name in ("folds", "shuffled_folds"):
            folds = frozen["signals"][key][block_name]
            assert folds["identical_folds_all_rungs"] is True
            assert folds["shared_group_pairs_outer"] == 0
            assert folds["internal_split_rungs"] == []


def test_frozen_beats_flags_follow_the_rule(frozen):
    for key in frozen["signals"]:
        for row in frozen["signals"][key]["classification"]:
            delta = row["delta_auc_vs_velocity"]
            if delta is None:
                continue
            expected = ladder.beats_baseline(delta["value"], delta["ci_low"],
                                             delta["ci_high"], frozen["ship_min_delta_auc"])
            assert row["beats_velocity"] is expected
        for row in frozen["signals"][key]["regression"]:
            delta = row["delta_r2_vs_velocity"]
            if delta is None:
                continue
            expected = ladder.beats_baseline(delta["value"], delta["ci_low"],
                                             delta["ci_high"], frozen["r2_gate"])
            assert row["beats_velocity"] is expected


# ---------------------------------------------------------------- registration

def test_predictions_file_pinned_and_written_from_trial(frozen, trial):
    path = os.path.join(RESULTS, frozen["predictions_file"])
    with open(path, "rb") as handle:
        data = handle.read()
    digest = hashlib.sha256(data).hexdigest()
    assert digest[:ladder.SHA_PREFIX_LEN] == frozen["predictions_sha256_prefix"]
    assert data.decode("utf-8") == ladder_predictions.predictions_markdown(trial)
    assert ladder.REGISTRATION_DATE in frozen["predictions_file"]


def test_verdicts_are_recomputed_by_code(frozen):
    without = dict(frozen)
    without.pop("verdicts")
    assert ladder_predictions.judge_all(without) == frozen["verdicts"]


def test_results_json_holds_aggregates_only(frozen, trial):
    banned_keys = {"card_id", "group_id", "release_year"}
    n_rows = frozen["preset"]["n_rows"]

    def walk(value):
        if isinstance(value, dict):
            for key in value:
                assert key not in banned_keys
                walk(value[key])
        elif isinstance(value, list):
            assert len(value) < 50, "a list this long could be per-card data"
            assert len(value) != n_rows
            for item in value:
                walk(item)

    walk(frozen)
    walk(trial)
    for name in (ladder.FROZEN_FILENAME, ladder.TRIAL_FILENAME):
        size = os.path.getsize(os.path.join(RESULTS, name))
        assert size < 150_000
