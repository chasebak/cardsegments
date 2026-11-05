"""Buy-list grid: hand fixtures, grid invariants, regret, calibration
(Brier, Brier skill, ECE), the frozen-run match and freshness of
results/buylist_grid.json.

Why: the page and the README explorer block copy these numbers, so the
logic checks of the plan (confusion grid, regret, velocity comparison,
disclosure) live here as tests that can fail; each check has a control
that must fail.
"""

import copy
import json
import math
import os

import numpy as np

from cardsegments import buylist, explorer, ladder
from cardsegments.supervised import COST_SETTINGS_DEFAULT, threshold_cost_curve

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGIT_SHIFT = 1.5


def committed_grid():
    with open(os.path.join(REPO_ROOT, buylist.GRID_PATH_REL)) as handle:
        return json.load(handle)


def committed_frozen():
    with open(os.path.join(REPO_ROOT, "results", ladder.FROZEN_FILENAME)) as handle:
        return json.load(handle)


def rounded(value):
    return json.loads(json.dumps(ladder.clean_for_json(value), sort_keys=True))


def without_seconds(payload):
    stripped = copy.deepcopy(payload)
    for key in stripped["signals"]:
        stripped["signals"][key].pop("seconds")
    return stripped


# ---------------------------------------------------------------- hand fixtures

def test_grid_row_hand_fixture():
    y = np.array([1, 0, 1, 0])
    probs = np.array([0.9, 0.8, 0.3, 0.1])
    row = buylist.grid_row(y, probs, 0.5)
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 1, 1)
    assert row["n_flagged"] == 2
    assert row["parked"] == 0.5
    assert row["missed_share"] == 0.5
    on_threshold = buylist.grid_row(y, probs, 0.8)
    assert on_threshold["n_flagged"] == 2  # P >= t: a tie goes on the list


def test_empty_list_has_no_parked_rate_not_zero():
    y = np.array([1, 0, 1, 0])
    probs = np.array([0.9, 0.8, 0.3, 0.1])
    row = buylist.grid_row(y, probs, 0.95)
    assert row["n_flagged"] == 0
    assert row["parked"] is None
    assert row["missed_share"] == 1.0


def test_decision_cost_hand_fixture():
    cost = buylist.decision_cost(fp=2, fn=3, n_rows=10, m=0.10, h=0.10)
    assert math.isclose(cost, (0.20 * 2 + 0.10 * 3) / 10)


def test_regret_hand_fixture():
    """t* = 2/3 flags one seller and misses one: cost 0.1/4 = 0.025; the
    hindsight grid point 0.5 flags both sellers and no sitter: cost 0."""
    y = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.6, 0.3, 0.1])
    block = buylist.threshold_regret(y, probs, 0.10, 0.10, grid=np.array([0.2, 0.5, 0.8]))
    assert math.isclose(block["cost_t_star"], 0.025)
    assert block["best_grid_cost"] == 0.0
    assert block["best_grid_threshold"] == 0.5
    assert math.isclose(block["regret"], 0.025)
    assert block["regret_share"] == 1.0
    assert math.isclose(block["buy_none_cost"], 0.10 * 2 / 4)


def test_regret_is_zero_when_t_star_beats_the_grid():
    """Control for the min(): t* cheaper than every grid point gives 0, never < 0."""
    y = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.6, 0.55, 0.1])
    block = buylist.threshold_regret(y, probs, 0.10, 0.10, grid=np.array([0.2]))
    assert block["best_grid_cost"] > block["cost_t_star"]
    assert block["regret"] == 0.0


# ---------------------------------------------------------------- calibration

def test_base_rate_predictor_has_brier_skill_exactly_zero():
    """p_bar = 1/4 is exact in binary, so Brier = p_bar (1 - p_bar) exactly."""
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    base_rate = np.full(8, 0.25)
    assert buylist.brier_score(y, base_rate) == 0.1875
    assert buylist.brier_skill(y, base_rate) == 0.0


def test_control_brier_skill_moves_away_from_zero():
    """Controls: perfect scores give 1, a flat 0.5 does worse than the base
    rate (< 0), and a single-class label has no skill (None)."""
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    assert buylist.brier_skill(y, y.astype(float)) == 1.0
    assert buylist.brier_skill(y, np.full(8, 0.5)) < 0.0
    assert buylist.brier_skill(np.zeros(8), np.full(8, 0.1)) is None


def three_bin_fixture():
    """150 cards in 3 bins of 50: predicted 0.1 / 0.5 / 0.9, sold 10 / 25 / 40."""
    probs = np.concatenate([np.full(50, 0.1), np.full(50, 0.5), np.full(50, 0.9)])
    y = np.zeros(150, dtype=int)
    y[0:10] = 1
    y[50:75] = 1
    y[100:140] = 1
    return y, probs


def test_ece_three_bin_hand_fixture():
    """|0.1 - 0.2| x 50 + |0.5 - 0.5| x 50 + |0.9 - 0.8| x 50, over 150 = 1/15."""
    y, probs = three_bin_fixture()
    block = buylist.ece(y, probs, min_bin_n=50, n_bins=3)
    assert block["n_bins"] == 3
    assert block["smallest_bin_n"] == 50
    assert math.isclose(block["ece"], 1.0 / 15.0)


def test_control_ece_is_zero_on_calibrated_bins_and_merges_small_bins():
    y, probs = three_bin_fixture()
    calibrated = np.concatenate([np.full(50, 0.2), np.full(50, 0.5), np.full(50, 0.8)])
    assert buylist.ece(y, calibrated, min_bin_n=50, n_bins=3)["ece"] < 1e-12
    # floor 60: bins 1+2 merge (100 cards) and the short trailing bin joins
    # them, so one bin of 150 whose mean prediction equals its sell rate
    merged = buylist.ece(y, probs, min_bin_n=60, n_bins=3)
    assert merged["n_bins"] == 1
    assert merged["smallest_bin_n"] == 150
    assert merged["ece"] < 1e-12


def test_committed_calibration_bins_meet_the_floor():
    grid = committed_grid()
    for key in grid["signals"]:
        for name in buylist.MODEL_NAMES:
            block = grid["signals"][key]["models"][name]["calibration"]
            assert block["ece_min_bin_n"] == 50
            assert block["ece_smallest_bin_n"] >= 50
            assert 0.0 <= block["ece"] < 1.0
            assert math.isclose(block["brier_skill"],
                                1.0 - block["brier"] / block["base_rate_brier"], rel_tol=1e-8)


def test_paired_delta_of_identical_lists_is_exactly_zero():
    """Both lists are scored on the SAME resampled groups: identical
    probabilities give a delta of 0 in every replicate. An unpaired
    bootstrap would give a CI of nonzero width."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    probs = rng.random(400)
    groups = np.repeat(np.arange(40), 10)
    block = buylist.parked_delta_ci(y, probs, probs.copy(), groups, 0.5, 200, 1)
    assert block["value"] == 0.0
    assert block["ci_low"] == 0.0 and block["ci_high"] == 0.0
    shuffled = probs[rng.permutation(400)]
    unpaired = buylist.parked_delta_ci(y, probs, shuffled, groups, 0.5, 200, 1)
    assert unpaired["ci_high"] - unpaired["ci_low"] > 0.0


# ---------------------------------------------------------------- grid invariants

def grid_problems(rows, n_rows):
    """'t=..: rule' for each row breaking the confusion-grid invariants.

    Rules: TP+FP+FN+TN = n; n_flagged = TP+FP; parked = FP/n_flagged (None
    when empty); as t rises FP and n_flagged never rise and FN never falls.
    """
    problems = []
    previous = None
    for row in rows:
        label = "t=%.2f" % row["threshold"]
        if row["tp"] + row["fp"] + row["fn"] + row["tn"] != n_rows:
            problems.append(label + ": sum")
        if row["n_flagged"] != row["tp"] + row["fp"]:
            problems.append(label + ": flagged")
        if row["n_flagged"] == 0:
            if row["parked"] is not None:
                problems.append(label + ": parked")
        elif not math.isclose(row["parked"], row["fp"] / row["n_flagged"], rel_tol=1e-9):
            problems.append(label + ": parked")
        if previous is not None:
            if row["fp"] > previous["fp"]:
                problems.append(label + ": FP rose")
            if row["fn"] < previous["fn"]:
                problems.append(label + ": FN fell")
            if row["n_flagged"] > previous["n_flagged"]:
                problems.append(label + ": list grew")
        previous = row
    return problems


def test_committed_grid_obeys_the_invariants():
    grid = committed_grid()
    for key in grid["signals"]:
        block = grid["signals"][key]
        for name in buylist.MODEL_NAMES:
            rows = block["models"][name]["grid"]
            assert len(rows) == 49
            assert grid_problems(rows, block["n_rows"]) == []


def test_control_grid_check_catches_a_rising_fp_and_a_lost_card():
    grid = committed_grid()
    block = grid["signals"]["0.9"]
    rows = copy.deepcopy(block["models"][ladder.HEADLINE_MODEL]["grid"])
    rows[30]["fp"] = rows[29]["fp"] + 1
    rows[10]["tn"] = rows[10]["tn"] - 1
    problems = grid_problems(rows, block["n_rows"])
    assert "t=0.22: sum" in problems
    assert "t=0.62: FP rose" in problems


def test_grid_thresholds_are_the_supervised_cost_curve_grid():
    grid = committed_grid()
    rows = grid["signals"]["0.0"]["models"][ladder.BASELINE_CLASSIFIER]["grid"]
    thresholds = []
    for row in rows:
        thresholds.append(row["threshold"])
    expected = rounded(list(np.arange(0.02, 0.99, 0.02)))
    assert thresholds == expected
    assert grid["grid"] == {"start": 0.02, "stop": 0.98, "n_points": 49}


def test_regret_is_non_negative_in_every_committed_cell():
    grid = committed_grid()
    cells = explorer.all_cells(grid)
    assert len(cells) == 2 * 3 * 2
    for _key, _setting, _name, block in cells:
        assert block["regret"] >= 0.0
        floor = min(block["best_grid_cost"], block["cost_t_star"])
        assert math.isclose(block["regret"], block["cost_t_star"] - floor, abs_tol=1e-10)
        assert len(block["grid_costs"]) == 49


# ---------------------------------------------------------------- against the functions

def test_committed_grid_is_fresh(buylist_recompute):
    """The recompute reproduces the committed grid: exactly, or (after a
    cross-platform lbfgs difference) with at most one card moved per cell.
    Moved cells are reported in the assertion message."""
    payload, _probs = buylist_recompute
    result = buylist.recompute_problems(without_seconds(rounded(payload)),
                                        without_seconds(committed_grid()))
    assert result["problems"] == [], result
    print("cells moved by one card:", result["moved_counts"])


def test_recompute_tolerance_rules():
    committed = {"tp": [10, 20], "parked": [0.5, 0.25], "cost": 0.0123456789, "label": "S"}
    assert buylist.recompute_problems(copy.deepcopy(committed), committed) == {
        "problems": [], "moved_counts": []}
    one_card = copy.deepcopy(committed)
    one_card["tp"][1] = 21
    one_card["parked"][1] = 0.24
    one_card["cost"] = 0.0123556789
    moved = buylist.recompute_problems(one_card, committed)
    assert moved["problems"] == []
    assert moved["moved_counts"] == ["/tp[1]: 20 -> 21"]


def test_control_recompute_tolerance_catches_real_drift():
    committed = {"tp": [10, 20], "parked": [0.5, 0.25], "cost": 0.0123456789, "label": "S"}
    two_cards = copy.deepcopy(committed)
    two_cards["tp"][0] = 12
    assert buylist.recompute_problems(two_cards, committed)["problems"] == [
        "/tp[0]: count 10 -> 12 (more than one card)"]
    float_only = copy.deepcopy(committed)
    float_only["cost"] = 0.0123456790
    assert buylist.recompute_problems(float_only, committed)["problems"] == [
        "/cost: 0.0123456789 -> 0.012345679"]
    relabelled = copy.deepcopy(committed)
    relabelled["label"] = "R"
    assert buylist.recompute_problems(relabelled, committed)["problems"] == ["/label: 'S' -> 'R'"]
    reshaped = copy.deepcopy(committed)
    reshaped["tp"].append(30)
    assert buylist.recompute_problems(reshaped, committed)["problems"] == [
        "/tp: 'length' -> 'length'"]


def test_grid_costs_reproduce_threshold_cost_curve(buylist_recompute):
    payload, probs = buylist_recompute
    for key in probs:
        y, _groups, model_probs, velocity_probs = probs[key]
        by_model = {ladder.HEADLINE_MODEL: model_probs, ladder.BASELINE_CLASSIFIER: velocity_probs}
        settings = payload["signals"][key]["cost_settings"]
        positions = range(len(COST_SETTINGS_DEFAULT))
        for position in positions:
            m, h = COST_SETTINGS_DEFAULT[position]
            for name in buylist.MODEL_NAMES:
                curve = threshold_cost_curve(y, by_model[name], m, h)
                curve_costs = []
                for point in curve["points"]:
                    curve_costs.append(point["cost"])
                grid_costs = settings[position]["models"][name]["grid_costs"]
                assert rounded(grid_costs) == rounded(curve_costs)
                assert math.isclose(settings[position]["t_star"], curve["t_star"])


def test_control_cost_curve_check_fails_under_other_costs(buylist_recompute):
    payload, probs = buylist_recompute
    y, _groups, model_probs, _velocity = probs["0.9"]
    curve = threshold_cost_curve(y, model_probs, 0.10, 0.30)
    curve_costs = []
    for point in curve["points"]:
        curve_costs.append(point["cost"])
    grid_costs = payload["signals"]["0.9"]["cost_settings"][0]["models"][
        ladder.HEADLINE_MODEL]["grid_costs"]
    assert rounded(grid_costs) != rounded(curve_costs)


def test_shifted_logits_raise_the_regret_and_the_ece(buylist_recompute):
    """Miscalibration control: inflating every logit by 1.5 moves the
    hindsight-best threshold away from t*, so regret must grow, and every
    bin's average prediction moves above its sell rate, so ECE must grow
    and Brier skill must fall."""
    _payload, probs = buylist_recompute
    y, _groups, model_probs, _velocity = probs["0.9"]
    clipped = np.clip(model_probs, 1e-12, 1 - 1e-12)
    logits = np.log(clipped / (1 - clipped))
    shifted = 1.0 / (1.0 + np.exp(-(logits + LOGIT_SHIFT)))
    honest = buylist.threshold_regret(y, model_probs, 0.10, 0.10)
    miscalibrated = buylist.threshold_regret(y, shifted, 0.10, 0.10)
    assert miscalibrated["regret"] > honest["regret"]
    assert miscalibrated["regret_share"] > 2 * honest["regret_share"]
    honest_ece = buylist.ece(y, model_probs)["ece"]
    shifted_ece = buylist.ece(y, shifted)["ece"]
    assert shifted_ece > 2 * honest_ece
    assert buylist.brier_skill(y, shifted) < buylist.brier_skill(y, model_probs)


def test_both_lists_scored_at_the_same_t_star(buylist_recompute):
    payload, probs = buylist_recompute
    for key in probs:
        y, _groups, model_probs, velocity_probs = probs[key]
        setting = payload["signals"][key]["cost_settings"][0]
        model_cell = setting["models"][ladder.HEADLINE_MODEL]["at_t_star"]
        velocity_cell = setting["models"][ladder.BASELINE_CLASSIFIER]["at_t_star"]
        assert model_cell["threshold"] == velocity_cell["threshold"] == setting["t_star"]
        expected = ladder.parked_capital_rate(y, velocity_probs, setting["t_star"]) - \
            ladder.parked_capital_rate(y, model_probs, setting["t_star"])
        assert setting["delta_parked"]["value"] == expected


# ---------------------------------------------------------------- frozen match

def test_t_star_rows_equal_the_frozen_ladder_kpi():
    assert explorer.frozen_match_problems(committed_grid(), committed_frozen()) == []


def test_control_frozen_match_catches_one_changed_count():
    grid = committed_grid()
    cell = grid["signals"]["0.0"]["cost_settings"][0]["models"][
        ladder.BASELINE_CLASSIFIER]["at_t_star"]
    cell["fp"] = cell["fp"] + 1
    grid["signals"]["0.9"]["cost_settings"][0]["delta_parked"]["ci_low"] = 0.0
    problems = explorer.frozen_match_problems(grid, committed_frozen())
    assert problems == ["signal 0.0: velocity fp differs from the frozen run",
                        "signal 0.9: delta_parked ci_low differs from the frozen run"]


# ---------------------------------------------------------------- disclosure

def test_grid_file_is_synthetic_aggregates_only():
    grid = committed_grid()
    assert grid["label"].startswith("SYNTHETIC")
    keys = explorer.collect_keys(grid, set())
    assert keys & set(explorer.BANNED_KEYS) == set()
    n_rows = grid["signals"]["0.0"]["n_rows"]
    assert explorer.longest_list(grid) <= explorer.MAX_LIST_LENGTH < n_rows
