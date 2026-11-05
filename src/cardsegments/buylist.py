"""Buy-list threshold grid for the Buy List Explorer page (SYNTHETIC only).

    python -m cardsegments.buylist build    # ~1 min locally; never in CI

Recomputes the pooled out-of-fold P(liquid) of the two buy-list rungs, the
full logistic (L2, C = 1; ladder.HEADLINE_MODEL) and the velocity logistic
(trailing sales; ladder.BASELINE_CLASSIFIER), on the frozen ladder's exact
protocol: the illustrative preset (synth.illustrative_preset), the same 10
CV seeds, the same StratifiedGroupKFold folds (ladder.cross_validate_ladder
with only those two rungs). Logistic fits do not depend on which other
rungs run, so the pooled probabilities are the frozen run's; the tests
prove it by matching every t* confusion count and the parked-rate CI in
results/ladder_frozen.json.

From the two probability vectors it writes AGGREGATES only to
results/buylist_grid.json:
  - a 49-row confusion grid per model (t = 0.02, 0.04, ..., 0.98, the grid
    of supervised.threshold_cost_curve): TP, FP, FN, TN, the parked rate
    FP / (TP + FP) and the missed-seller share FN / (TP + FN);
  - for each of the three preset cost settings (supervised.
    COST_SETTINGS_DEFAULT), the EXACT t* = (m + h) / (2m + h) cell (2/3 and
    3/4 are not grid points), the decision cost per card at every grid
    point, the hindsight-best grid threshold, the threshold regret, and a
    paired group-bootstrap CI of parked(velocity) - parked(model) at t*;
  - per model, the pooled calibration scores: Brier, Brier skill against
    the base-rate predictor, and ECE over quantile bins with n >= 50.
No per-card value, card id or group id leaves this module; the longest
list is the 49-row grid.

Failure mode: the recompute is only as good as its match to the frozen
run; if a future change to the ladder makes the logistic fits differ, the
t*-match tests in tests/test_buylist.py fail instead of the page drifting.
"""

import argparse
import os
import time

import numpy as np

from cardsegments import ladder, report, synth
from cardsegments.supervised import (
    COST_SETTINGS_DEFAULT,
    calibration_bins,
    confusion_at_threshold,
    cost_threshold,
)

GRID_FILENAME = "buylist_grid.json"
GRID_PATH_REL = "results/" + GRID_FILENAME
# The grid of supervised.threshold_cost_curve (its default), so the page's
# cost curves are the curves the supervised protocol already draws.
THRESHOLD_GRID = np.arange(0.02, 0.99, 0.02)
MODEL_NAMES = (ladder.HEADLINE_MODEL, ladder.BASELINE_CLASSIFIER)
LABEL = ("SYNTHETIC - illustrative preset (round values; not the private cohort)")
# Recompute-vs-committed tolerance (tests): count leaves may differ by one
# card; float leaves agree to this relative tolerance when no count differs.
COUNT_KEYS = ("tp", "fp", "fn", "tn", "n_flagged", "n", "n_rows", "n_sellers")
COUNT_TOLERANCE = 1
FLOAT_REL_TOLERANCE = 1e-9
FLOAT_ABS_TOLERANCE = 1e-12
# When a count did move by one card, the floats it feeds may move too:
# ratios of a list's own counts (re-checked on the committed grid by the
# invariant tests) are not compared, and other floats get this loose bound.
RATIO_KEYS = ("parked", "missed_share", "regret_share", "saving_vs_buy_none", "value",
              "ci_low", "ci_high")
FLIPPED_FLOAT_ABS_TOLERANCE = 5e-3


def rate_or_none(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def grid_row(y, probs, threshold):
    """One buy-list cell at threshold t.

    Rule: a card is on the buy list when P(liquid) >= t (the comparison of
    supervised.confusion_at_threshold). parked = FP / (TP + FP), the share
    of the list that sat; missed_share = FN / (TP + FN), the share of the
    cards that went on to sell that the list passed over. Failure mode: an
    empty list has no parked rate (None, drawn as a gap), never 0.
    """
    confusion = confusion_at_threshold(y, probs, threshold)
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    tn = confusion["tn"]
    n_flagged = tp + fp
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_flagged": n_flagged,
            "parked": rate_or_none(fp, n_flagged),
            "missed_share": rate_or_none(fn, tp + fn)}


def confusion_grid(y, probs, grid=THRESHOLD_GRID):
    """grid_row at every threshold of the grid, lowest threshold first."""
    rows = []
    for threshold in grid:
        rows.append(grid_row(y, probs, threshold))
    return rows


def decision_cost(fp, fn, n_rows, m, h):
    """Decision cost per card: ((m + h) FP + m FN) / N (illustrative units).

    A bought card that sits costs its margin m plus holding cost h; a
    passed card that would have sold costs the margin m it would have
    earned. Baseline: "buy none" costs m x (number of sellers) / N. Failure
    mode: m and h are illustrative, so the cost is a ranking device, not a
    dollar figure.
    """
    return ((m + h) * fp + m * fn) / n_rows


def threshold_regret(y, probs, m, h, grid=THRESHOLD_GRID):
    """How close the cost-set threshold t* comes to the hindsight best.

    Formula: regret = cost(t*) - min(min over the grid of cost(t), cost(t*)),
    with cost(t*) from the EXACT t* cell, not the nearest grid point. So
    regret >= 0 always, and regret = 0 means no grid threshold, chosen after
    seeing the outcomes, would have done better. Baseline: the hindsight-best
    grid threshold (which needs the test labels; t* needs only m and h).
    Failure mode: miscalibrated probabilities push the best threshold away
    from t*, which shows up here as regret > 0.
    """
    n_rows = len(y)
    t_star = cost_threshold(m, h)
    at_t_star = grid_row(y, probs, t_star)
    cost_t_star = decision_cost(at_t_star["fp"], at_t_star["fn"], n_rows, m, h)
    grid_costs = []
    best_cost = None
    best_threshold = None
    for threshold in grid:
        row = grid_row(y, probs, threshold)
        cost = decision_cost(row["fp"], row["fn"], n_rows, m, h)
        grid_costs.append(cost)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    floor_cost = min(best_cost, cost_t_star)
    regret = cost_t_star - floor_cost
    n_sellers = int(np.sum(y == 1))
    buy_none_cost = m * n_sellers / n_rows
    return {"m": m, "h": h, "t_star": t_star, "at_t_star": at_t_star,
            "cost_t_star": cost_t_star, "grid_costs": grid_costs,
            "best_grid_cost": best_cost, "best_grid_threshold": best_threshold,
            "regret": regret, "regret_share": rate_or_none(regret, cost_t_star),
            "buy_none_cost": buy_none_cost,
            "saving_vs_buy_none": rate_or_none(buy_none_cost - cost_t_star, buy_none_cost)}


def brier_score(y, probs):
    """Brier = mean (p - y)^2 over the cards (lower is better)."""
    y_float = np.asarray(y, dtype=float)
    p_float = np.asarray(probs, dtype=float)
    return float(np.mean((p_float - y_float) ** 2))


def brier_skill(y, probs):
    """Brier skill = 1 - Brier / (p_bar (1 - p_bar)), p_bar = mean(y).

    The reference p_bar (1 - p_bar) is the Brier of the base-rate
    predictor (every card scored p_bar), so that predictor scores 0, a
    perfect one 1, and anything below 0 does worse than quoting the base
    rate. Failure mode: when every card has the same label the reference
    is 0 and skill is undefined (None); skill mixes ranking and
    calibration, so a well-ranked list can score high while its 0.70 does
    not mean 7 in 10 (ECE sits beside it for that).
    """
    base_rate = float(np.mean(np.asarray(y, dtype=float)))
    reference = base_rate * (1.0 - base_rate)
    if reference == 0.0:
        return None
    return 1.0 - brier_score(y, probs) / reference


def ece(y, probs, min_bin_n=report.MIN_CELL_N, n_bins=10):
    """Expected calibration error over quantile bins with n >= min_bin_n.

    ECE = sum over bins b of (n_b / N) |mean predicted_b - observed rate_b|,
    on supervised.calibration_bins (n_bins quantile intervals, merged left
    to right until every bin holds at least min_bin_n cards, so no bin is
    below the n >= 50 floor). 0 = every bin's average prediction equals its
    sell rate. Failure mode: errors of opposite sign inside one bin cancel,
    so a small ECE does not prove calibration within a bin; with fewer than
    min_bin_n cards there is one bin and ECE is |mean p - base rate|.
    Returns {"ece", "n_bins", "smallest_bin_n"}.
    """
    y_array = np.asarray(y)
    p_array = np.asarray(probs, dtype=float)
    bins = calibration_bins(y_array, p_array, min_bin_n=min_bin_n, n_bins=n_bins)
    n_rows = len(y_array)
    total = 0.0
    smallest = None
    for row in bins:
        gap = abs(row["mean_predicted"] - row["observed_rate"])
        total += row["n"] * gap
        if smallest is None or row["n"] < smallest:
            smallest = row["n"]
    return {"ece": total / n_rows, "n_bins": len(bins), "smallest_bin_n": smallest}


def calibration_block(y, probs):
    """Brier, Brier skill and ECE of one pooled out-of-fold probability vector."""
    base_rate = float(np.mean(np.asarray(y, dtype=float)))
    binned = ece(y, probs)
    return {"brier": brier_score(y, probs), "brier_skill": brier_skill(y, probs),
            "base_rate_brier": base_rate * (1.0 - base_rate), "ece": binned["ece"],
            "ece_n_bins": binned["n_bins"], "ece_smallest_bin_n": binned["smallest_bin_n"],
            "ece_min_bin_n": report.MIN_CELL_N}


def parked_delta_ci(y, model_probs, velocity_probs, groups, threshold, n_boot, seed):
    """parked(velocity) - parked(model) at threshold, paired group bootstrap.

    The same resampled set groups score both lists in every replicate
    (ladder.paired_group_bootstrap_delta), exactly as ladder.buy_list_kpi
    does at t* = 2/3, so the default setting reproduces the frozen CI.
    Positive = the model's list parks less capital. Failure mode: a
    replicate with an empty list is skipped and counted (n_boot_skipped).
    """
    def parked_metric(y_b, probs_b):
        return ladder.parked_capital_rate(y_b, probs_b, threshold)

    model_parked = ladder.parked_capital_rate(y, model_probs, threshold)
    velocity_parked = ladder.parked_capital_rate(y, velocity_probs, threshold)
    if model_parked is None or velocity_parked is None:
        value = None
    else:
        value = velocity_parked - model_parked
    ci = ladder.paired_group_bootstrap_delta(y, velocity_probs, model_probs, groups,
                                             parked_metric, n_boot, seed)
    return {"value": value, **ci}


def signal_block(y, groups, model_probs, velocity_probs, n_boot, boot_seed,
                 cost_settings=COST_SETTINGS_DEFAULT, grid=THRESHOLD_GRID):
    """Grid, per-setting t* cells, regret and parked-delta CIs for one signal."""
    probs_by_model = {ladder.HEADLINE_MODEL: model_probs,
                      ladder.BASELINE_CLASSIFIER: velocity_probs}
    models = {}
    for name in MODEL_NAMES:
        models[name] = {"grid": confusion_grid(y, probs_by_model[name], grid),
                        "calibration": calibration_block(y, probs_by_model[name])}
    settings = []
    for m, h in cost_settings:
        regret_by_model = {}
        for name in MODEL_NAMES:
            regret_by_model[name] = threshold_regret(y, probs_by_model[name], m, h, grid)
        t_star = cost_threshold(m, h)
        delta = parked_delta_ci(y, model_probs, velocity_probs, groups, t_star,
                                n_boot, boot_seed)
        settings.append({"m": m, "h": h, "t_star": t_star, "models": regret_by_model,
                         "delta_parked": delta})
    return {"models": models, "cost_settings": settings,
            "n_rows": int(len(y)), "n_sellers": int(np.sum(y == 1)),
            "buy_everything_parked": float(1.0 - np.mean(y))}


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def leaf_differences(recomputed, committed, path, key, found):
    """Append (path, key, recomputed, committed) for every leaf that differs.

    Walks two JSON-like values in parallel; a dict whose keys differ or a
    list whose length differs is one difference at that path.
    """
    if isinstance(recomputed, dict) and isinstance(committed, dict):
        if sorted(recomputed.keys()) != sorted(committed.keys()):
            found.append((path, key, "keys", "keys"))
            return found
        names = sorted(recomputed.keys())
        for name in names:
            leaf_differences(recomputed[name], committed[name], path + "/" + name, name, found)
        return found
    if isinstance(recomputed, list) and isinstance(committed, list):
        if len(recomputed) != len(committed):
            found.append((path, key, "length", "length"))
            return found
        positions = range(len(recomputed))
        for position in positions:
            leaf_differences(recomputed[position], committed[position],
                             "%s[%d]" % (path, position), key, found)
        return found
    if recomputed != committed:
        found.append((path, key, recomputed, committed))
    return found


def recompute_problems(recomputed, committed):
    """Why a recompute does NOT reproduce a committed file (empty = it does).

    Rule: an exact match passes; so does one where count leaves (COUNT_KEYS)
    differ by at most one card and floats by FLOAT_REL_TOLERANCE (float
    noise of lbfgs across BLAS builds). If any count moved by one card, the
    ratios built on it (RATIO_KEYS) are not compared and other floats may
    differ by FLIPPED_FLOAT_ABS_TOLERANCE. A count off by more than one, a
    changed string, None or structure always fails. Each problem names its
    path, and so does each tolerated one-card move (returned too, so a test
    can report which cells differ). Failure mode: a stale file that differs
    by only one card per cell would pass; the frozen-run match tests
    (no recompute) still pin the t* cells exactly.
    """
    differences = leaf_differences(recomputed, committed, "", "", [])
    moved_counts = []
    problems = []
    float_diffs = []
    for path, key, new, old in differences:
        if key in COUNT_KEYS and is_number(new) and is_number(old):
            if abs(new - old) <= COUNT_TOLERANCE:
                moved_counts.append("%s: %s -> %s" % (path, old, new))
            else:
                problems.append("%s: count %s -> %s (more than one card)" % (path, old, new))
        elif is_number(new) and is_number(old):
            float_diffs.append((path, key, new, old))
        else:
            problems.append("%s: %r -> %r" % (path, old, new))
    for path, key, new, old in float_diffs:
        close = bool(np.isclose(new, old, rtol=FLOAT_REL_TOLERANCE, atol=FLOAT_ABS_TOLERANCE))
        if close:
            continue
        if moved_counts:
            if key in RATIO_KEYS:
                continue
            if abs(new - old) <= FLIPPED_FLOAT_ABS_TOLERANCE:
                continue
        problems.append("%s: %r -> %r" % (path, old, new))
    return {"problems": problems, "moved_counts": moved_counts}


def two_rungs():
    rungs = []
    for name in (ladder.BASELINE_CLASSIFIER, ladder.HEADLINE_MODEL):
        rungs.append(ladder.rung_by_name(ladder.CLASSIFICATION_RUNGS, name))
    return rungs


def pooled_buy_list_probs(frame, manifest, cv_seeds):
    """Pooled out-of-fold P(liquid) of the model and velocity rungs.

    Same call as the frozen run (ladder.cross_validate_ladder) restricted
    to the two rungs, so every fold and fit is the frozen one.
    """
    raw = ladder.cross_validate_ladder(frame, manifest["feature_cols"],
                                       manifest["baseline_cols"], cv_seeds,
                                       classification_rungs=two_rungs(),
                                       regression_rungs=[])
    model_probs = ladder.pooled(raw["oof_class"][ladder.HEADLINE_MODEL])
    velocity_probs = ladder.pooled(raw["oof_class"][ladder.BASELINE_CLASSIFIER])
    return raw["y"], raw["groups"], model_probs, velocity_probs


def run(n_boot=ladder.B_BOOT, cv_seeds=ladder.FROZEN_CV_SEEDS, verbose=True, probs_out=None):
    """Both planted signals of the illustrative preset (the frozen run's).

    probs_out, if a dict, receives {signal key: (y, groups, model_probs,
    velocity_probs)} so tests can check the aggregates against the
    functions without a second recompute; those per-card arrays are never
    written to disk.
    """
    signals = {}
    for cluster_signal in ladder.SIGNALS:
        started = time.perf_counter()
        frame, manifest, truth = synth.illustrative_preset(cluster_signal,
                                                           seed=synth.ILLUSTRATIVE_SEED)
        y, groups, model_probs, velocity_probs = pooled_buy_list_probs(frame, manifest,
                                                                       cv_seeds)
        fit_seconds = time.perf_counter() - started
        if probs_out is not None:
            probs_out[ladder.signal_key(cluster_signal)] = (y, groups, model_probs,
                                                             velocity_probs)
        block = signal_block(y, groups, model_probs, velocity_probs, n_boot, ladder.BOOT_SEED)
        block["dataset"] = ladder.dataset_block(frame, truth)
        block["seconds"] = {"fits": fit_seconds,
                            "total": time.perf_counter() - started}
        if verbose:
            print("[buylist] signal %.1f: %.1f s" % (cluster_signal,
                                                     block["seconds"]["total"]))
        signals[ladder.signal_key(cluster_signal)] = block
    return {"label": LABEL, "models": list(MODEL_NAMES),
            "cv_seeds": list(cv_seeds), "n_folds": ladder.N_FOLDS, "n_boot": n_boot,
            "boot_seed": ladder.BOOT_SEED, "data_seed": synth.ILLUSTRATIVE_SEED,
            "grid": {"start": float(THRESHOLD_GRID[0]), "stop": float(THRESHOLD_GRID[-1]),
                     "n_points": int(len(THRESHOLD_GRID))},
            "signals": signals}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Buy-list threshold grid on the synthetic preset (heavy; not CI).")
    parser.add_argument("stage", choices=["build"])
    parser.parse_args(argv)
    out_dir = ladder.results_dir()
    os.makedirs(out_dir, exist_ok=True)
    payload = run()
    print("wrote", ladder.write_json(payload, os.path.join(out_dir, GRID_FILENAME)))


if __name__ == "__main__":
    main()
