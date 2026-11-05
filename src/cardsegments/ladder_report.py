"""Render the synthetic model-ladder results into README, results/LADDER.md,
DECISIONS.md and the docs/METHOD.md assumptions table.

Every number printed here is read from results/ladder_frozen.json (written by
`python -m cardsegments.ladder frozen`), results/buylist_grid.json (pooled
calibration and decision cost; `python -m cardsegments.buylist build`) and
results/leak_detection.json (the band's measured detection limit; `python -m
cardsegments.ladder leakdetect`); this module only formats. The
generated text sits between marker comments, and `--check` (run by
tests/test_readme_drift.py in CI) fails when a committed file differs from
what the JSON renders to. "Why this metric" text lives in WHY_THIS_METRIC,
the one copy the README, LADDER.md and the page share.

    python -m cardsegments.ladder_report build     # rewrite the blocks
    python -m cardsegments.ladder_report --check   # exit 1 if any is stale
"""

import argparse
import json
import os
import sys

from cardsegments import buylist, ladder, report, synth
from cardsegments.supervised import SHIP_MIN_DELTA

FROZEN_PATH_REL = "results/" + ladder.FROZEN_FILENAME
GRID_PATH_REL = buylist.GRID_PATH_REL
LEAK_PATH_REL = "results/" + ladder.LEAK_FILENAME
SIGNAL_LABELS = {"0.0": "signal 0 (no planted segment effect)",
                 "0.9": "signal 0.9 (planted segment effect)"}
SIGNAL_SHORT = {"0.0": "signal 0", "0.9": "signal 0.9"}

WHY_THIS_METRIC = {
    "parked_rate": {
        "question": "Of the cards the buy list flags at t*, what share sits unsold and "
                    "ties up cash, and does the model's list beat the free trailing-sales "
                    "list?",
        "kind": "Counted: FP / flagged on pooled out-of-fold predictions of SYNTHETIC "
                "cards; truth is the generator's drawn sold count (>= 5 sales = liquid).",
        "lies": "It would lie if folds leaked or t* were tuned on test data; t* comes "
                "from costs only, and the shuffled-label band, group-disjoint folds and "
                "oracle-bound tests guard the folds (tests/test_ladder.py).",
    },
    "classification_ladder": {
        "question": "Does a more complex model rank cards by 'will it sell' better than "
                    "a logistic on trailing sales?",
        "kind": "Modelled: out-of-fold ROC-AUC and Brier on SYNTHETIC cards, 5 folds x "
                "10 seeds on identical grouped folds; truth is the planted P(liquid).",
        "lies": "Leakage would inflate every rung; the shuffled-label band and the "
                "'no rung beats the oracle' check would then fail (tests/test_ladder.py).",
    },
    "count_ladder": {
        "question": "Which kind of regression best predicts how many times a card sells?",
        "kind": "Modelled: out-of-fold MSE / MAE / R^2 on log1p(sold count) plus Poisson "
                "deviance, SYNTHETIC cards; truth is the planted Poisson rate.",
        "lies": "R^2 on log1p hides count-scale errors and a constant has R^2 <= 0 by "
                "construction; the oracle row bounds every rung, and the GLM-oracle "
                "deviance gap is checked at signal 0 only, where the planted log-rate is "
                "linear in trailing sales; at signal 0.9 the GLM, linear in the features, "
                "cannot represent the planted segment offsets, so its gap is printed, not "
                "bounded (results/LADDER.md; tests/test_ladder.py).",
    },
    "buy_list_matrices": {
        "question": "If I buy every card scored at or above this threshold, how many sell, "
                    "how many sit, and how many sellers do I pass over?",
        "kind": "Counted: 2x2 confusion counts of pooled out-of-fold predictions on "
                "SYNTHETIC cards at 49 precomputed thresholds; truth is the generator's "
                "drawn sold count (>= 5 sales = liquid).",
        "lies": "It would lie if the page computed its own cells or the grid drifted from "
                "the frozen run; tests check TP+FP+FN+TN = n at every threshold, FP never "
                "rising and FN never falling as t rises, and the t* cell equal to the frozen "
                "ladder's (tests/test_buylist.py).",
    },
    "missed_sellers": {
        "question": "How many cards that went on to sell does each buy list pass over?",
        "kind": "Counted: FN and FN / (TP + FN) at the threshold, pooled out-of-fold "
                "predictions of SYNTHETIC cards; truth is the drawn sold count.",
        "lies": "A list can look safe by flagging almost nothing; parked rate and missed "
                "sellers are always shown together, and FN must never fall as the threshold "
                "rises (tests/test_buylist.py).",
    },
    "threshold_regret": {
        "question": "Does the threshold set from my costs alone come close to the best "
                    "threshold I could have picked after seeing the outcomes?",
        "kind": "Modelled cost: ((m+h) FP + m FN) / N in illustrative units at the exact t* "
                "cell vs the cheapest of the 49 grid thresholds, SYNTHETIC cards; regret = "
                "cost(t*) - min(grid best, cost(t*)).",
        "lies": "Miscalibrated probabilities would push the best threshold away from t*; "
                "regret >= 0 is tested, a shifted-logit vector must raise it, and t* comes "
                "from m and h only, never from test labels (tests/test_buylist.py).",
    },
    "leakage": {
        "question": "Could the ladder's numbers come from leakage instead of signal?",
        "kind": "Simulated control: the same folds with labels permuted at row level; a "
                "clean pipeline scores AUC near 0.5 and R^2 near or below 0.",
        "lies": "A leak weaker than the band's measured detection limit would pass (the "
                "band uses a single-fold sd for a 50-cell fold mean, so it is wide); the "
                "band was written down from trial seeds before the frozen run, a planted "
                "label leak must fall outside it, and a moderate-leak control records the "
                "weakest leak it catches (tests/test_ladder.py).",
    },
    "calibration": {
        "question": "When the model says 0.70, do about 7 in 10 of those cards sell, so "
                    "that a threshold set from costs means what it says?",
        "kind": "Modelled: Brier = mean (p - y)^2, skill = 1 - Brier / (p_bar (1 - p_bar)) "
                "and ECE over quantile bins with n >= 50, on pooled out-of-fold "
                "probabilities of SYNTHETIC cards; truth is the drawn sold count.",
        "lies": "Errors of opposite sign inside one bin cancel in ECE, and skill mixes "
                "ranking with calibration; tests check that the base-rate predictor scores "
                "skill 0 exactly, a hand-made 3-bin ECE, and that a +1.5 logit shift raises "
                "ECE (tests/test_buylist.py).",
    },
    "tradeoff_curves": {
        "question": "As I raise the buy threshold, how much parked capital do I save and "
                    "how many sellers do I give up, for each list?",
        "kind": "Counted: parked rate FP / (TP + FP) and missed-seller share FN / (TP + FN) "
                "at 49 precomputed thresholds, pooled out-of-fold predictions of SYNTHETIC "
                "cards; truth is the generator's drawn sold count.",
        "lies": "At the highest thresholds a list can hold only a handful of cards, so the "
                "parked rate jumps (this synthetic page applies no n >= 50 floor); the "
                "chart draws only the precomputed stops, never interpolated values, and "
                "FN never falling as t rises is tested (tests/test_buylist.py).",
    },
    "grid_table": {
        "question": "What exact numbers sit behind every point on the charts?",
        "kind": "Counted: list size, parked rate, missed-seller share and decision cost per "
                "card at each of the 49 thresholds for both lists, copied from "
                "results/buylist_grid.json (rates shown to 0.1 pp, cost to 4 decimals); "
                "SYNTHETIC cards.",
        "lies": "It would lie if the page computed its own cells instead of copying them; "
                "tests check that the page columns equal the grid functions and the cost "
                "column equals supervised.threshold_cost_curve (tests/test_explorer.py, "
                "tests/test_buylist.py).",
    },
}

ASSUMPTIONS = [
    {"knob": "margin m, holding cost h", "value": "%.2f, %.2f" % (ladder.DEFAULT_M,
                                                                   ladder.DEFAULT_H),
     "source": "illustrative",
     "if_wrong": "t* moves; the buy list grows or shrinks; parked rates change for both "
                 "models alike"},
    {"knob": "cost threshold t* = (m+h)/(2m+h)", "value": "2/3",
     "source": "design choice (cost formula)",
     "if_wrong": "a different cost ratio gives a different t*; nothing is tuned on test data"},
    {"knob": "liquid label", "value": ">= %d sales in the outcome window"
                                      % synth.LIQUID_MIN_SOLD,
     "source": "design choice (mirrors the real export)",
     "if_wrong": "base rate and both ladders' targets move together"},
    {"knob": "outcome window", "value": "61 days on real data; none on synthetic",
     "source": "design choice",
     "if_wrong": "synthetic counts have no calendar, so window effects are not tested"},
    {"knob": "preset size", "value": "n = {:,}, {} set groups".format(
        synth.ILLUSTRATIVE_N_ROWS, 3 * synth.ILLUSTRATIVE_GROUPS_PER_CLUSTER),
     "source": "illustrative (round values, never the private cohort)",
     "if_wrong": "CIs widen or narrow; fewer groups mean noisier group bootstraps"},
    {"knob": "planted base rate", "value": "%.2f" % synth.THRESHOLD_TARGET_BASE_RATE,
     "source": "illustrative",
     "if_wrong": "the buy list at t* changes size; parked rates shift"},
    {"knob": "planted count model", "value": "log rate = a + %.1f x sales_z + signal x "
                                             "(-1, +1, 0) by cluster"
                                             % synth.THRESHOLD_LOG_LAMBDA_SLOPE,
     "source": "design choice",
     "if_wrong": "a non-Poisson market would favour other count models than the GLM"},
    {"knob": "cluster_signal", "value": "0 and 0.9", "source": "design choice (null + planted)",
     "if_wrong": "the model's lead over velocity scales with the signal"},
    {"knob": "missing cluster feature", "value": "~30% of rows (gem rate), median-imputed",
     "source": "design choice", "if_wrong": "more missingness weakens the full models"},
    {"knob": "CV", "value": "StratifiedGroupKFold %d folds x %d seeds (trial seeds %s)"
                            % (ladder.N_FOLDS, len(ladder.FROZEN_CV_SEEDS),
                               "100-102"),
     "source": "design choice",
     "if_wrong": "fewer seeds give fewer fold cells; pooled CIs depend mainly on the set groups"},
    {"knob": "bootstrap", "value": "B = %d, group (set) level" % ladder.B_BOOT,
     "source": "design choice",
     "if_wrong": "CI endpoints carry Monte Carlo error (not measured here)"},
    {"knob": "ship bar (AUC)", "value": "+%.2f and CI excludes 0" % SHIP_MIN_DELTA,
     "source": "pre-registered (existing nested-gate rule)",
     "if_wrong": "a lower bar would call smaller lifts 'beats'"},
    {"knob": "R^2 gate", "value": "from trial-seed velocity-OLS fold sd",
     "source": "pre-registered", "if_wrong": "a lower gate would call smaller gains 'beats'"},
    {"knob": "shuffle band", "value": "0.5 +/- 3 sd of trial single-fold AUCs",
     "source": "pre-registered", "if_wrong": "a wider band hides weaker leaks"},
    {"knob": "disclosure floor", "value": "n >= %d per published cell" % report.MIN_CELL_N,
     "source": "design choice (privacy)", "if_wrong": "small cells could identify cards"},
    {"knob": "gbc / gbr", "value": "scikit-learn defaults, random_state %d" % ladder.FIT_SEED,
     "source": "published default", "if_wrong": "tuned trees might gain; not tested here"},
    {"knob": "Huber epsilon", "value": "1.35", "source": "published default",
     "if_wrong": "a different epsilon trades robustness for efficiency"},
]

README_HEADLINE = ("<!-- ladder-headline:begin (generated by python -m "
                   "cardsegments.ladder_report build; do not edit) -->",
                   "<!-- ladder-headline:end -->")
README_SECTION = ("<!-- ladder-section:begin (generated by python -m "
                  "cardsegments.ladder_report build; do not edit) -->",
                  "<!-- ladder-section:end -->")
DECISIONS_BLOCK = ("<!-- ladder-decisions:begin (generated by python -m "
                   "cardsegments.ladder_report build; do not edit) -->",
                   "<!-- ladder-decisions:end -->")
METHOD_BLOCK = ("<!-- ladder-assumptions:begin (generated by python -m "
                "cardsegments.ladder_report build; do not edit) -->",
                "<!-- ladder-assumptions:end -->")


# ---------------------------------------------------------------- formatting

def count(value):
    """Whole number with thousands separators: 12000 -> '12,000'."""
    return "{:,}".format(int(value))


def num(value, decimals=4):
    if value is None:
        return "n/a"
    return ("%." + str(decimals) + "f") % value


def signed(value, decimals=4):
    if value is None:
        return "n/a"
    return ("%+." + str(decimals) + "f") % value


def pct(value, decimals=1):
    if value is None:
        return "n/a"
    return ("%." + str(decimals) + "f%%") % (100.0 * value)


def pp(value, decimals=2):
    if value is None:
        return "n/a"
    return ("%+." + str(decimals) + "f pp") % (100.0 * value)


def ci_text(block, formatter):
    """'[low, high]' with the formatter's own decimals."""
    if block is None or block.get("ci_low") is None:
        return "n/a"
    return "[%s, %s]" % (formatter(block["ci_low"]), formatter(block["ci_high"]))


def mean_sd_text(block, decimals=4):
    if block is None or block.get("mean") is None:
        return "n/a"
    if block.get("sd") is None:
        return num(block["mean"], decimals)
    return "%s ± %s" % (num(block["mean"], decimals), num(block["sd"], decimals))


def yes_no(value):
    if value is None:
        return "—"
    if value:
        return "yes"
    return "no"


def why_box(key):
    why = WHY_THIS_METRIC[key]
    lines = ["> **Why this metric.** %s" % why["question"],
             "> *Kind of number:* %s" % why["kind"],
             "> *What would make it lie:* %s" % why["lies"]]
    return "\n".join(lines)


def find_row(rows, name):
    for row in rows:
        if row["name"] == name:
            return row
    raise KeyError(name)


def json_line(json_text, path):
    """1-based line of the key path in a pretty-printed results JSON.

    The JSON is written with indent=1 and sorted keys, so the key at depth d
    sits on a line indented by d spaces; the search walks the path in order.
    An integer 0 in the path steps into the first element of a list (one
    more indent level); other list positions are refused. Failure mode: a
    key missing from its parent but present later in the file would be
    found in the wrong object; the citation tests read the cited line back.
    """
    lines = json_text.split("\n")
    start = 0
    found_line = None
    depth = 1
    for key in path:
        if isinstance(key, int):
            if key != 0:
                raise ValueError("json_line cites only the first list element")
            depth += 1
            continue
        needle = " " * depth + json.dumps(key) + ":"
        found_line = None
        n_lines = len(lines)
        position = start
        while position < n_lines:
            if lines[position].startswith(needle):
                found_line = position
                break
            position += 1
        if found_line is None:
            raise KeyError("/".join(path))
        start = found_line + 1
        depth += 1
    return found_line + 1


def cite(json_text, path, rel_path=FROZEN_PATH_REL):
    return "`%s:%d`" % (rel_path, json_line(json_text, path))


# ---------------------------------------------------------------- answers

def headline_sentence(frozen):
    """The README's bold headline: parked rate, model vs velocity, at 0.9,
    with the signal-0 null and delta AUC beside it. Wording picked by rule
    from the CI signs, never typed by hand."""
    planted = frozen["signals"]["0.9"]["kpi"]
    null = frozen["signals"]["0.0"]["kpi"]
    planted_delta = planted["delta_parked"]
    null_delta = null["delta_parked"]
    if planted_delta["ci_low"] > 0:
        planted_clause = "parks less capital"
    elif planted_delta["ci_high"] < 0:
        planted_clause = "parks MORE capital"
    else:
        planted_clause = "is not shown to park less capital"
    if null_delta["ci_low"] <= 0 <= null_delta["ci_high"]:
        null_clause = "with no planted segment effect the gap vanishes"
    elif null_delta["ci_low"] > 0:
        null_clause = "even with no planted segment effect the model's list parks less"
    else:
        null_clause = "with no planted segment effect the model's list parks more"
    sentence = ("**SYNTHETIC (illustrative preset): at the cost-set threshold t* = 2/3 the "
                "full logistic's buy list %s than the trailing-sales logistic's — "
                "%s vs %s of flagged cards sat unsold (Δ %s, 95%% CI %s; ΔAUC %s %s) "
                "when segments carry planted signal; %s (Δ %s, CI %s; ΔAUC %s %s).**"
                % (planted_clause, pct(planted["model_parked"]),
                   pct(planted["velocity_parked"]), pp(planted_delta["value"]),
                   ci_text(planted_delta, pp_short), signed(planted["delta_auc"]["value"]),
                   ci_text(planted["delta_auc"], signed),
                   null_clause, pp(null_delta["value"]), ci_text(null_delta, pp_short),
                   signed(null["delta_auc"]["value"]), ci_text(null["delta_auc"], signed)))
    return sentence


def pp_short(value, decimals=2):
    return ("%+." + str(decimals) + "f") % (100.0 * value)


def classification_answer(rows, key):
    """Which classifier is better, and why? Chosen by rule from the rows."""
    best = None
    for row in rows:
        if row["constant"] or row["name"] == "oracle":
            continue
        if best is None or row["auc_folds"]["mean"] > best["auc_folds"]["mean"]:
            best = row
    oracle = find_row(rows, "oracle")
    winners = []
    for row in rows:
        if row["beats_velocity"] is True:
            winners.append(row["name"])
    gap_to_oracle = oracle["auc_folds"]["mean"] - best["auc_folds"]["mean"]
    if len(winners) == 0:
        verdict = ("No rung beats the velocity logistic under the rule (CI excludes 0 AND "
                   "ΔAUC ≥ +%.2f)" % SHIP_MIN_DELTA)
        if key == "0.0":
            verdict += ("; in this generator at signal 0 the planted P(liquid) depends on "
                        "trailing sales alone, so velocity already ranks cards in the "
                        "oracle's order")
    else:
        names = ", ".join(winners)
        verdict = ("%s beat%s the velocity logistic under the rule (CI excludes 0 AND "
                   "ΔAUC ≥ +%.2f): the extra features proxy the planted segments that "
                   "trailing sales cannot see" % (names, "" if len(winners) > 1 else "s",
                                                  SHIP_MIN_DELTA))
    margin_text = simpler_rung_margin_text(rows, best)
    return ("**Answer (%s).** Highest fold-mean AUC: %s (%s). %s.%s The best rung sits %s "
            "AUC below the oracle (%s)."
            % (SIGNAL_SHORT[key], best["name"], num(best["auc_folds"]["mean"]), verdict,
               margin_text, num(gap_to_oracle), num(oracle["auc_folds"]["mean"])))


def simpler_rung_margin_text(rows, best):
    """' X leads Y by +d ..., so the simpler model is kept.' or ''.

    Rule: Y is the rung just below the top rung in ladder order (skipping
    constant rungs and the oracle). When the top rung's fold-mean AUC lead
    over Y is under the +0.01 bar, the extra complexity is not worth it and
    the sentence says so; otherwise (or when there is no simpler rung) the
    answer gets no sentence. Failure mode: the margin is a fold-mean
    difference, not a paired CI, so it decides only 'simpler is kept', never
    'the complex rung beats'.
    """
    simpler = None
    for row in rows:
        if row["name"] == best["name"]:
            break
        if row["constant"] or row["name"] == "oracle":
            continue
        simpler = row
    if simpler is None:
        return ""
    margin = best["auc_folds"]["mean"] - simpler["auc_folds"]["mean"]
    if margin >= SHIP_MIN_DELTA:
        return ""
    return (" %s leads %s by %s fold-mean AUC, under the +%.2f bar, so the simpler "
            "model is kept." % (best["name"], simpler["name"], signed(margin),
                                SHIP_MIN_DELTA))


def regression_answer(rows, key, gate):
    best = None
    for row in rows:
        if row["constant"] or row["name"] == "oracle":
            continue
        if best is None or row["r2_folds"]["mean"] > best["r2_folds"]["mean"]:
            best = row
    oracle = find_row(rows, "oracle")
    ols = find_row(rows, "ols")
    glm = find_row(rows, "poisson_glm")
    winners = []
    for row in rows:
        if row["beats_velocity"] is True:
            winners.append(row["name"])
    if len(winners) == 0:
        verdict = "No rung PASSES the R² gate (ΔR² CI excludes 0 AND ΔR² ≥ %s)" % num(gate, 3)
    else:
        verdict = ("PASSES the R² gate (ΔR² CI excludes 0 AND ΔR² ≥ %s): %s"
                   % (num(gate, 3), ", ".join(winners)))
    family_gap = glm["r2_folds"]["mean"] - ols["r2_folds"]["mean"]
    if family_gap > 0:
        family = ("On the same features the Poisson GLM beats OLS by %s fold-mean R²: the "
                  "count is Poisson with a log link, which a straight line on log1p "
                  "misses" % signed(family_gap))
    else:
        family = ("On the same features OLS matches or beats the Poisson GLM (%s fold-mean "
                  "R²)" % signed(family_gap))
    return ("**Answer (%s).** Highest fold-mean R²: %s (%s). %s. %s. Oracle R² %s."
            % (SIGNAL_SHORT[key], best["name"], num(best["r2_folds"]["mean"]), verdict,
               family, num(oracle["r2_folds"]["mean"])))


# ---------------------------------------------------------------- tables

def classification_table(rows):
    lines = ["| model | AUC mean ± sd (50 folds) | pooled AUC [95% CI] | ΔAUC vs velocity "
             "[95% CI] | beats? | Brier mean ± sd | fit seconds |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["pooled_auc"] is None:
            pooled_text = "excluded (constant)"
        elif row["name"] == "oracle":
            pooled_text = num(row["pooled_auc"]["value"])
        else:
            pooled_text = "%s %s" % (num(row["pooled_auc"]["value"]),
                                     ci_text(row["pooled_auc"], num))
        delta = row["delta_auc_vs_velocity"]
        if delta is None:
            delta_text = "—"
        else:
            delta_text = "%s %s" % (signed(delta["value"]), ci_text(delta, signed))
        if row["fit_seconds"] is None:
            seconds_text = "—"
        else:
            seconds_text = num(row["fit_seconds"], 1)
        lines.append("| %s | %s | %s | %s | %s | %s | %s |"
                     % (row["label"], mean_sd_text(row["auc_folds"]), pooled_text,
                        delta_text, yes_no(row["beats_velocity"]),
                        mean_sd_text(row["brier_folds"]), seconds_text))
    return "\n".join(lines)


def regression_table(rows):
    lines = ["| model | MSE mean ± sd | MAE mean ± sd | R² mean ± sd | pooled R² | ΔR² vs "
             "velocity OLS [95% CI] | passes gate? | Poisson deviance | fit seconds |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        delta = row["delta_r2_vs_velocity"]
        if delta is None:
            delta_text = "—"
        else:
            delta_text = "%s %s" % (signed(delta["value"]), ci_text(delta, signed))
        if row["deviance_folds"] is None:
            deviance_text = "n/a (log1p target)"
        else:
            deviance_text = mean_sd_text(row["deviance_folds"])
        if row["fit_seconds"] is None:
            seconds_text = "—"
        else:
            seconds_text = num(row["fit_seconds"], 1)
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (row["label"], mean_sd_text(row["mse_folds"]),
                        mean_sd_text(row["mae_folds"]), mean_sd_text(row["r2_folds"]),
                        num(row["pooled_r2"]), delta_text, yes_no(row["beats_velocity"]),
                        deviance_text, seconds_text))
    return "\n".join(lines)


def buy_list_table(kpi):
    lines = ["| buy list at t* = 2/3 | flagged | sat unsold (FP) | sold (TP) | parked rate | "
             "missed sellers (FN) |", "|---|---|---|---|---|---|"]
    entries = [("full logistic, L2 C=1", kpi["model_confusion"]),
               ("velocity logistic", kpi["velocity_confusion"])]
    if "oracle_confusion" in kpi:
        entries.append(("ORACLE (planted P(liquid))", kpi["oracle_confusion"]))
    for label, confusion in entries:
        flagged = confusion["tp"] + confusion["fp"]
        if flagged > 0:
            rate = confusion["fp"] / flagged
        else:
            rate = None
        lines.append("| %s | %d | %d | %d | %s | %d |"
                     % (label, flagged, confusion["fp"], confusion["tp"], pct(rate),
                        confusion["fn"]))
    lines.append("| buy everything (context only) | %d | %d | %d | %s | 0 |"
                 % (kpi["model_confusion"]["n"],
                    kpi["model_confusion"]["fp"] + kpi["model_confusion"]["tn"],
                    kpi["model_confusion"]["tp"] + kpi["model_confusion"]["fn"],
                    pct(kpi["buy_everything_parked"])))
    return "\n".join(lines)


def leakage_table(frozen):
    band = frozen["band"]
    lines = ["| signal | rung | shuffled AUC mean ± sd | inside band [%s, %s]? | shuffled "
             "R² mean ± sd (≤ %s?) |" % (num(band["auc_low"], 3), num(band["auc_high"], 3),
                                          num(band["r2_high"], 3)),
             "|---|---|---|---|---|"]
    keys = sorted(frozen["signals"].keys())
    pairs = [("velocity", "velocity_ols"), ("logistic_unpenalized", "ols"),
             ("logistic_l2_c1", "huber"), ("gbc", "poisson_glm"), (None, "gbr")]
    for key in keys:
        leakage = frozen["signals"][key]["leakage"]
        for class_name, reg_name in pairs:
            if class_name is None:
                class_text = "—"
                inside_text = "—"
                rung_text = reg_name
            else:
                block = leakage["auc_by_rung"][class_name]
                class_text = mean_sd_text(block)
                inside = band["auc_low"] <= block["mean"] <= band["auc_high"]
                inside_text = yes_no(inside)
                rung_text = "%s / %s" % (class_name, reg_name)
            reg_block = leakage["r2_by_rung"][reg_name]
            reg_ok = reg_block["mean"] <= band["r2_high"]
            lines.append("| %s | %s | %s | %s | %s (%s) |"
                         % (key, rung_text, class_text, inside_text,
                            mean_sd_text(reg_block), yes_no(reg_ok)))
    return "\n".join(lines)


def verdict_table(frozen):
    lines = ["| id | prediction | verdict | deciding statistic |", "|---|---|---|---|"]
    for row in frozen["verdicts"]:
        lines.append("| %s | %s | %s | %s |" % (row["id"], row["kind"], row["verdict"],
                                                 row["deciding_statistic"]))
    return "\n".join(lines)


def per_signal_cite(json_text, prefix, suffix, rel_path=FROZEN_PATH_REL):
    """'0.9: `file:line`; 0: `file:line`' for the same key path in both signals."""
    planted = cite(json_text, prefix + ["0.9"] + suffix, rel_path)
    null = cite(json_text, prefix + ["0.0"] + suffix, rel_path)
    return "0.9 %s; 0 %s" % (planted, null)


def calibration_text(frozen, grid, key):
    """'model Brier b (fold mean), skill s, ECE e vs velocity ...' for one signal."""
    parts = []
    names = (ladder.HEADLINE_MODEL, ladder.BASELINE_CLASSIFIER)
    for name in names:
        fold_brier = find_row(frozen["signals"][key]["classification"], name)["brier_folds"]
        pooled = grid["signals"][key]["models"][name]["calibration"]
        parts.append("%s Brier %s (fold mean), skill %s, ECE %s"
                     % (KPI_MODEL_SHORT[name], num(fold_brier["mean"]),
                        num(pooled["brier_skill"], 3), num(pooled["ece"], 3)))
    return "; ".join(parts)


def regret_text(grid, key):
    """cost(t*) per list vs 'buy none', and regret share, at t* = 2/3."""
    setting = grid["signals"][key]["cost_settings"][0]
    parts = []
    names = (ladder.HEADLINE_MODEL, ladder.BASELINE_CLASSIFIER)
    for name in names:
        block = setting["models"][name]
        parts.append("%s cost(t*) %s, regret %s" % (KPI_MODEL_SHORT[name],
                                                   num(block["cost_t_star"], 5),
                                                   pct(block["regret_share"], 2)))
    buy_none = setting["models"][ladder.HEADLINE_MODEL]["buy_none_cost"]
    return "%s; 'buy none' %s" % ("; ".join(parts), num(buy_none, 5))


KPI_MODEL_SHORT = {ladder.HEADLINE_MODEL: "model", ladder.BASELINE_CLASSIFIER: "velocity"}


def kpi_card_table(frozen, json_text, grid, grid_text):
    planted = frozen["signals"]["0.9"]
    null = frozen["signals"]["0.0"]
    planted_reg_best = best_regression_row(planted["regression"])
    null_reg_best = best_regression_row(null["regression"])
    lines = ["| KPI | formula | why it matters | compared against | what the software "
             "does to it | measured on | value now |", "|---|---|---|---|---|---|---|"]
    lines.append(
        "| **Parked-capital rate at t\\*** (headline) | FP/(TP+FP) among cards with pooled "
        "out-of-fold P(liquid) ≥ t* = (m+h)/(2m+h) = 2/3; Δ = parked(velocity) − "
        "parked(model), paired group bootstrap | A flagged card that sits ties up cash "
        "for the outcome window | velocity logistic, same folds, same t* ('buy everything' "
        "is context only) | screens buys at a threshold set from costs | synthetic truth "
        "| signal 0.9: model %s vs velocity %s, Δ %s %s (%s); signal 0: model %s vs "
        "velocity %s, Δ %s %s (%s) |"
        % (pct(planted["kpi"]["model_parked"]), pct(planted["kpi"]["velocity_parked"]),
           pp(planted["kpi"]["delta_parked"]["value"]),
           ci_text(planted["kpi"]["delta_parked"], pp_short),
           cite(json_text, ["signals", "0.9", "kpi", "delta_parked"]),
           pct(null["kpi"]["model_parked"]), pct(null["kpi"]["velocity_parked"]),
           pp(null["kpi"]["delta_parked"]["value"]),
           ci_text(null["kpi"]["delta_parked"], pp_short),
           cite(json_text, ["signals", "0.0", "kpi", "delta_parked"])))
    lines.append(
        "| Lift over velocity | ΔAUC (pooled OOF), paired group bootstrap; 'beats' = CI "
        "excludes 0 AND Δ ≥ +0.01 | what the extra features buy beyond trailing sales | "
        "velocity logistic | measures the lift and its CI | synthetic truth | signal 0.9: "
        "%s %s, beats: %s; signal 0: %s %s, beats: %s (%s) |"
        % (signed(planted["kpi"]["delta_auc"]["value"]),
           ci_text(planted["kpi"]["delta_auc"], signed),
           yes_no(planted["kpi"]["auc_beats_velocity"]),
           signed(null["kpi"]["delta_auc"]["value"]), ci_text(null["kpi"]["delta_auc"], signed),
           yes_no(null["kpi"]["auc_beats_velocity"]),
           per_signal_cite(json_text, ["signals"], ["kpi", "delta_auc"])))
    lines.append(
        "| Decision cost and threshold regret | cost(t) = ((m+h)·FP + m·FN)/N at m = h = "
        "0.10; regret = cost(t*) − min(grid minimum, cost(t*)), as a share of cost(t*) | "
        "near-0 regret means a threshold set from the cost ratio works without peeking at "
        "outcomes | 'buy none'; the hindsight-best of 49 grid thresholds | lowers cost per "
        "card | synthetic truth (illustrative cost units) | signal 0.9: %s; signal 0: %s "
        "(%s) |"
        % (regret_text(grid, "0.9"), regret_text(grid, "0.0"),
           per_signal_cite(grid_text, ["signals"], ["cost_settings", 0, "models",
                                                    ladder.HEADLINE_MODEL, "cost_t_star"],
                           GRID_PATH_REL)))
    lines.append(
        "| Calibration | Brier = mean (p − y)² (fold mean); skill = 1 − Brier/(p̄(1 − p̄)) "
        "and ECE over quantile bins with n ≥ %d (pooled out-of-fold) | a fixed t* works "
        "only if 0.70 means 7 in 10 sell | velocity logistic; the base-rate predictor "
        "(skill 0) | shows whether the probabilities can be trusted | synthetic truth | "
        "signal 0.9: %s; signal 0: %s (Brier %s; skill, ECE %s) |"
        % (report.MIN_CELL_N, calibration_text(frozen, grid, "0.9"),
           calibration_text(frozen, grid, "0.0"),
           per_signal_cite(json_text, ["signals"], ["classification"]),
           per_signal_cite(grid_text, ["signals"], ["models", ladder.HEADLINE_MODEL,
                                                    "calibration"], GRID_PATH_REL)))
    lines.append(
        "| Count-regression gain | ΔR² on log1p(sold count), best rung vs velocity OLS; "
        "Poisson deviance vs the oracle | answers 'which type of regression' | velocity "
        "OLS; the planted Poisson rate | ranks the regression families | synthetic truth | "
        "signal 0.9: %s ΔR² %s %s; signal 0: %s ΔR² %s %s; gate %s (%s) |"
        % (planted_reg_best["name"], signed(planted_reg_best["delta_r2_vs_velocity"]["value"]),
           ci_text(planted_reg_best["delta_r2_vs_velocity"], signed),
           null_reg_best["name"], signed(null_reg_best["delta_r2_vs_velocity"]["value"]),
           ci_text(null_reg_best["delta_r2_vs_velocity"], signed),
           num(frozen["r2_gate"], 3),
           per_signal_cite(json_text, ["signals"], ["regression"])))
    lines.append("| Gate power (MDE) | smallest planted signal at which the gate ships in "
                 "≥ 80% of seeds | a null counts only if the test could find something | "
                 "the signal-0 null | turns 'found nothing' into 'could detect ≥ X' | "
                 "synthetic truth | to measure |")
    return "\n".join(lines)


def best_regression_row(rows):
    best = None
    for row in rows:
        if row["constant"] or row["name"] == "oracle" or row["delta_r2_vs_velocity"] is None:
            continue
        if best is None or row["r2_folds"]["mean"] > best["r2_folds"]["mean"]:
            best = row
    return best


def guarantees_list(frozen):
    parts = []
    keys = sorted(frozen["signals"].keys())
    identical = True
    shared = 0
    internal = 0
    band_pass = True
    oracle_violations = 0
    for key in keys:
        block = frozen["signals"][key]
        identical = identical and block["folds"]["identical_folds_all_rungs"]
        shared += block["folds"]["shared_group_pairs_outer"]
        internal += len(block["folds"]["internal_split_rungs"])
        band_pass = band_pass and block["leakage"]["band_check"]["passes"]
        oracle_violations += len(block["oracle_violations"])
    parts.append("- Every rung of both ladders scored on identical outer folds: %s; set "
                 "groups shared between train and test: %d; rungs with an internal "
                 "split: %d." % (yes_no(identical), shared, internal))
    parts.append("- Shuffled-label control inside the pre-registered band at both signals: "
                 "%s." % yes_no(band_pass))
    parts.append("- Rungs beating the planted-truth oracle by more than 0.01: %d."
                 % oracle_violations)
    parts.append("- Predictions file pinned before the frozen run (sha256 prefix "
                 "`%s`); a test fails if it changes." % frozen["predictions_sha256_prefix"])
    parts.append("- The illustrative preset's defaults leave the committed synth/ example "
                 "datasets unchanged (tests/test_synth.py).")
    return "\n".join(parts)


# ---------------------------------------------------------------- documents

def readme_headline(frozen):
    return headline_sentence(frozen)


def readme_section(frozen, json_text, grid, grid_text):
    planted = frozen["signals"]["0.9"]
    null = frozen["signals"]["0.0"]
    lines = ["## Model ladders and the buy list (SYNTHETIC, illustrative preset)", ""]
    lines.append("Two grouped k-fold model ladders on identical folds (5 folds × %d seeds), "
                 "run on the illustrative synthetic preset: n = %s cards in %d set groups, "
                 "planted base rate %s (realized %s at signal 0, %s at signal 0.9), "
                 "liquid = at least %d sales. Round illustrative values, not the private "
                 "cohort. Predictions were registered first "
                 "([`results/%s`](results/%s)); full tables, answers and the shuffled-label "
                 "control are in [`results/LADDER.md`](results/LADDER.md)."
                 % (len(frozen["cv_seeds"]), count(frozen["preset"]["n_rows"]),
                    frozen["preset"]["n_groups"], num(frozen["preset"]["target_base_rate"], 2),
                    num(null["dataset"]["base_rate"], 3),
                    num(planted["dataset"]["base_rate"], 3),
                    frozen["preset"]["liquid_min_sold"], frozen["predictions_file"],
                    frozen["predictions_file"]))
    lines.append("")
    lines.append(kpi_card_table(frozen, json_text, grid, grid_text))
    lines.append("")
    lines.append(why_box("parked_rate"))
    lines.append("")
    lines.append(why_box("calibration"))
    lines.append("")
    lines.append("**Guarantees (tests, not KPIs)**")
    lines.append("")
    lines.append(guarantees_list(frozen))
    lines.append("")
    lines.append("### Classification ladder, signal 0.9 (planted segment effect)")
    lines.append("")
    lines.append(classification_table(planted["classification"]))
    lines.append("")
    lines.append(classification_answer(planted["classification"], "0.9"))
    lines.append("")
    lines.append(classification_answer(null["classification"], "0.0"))
    lines.append("")
    lines.append(why_box("classification_ladder"))
    lines.append("")
    lines.append("### Count-regression ladder on log1p(sold count), signal 0.9")
    lines.append("")
    lines.append(regression_table(planted["regression"]))
    lines.append("")
    lines.append(regression_answer(planted["regression"], "0.9", frozen["r2_gate"]))
    lines.append("")
    lines.append(regression_answer(null["regression"], "0.0", frozen["r2_gate"]))
    lines.append("")
    lines.append(why_box("count_ladder"))
    lines.append("")
    lines.append("Pre-registered predictions: %d confirmed of %d (verdicts by code, "
                 "listed in `results/LADDER.md`)." % (count_confirmed(frozen),
                                                       len(frozen["verdicts"])))
    return "\n".join(lines)


def count_confirmed(frozen):
    confirmed = 0
    for row in frozen["verdicts"]:
        if row["verdict"] == "confirmed":
            confirmed += 1
    return confirmed


def band_limit_sentence(frozen, leak):
    """The band's measured detection limit, from results/leak_detection.json."""
    lowest, highest = ladder.honest_shuffled_auc_range(frozen)
    missed = leak["largest_missed"]
    caught = leak["smallest_caught"]
    if missed is None:
        missed_text = "every planted leak fell outside the band"
    else:
        missed_text = ("a leak scoring shuffled fold-mean AUC %s (label + N(0, %g²) noise) "
                       "still passes" % (num(missed["fold_mean_auc"]), missed["noise_sd"]))
    if caught is None:
        caught_text = "no planted leak fell outside it"
    else:
        caught_text = ("the weakest leak caught scores %s (noise sd %g)"
                       % (num(caught["fold_mean_auc"]), caught["noise_sd"]))
    return ("Detection limit of the registered band (moderate-leak control, `%s`, "
            "n = %s, %d fold cells per leak): %s, while %s; the limit lies between "
            "the two, since the band catches nothing at or below %s. Honest rungs' "
            "shuffled fold means span %s to %s across both signals, so the band (half-width "
            "%s, set from a single-fold sd) is far wider than the spread it guards. The band "
            "stays as registered; a future registration should set it from the sd of the "
            "fold MEAN across trial seeds."
            % (LEAK_PATH_REL, count(leak["n_rows"]), leak["n_cells"], missed_text,
               caught_text, num(missed["fold_mean_auc"]) if missed else "0.5",
               num(lowest), num(highest), num(frozen["band"]["auc_half_width"], 3)))


def glm_gap_sentence(block, key):
    """GLM - oracle deviance gap, and whether the 0.03 bound applies here."""
    gap = block["glm_oracle_deviance_gap"]
    tolerance = ladder.GLM_ORACLE_DEVIANCE_TOLERANCE
    if key == "0.0":
        return ("Poisson GLM minus oracle fold-mean deviance: %s (bounded at ≤ %s at this "
                "signal, where the planted log-rate is linear in trailing sales; "
                "tests/test_ladder.py)." % (num(gap), num(tolerance, 2)))
    return ("Poisson GLM minus oracle fold-mean deviance: %s, %s× the %s bound that holds "
            "at signal 0; not bounded here, because the GLM is linear in the features and "
            "cannot represent the planted segment offsets (gbr, the nonlinear rung, is fitted "
            "on log1p and has no Poisson deviance to compare)."
            % (num(gap), num(gap / tolerance, 1), num(tolerance, 2)))


def ladder_markdown(frozen, json_text, leak):
    lines = ["# Model ladders on the illustrative synthetic preset", ""]
    lines.append("Generated by `python -m cardsegments.ladder_report build` from "
                 "`%s`; do not edit by hand. Every number is SYNTHETIC: the illustrative "
                 "preset (n = %s, %d set groups, planted base rate %s, liquid = at least %d "
                 "sales), round values that are not the private cohort's."
                 % (FROZEN_PATH_REL, count(frozen["preset"]["n_rows"]),
                    frozen["preset"]["n_groups"],
                    num(frozen["preset"]["target_base_rate"], 2),
                    frozen["preset"]["liquid_min_sold"]))
    lines.append("")
    lines.append("Protocol: StratifiedGroupKFold, %d folds × %d CV seeds (%s), groups = set "
                 "ids; impute + scale refit inside every training fold; the same fold "
                 "indices for every rung of both ladders. Fold metrics are mean ± sd "
                 "(ddof = 1) over the 50 fold-seed cells; pooled metrics use per-card "
                 "out-of-fold predictions averaged over seeds, with a group-bootstrap CI "
                 "(B = %d). 'Beats' = the paired CI excludes 0 AND Δ ≥ +%s AUC (or ≥ the "
                 "R² gate %s). Fit seconds are wall-clock on the machine that ran the "
                 "frozen run (scikit-learn %s) and vary between machines."
                 % (frozen["n_folds"], len(frozen["cv_seeds"]), "0-9", frozen["n_boot"],
                    num(frozen["ship_min_delta_auc"], 2), num(frozen["r2_gate"], 3),
                    frozen["sklearn_version"]))
    lines.append("")
    lines.append("## 1. Leakage controls first")
    lines.append("")
    lines.append("Question: could these numbers come from leakage? Labels (and counts) are "
                 "permuted at row level with the grouped folds kept. The band was written "
                 "down from trial seeds before the frozen run "
                 "([`%s`](%s)). Constant rungs (majority, mean) are left out: their fold "
                 "AUC is 0.5 by definition and their pooled out-of-fold AUC is biased "
                 "(majority, real labels: %s at signal 0.9)."
                 % (frozen["predictions_file"], frozen["predictions_file"],
                    num(find_row(frozen["signals"]["0.9"]["classification"],
                                 "majority").get("pooled_auc_biased_constant"))))
    lines.append("")
    lines.append(leakage_table(frozen))
    lines.append("")
    lines.append("The Poisson GLM's shuffled R² is well below 0, not near it: with shuffled "
                 "counts it fits a near-constant mean and maps it through the Poisson "
                 "E[log1p N], but the shuffled counts are a mixture (over-dispersed), so "
                 "that mapping is biased on the log1p scale. Below 0 is what the bound "
                 "allows; only values above it would point to leakage.")
    lines.append("")
    lines.append(band_limit_sentence(frozen, leak))
    lines.append("")
    lines.append(why_box("leakage"))
    keys = ["0.9", "0.0"]
    section = 2
    for key in keys:
        block = frozen["signals"][key]
        lines.append("")
        lines.append("## %d. %s" % (section, SIGNAL_LABELS[key][0].upper()
                                     + SIGNAL_LABELS[key][1:]))
        lines.append("")
        lines.append("Dataset: n = %s, %d groups, realized base rate %s (planted %s), "
                     "share of cards with 0 sales %s."
                     % (count(block["dataset"]["n_rows"]), block["dataset"]["n_groups"],
                        num(block["dataset"]["base_rate"], 4),
                        num(block["dataset"]["planted_base_rate"], 4),
                        pct(block["dataset"]["share_zero_sold"])))
        lines.append("")
        lines.append("### Which classifier is better, and why?")
        lines.append("")
        lines.append(classification_table(block["classification"]))
        lines.append("")
        lines.append(classification_answer(block["classification"], key))
        lines.append("")
        lines.append("### The buy list at t* = 2/3")
        lines.append("")
        lines.append(buy_list_table(block["kpi"]))
        lines.append("")
        lines.append("Δ parked (velocity − model) %s, 95%% CI %s; ΔAUC (model − velocity) "
                     "%s %s." % (pp(block["kpi"]["delta_parked"]["value"]),
                                 ci_text(block["kpi"]["delta_parked"], pp_short),
                                 signed(block["kpi"]["delta_auc"]["value"]),
                                 ci_text(block["kpi"]["delta_auc"], signed)))
        lines.append("")
        lines.append("### Which regression is better, and why?")
        lines.append("")
        lines.append(regression_table(block["regression"]))
        lines.append("")
        lines.append(regression_answer(block["regression"], key, frozen["r2_gate"]))
        lines.append("")
        lines.append(glm_gap_sentence(block, key))
        section += 1
    lines.append("")
    lines.append(why_box("classification_ladder"))
    lines.append("")
    lines.append(why_box("count_ladder"))
    lines.append("")
    lines.append("## %d. Pre-registered predictions, judged by code" % section)
    lines.append("")
    lines.append(verdict_table(frozen))
    lines.append("")
    lines.append("Predictions file sha256 prefix `%s` (checked by tests/test_ladder.py)."
                 % frozen["predictions_sha256_prefix"])
    lines.append("")
    return "\n".join(lines)


def changed_verdicts_under_auc_bar(rows, gate):
    """Regression rungs whose PASS/FAIL would flip if the +0.01 AUC bar were
    used as the R^2 gate (the 'R^2 gate on its own scale' decision)."""
    flips = []
    for row in rows:
        delta = row["delta_r2_vs_velocity"]
        if delta is None:
            continue
        with_gate = ladder.beats_baseline(delta["value"], delta["ci_low"], delta["ci_high"],
                                          gate)
        with_auc_bar = ladder.beats_baseline(delta["value"], delta["ci_low"],
                                             delta["ci_high"], SHIP_MIN_DELTA)
        if with_gate != with_auc_bar:
            flips.append(row["name"])
    return flips


def decisions_block(frozen, leak):
    planted = frozen["signals"]["0.9"]
    null = frozen["signals"]["0.0"]
    gbc = find_row(planted["classification"], "gbc")
    l2 = find_row(planted["classification"], "logistic_l2_c1")
    unpen = find_row(planted["classification"], "logistic_unpenalized")
    majority = find_row(planted["classification"], "majority")
    flips = 0
    signal_keys = sorted(frozen["signals"].keys())
    for key in signal_keys:
        flips += len(changed_verdicts_under_auc_bar(frozen["signals"][key]["regression"],
                                                    frozen["r2_gate"]))
    band = frozen["band"]
    lines = []
    lines.append("### 2026-11-04 — Leakage and planted-truth checks before any ladder number")
    lines.append("Decided to run a row-level shuffled-label control on the same grouped "
                 "folds and an oracle row before reading the ladders, because the ≥ 5-sales "
                 "label sits close to trailing sales; measurement: every non-constant rung's "
                 "shuffled fold-mean AUC fell inside [%s, %s] at both signals (band check "
                 "passes: %s) and %d rung(s) beat the oracle. Alternative: a group-level "
                 "shuffle (whole sets swap labels) — rejected because unequal set sizes make "
                 "it ill-defined; the plan's row-level shuffle with grouped folds kept was "
                 "used."
                 % (num(band["auc_low"], 3), num(band["auc_high"], 3),
                    yes_no(planted["leakage"]["band_check"]["passes"]
                           and null["leakage"]["band_check"]["passes"]),
                    len(planted["oracle_violations"]) + len(null["oracle_violations"])))
    lines.append("")
    lines.append("### 2026-11-04 — Constant baselines left out of the shuffle band")
    lines.append("Decided to exclude majority and mean from the band because a per-fold "
                 "constant scores fold AUC exactly %s while its pooled out-of-fold AUC is "
                 "%s at signal 0.9 (biased by fold base rates); alternative (include them) "
                 "would test the splitter, not leakage."
                 % (num(majority["auc_folds"]["mean"]),
                    num(majority.get("pooled_auc_biased_constant"))))
    lines.append("")
    lines.append("### 2026-11-04 — Shuffle band and R² gate from trial seeds, written down first")
    lines.append("Decided band = 0.5 ± 3 × single-fold sd (%s, from %d trial cells) and R² "
                 "gate = %s (trial velocity-OLS fold sd, rounded up), registered before the "
                 "frozen run, because a band set from the frozen seeds would be fitted to the "
                 "result it judges; alternative (fixed round numbers) had no measurement "
                 "behind it." % (num(band["single_fold_auc_sd"]), band["n_auc_cells"],
                                 num(frozen["r2_gate"], 3)))
    lines.append("")
    lines.append("### 2026-11-04 — The R² gate on its own scale")
    if flips == 0:
        outcome = ("so on this run the choice is about using the right noise scale, not "
                   "about the outcome")
    else:
        outcome = "so the gate's scale decides those verdicts"
    lines.append("Decided the count ladder's gate is %s R² (the trial velocity-OLS fold "
                 "sd), not the +0.01 AUC bar, because R² and AUC have different fold noise; "
                 "measurement: reusing +0.01 would flip %d PASS/FAIL verdict(s) across the "
                 "two signals, %s." % (num(frozen["r2_gate"], 3), flips, outcome))
    lines.append("")
    lines.append("### 2026-11-04 — Reuse `gbc`; no new HGB rung")
    lines.append("Decided to keep the existing deterministic GradientBoostingClassifier "
                 "(random_state 42, defaults) as the boosting rung because at signal 0.9 its "
                 "fold-mean AUC is %s vs %s for the full logistic (%s), under the +0.01 "
                 "bar, at %s s of fitting for 50 cells; alternative (HistGradientBoosting + "
                 "threadpoolctl thread pinning) was not built: it adds a rung class and a "
                 "thread setting for a family already on the ladder."
                 % (num(gbc["auc_folds"]["mean"]), num(l2["auc_folds"]["mean"]),
                    signed(gbc["auc_folds"]["mean"] - l2["auc_folds"]["mean"]),
                    num(gbc["fit_seconds"], 1)))
    lines.append("")
    lines.append("### 2026-11-04 — `primary_logistic` relabelled as L2, C = 1")
    lines.append("Decided to label the old primary model honestly as 'full logistic, L2 "
                 "C=1' and add a truly unpenalized rung (C = inf) beside it, because "
                 "scikit-learn's default is L2 with C = 1; measurement: the two differ by "
                 "%s fold-mean AUC at signal 0.9, so the label was wrong but the number "
                 "barely moves. The bundle key `primary_logistic` is kept so the committed "
                 "aggregates bundle still renders; alternative (rename the key) would break "
                 "that bundle until a gated rerun."
                 % signed(l2["auc_folds"]["mean"] - unpen["auc_folds"]["mean"]))
    lines.append("")
    lines.append("### 2026-11-04 — Velocity, not 'buy everything', as the comparator")
    lines.append("Decided the headline compares against the velocity logistic because "
                 "'buy everything' parks %s of cards at signal 0.9 vs %s for velocity, a "
                 "straw man by %s; alternative kept only as a context row."
                 % (pct(planted["kpi"]["buy_everything_parked"]),
                    pct(planted["kpi"]["velocity_parked"]),
                    pp(planted["kpi"]["buy_everything_parked"]
                       - planted["kpi"]["velocity_parked"])))
    lines.append("")
    lines.append("### 2026-11-04 — Illustrative preset values; show the null and the planted case")
    lines.append("Decided on round illustrative values (n = %s, %d groups, planted base "
                 "rate %s; realized %s and %s) and to show signal 0 beside 0.9, because the "
                 "private cohort's n and base rate are licensed-feed aggregates and a demo "
                 "with only the planted case would hide that the model's lead depends on it; "
                 "measurement: ΔAUC %s %s at signal 0.9 vs %s %s at signal 0. Alternative "
                 "(size the preset to the private cohort) rejected: it would publish those "
                 "licensed-feed aggregates."
                 % (count(frozen["preset"]["n_rows"]), frozen["preset"]["n_groups"],
                    num(frozen["preset"]["target_base_rate"], 2),
                    num(null["dataset"]["base_rate"], 3),
                    num(planted["dataset"]["base_rate"], 3),
                    signed(planted["kpi"]["delta_auc"]["value"]),
                    ci_text(planted["kpi"]["delta_auc"], signed),
                    signed(null["kpi"]["delta_auc"]["value"]),
                    ci_text(null["kpi"]["delta_auc"], signed)))
    tolerance = ladder.GLM_ORACLE_DEVIANCE_TOLERANCE
    planted_gap = planted["glm_oracle_deviance_gap"]
    null_gap = null["glm_oracle_deviance_gap"]
    lines.append("")
    lines.append("### 2026-11-05 — The GLM-oracle deviance bound holds at signal 0 only")
    lines.append("Decided to keep the 'Poisson GLM deviance within %s of the oracle' check at "
                 "signal 0 only, because measurement: the gap is %s at signal 0 (the planted "
                 "log-rate is linear in trailing sales) but %s at signal 0.9, %s× the bound, "
                 "since the GLM is linear in the features and cannot represent the planted "
                 "segment offsets. Alternative (bound both signals) would fail at 0.9 by %s and "
                 "test the GLM's functional form, not leakage; the 0.9 gap is printed in "
                 "results/LADDER.md instead."
                 % (num(tolerance, 2), num(null_gap), num(planted_gap),
                    num(planted_gap / tolerance, 1), num(planted_gap - tolerance)))
    lowest, highest = ladder.honest_shuffled_auc_range(frozen)
    missed = leak["largest_missed"]
    caught = leak["smallest_caught"]
    lines.append("")
    lines.append("### 2026-11-05 — The leakage band kept as registered; its detection limit "
                 "measured")
    lines.append("Decided not to retune the pre-registered band (half-width %s) after seeing "
                 "that it is wide, because a band changed after the frozen run would be fitted "
                 "to the result it judges; measurement: honest rungs' shuffled fold means span "
                 "%s to %s, and a moderate-leak control (label + noise as a feature, n = %s) "
                 "passes at shuffled fold-mean AUC %s and is caught at %s, so the band misses "
                 "leaks up to about %s above 0.5. Alternative (a band from the sd of the fold "
                 "MEAN across trial seeds) is the rule for any future registration."
                 % (num(band["auc_half_width"], 3), num(lowest), num(highest),
                    count(leak["n_rows"]), num(missed["fold_mean_auc"]),
                    num(caught["fold_mean_auc"]), num(missed["fold_mean_auc"] - 0.5, 3)))
    return "\n".join(lines)


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


def load_results(rel_path, root=None):
    """(parsed JSON, raw text) of a committed results file."""
    if root is None:
        root = repo_root()
    path = os.path.join(root, rel_path)
    with open(path) as handle:
        json_text = handle.read()
    return json.loads(json_text), json_text


def load_frozen(root=None):
    return load_results(FROZEN_PATH_REL, root)


def replace_block(text, markers, body):
    begin, end = markers
    start = text.index(begin)
    stop = text.index(end)
    return text[:start + len(begin)] + "\n" + body + "\n" + text[stop:]


def extract_block(text, markers):
    begin, end = markers
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    return text[start:stop].strip("\n")


def expected_outputs(root=None):
    """{relative path: full expected text} for every generated file."""
    if root is None:
        root = repo_root()
    frozen, json_text = load_frozen(root)
    grid, grid_text = load_results(GRID_PATH_REL, root)
    leak, _leak_text = load_results(LEAK_PATH_REL, root)
    outputs = {}
    outputs["results/LADDER.md"] = ladder_markdown(frozen, json_text, leak)
    readme_path = os.path.join(root, "README.md")
    with open(readme_path) as handle:
        readme = handle.read()
    readme = replace_block(readme, README_HEADLINE, readme_headline(frozen))
    readme = replace_block(readme, README_SECTION, readme_section(frozen, json_text, grid,
                                                                     grid_text))
    outputs["README.md"] = readme
    decisions_path = os.path.join(root, "DECISIONS.md")
    with open(decisions_path) as handle:
        decisions = handle.read()
    outputs["DECISIONS.md"] = replace_block(decisions, DECISIONS_BLOCK,
                                            decisions_block(frozen, leak))
    method_path = os.path.join(root, "docs", "METHOD.md")
    with open(method_path) as handle:
        method = handle.read()
    outputs["docs/METHOD.md"] = replace_block(method, METHOD_BLOCK, assumptions_table())
    return outputs


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


def build(root=None):
    if root is None:
        root = repo_root()
    outputs = expected_outputs(root)
    paths = sorted(outputs.keys())
    for rel_path in paths:
        full_path = os.path.join(root, rel_path)
        with open(full_path, "w") as handle:
            handle.write(outputs[rel_path])
        print("wrote", rel_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the synthetic ladder report blocks.")
    parser.add_argument("stage", nargs="?", default="build", choices=["build"])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if a generated file is stale")
    args = parser.parse_args(argv)
    if args.check:
        stale = stale_files()
        if stale:
            print("stale:", ", ".join(stale))
            sys.exit(1)
        print("ladder report blocks are up to date")
        return
    build()


if __name__ == "__main__":
    main()
