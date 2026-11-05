"""Pre-registered predictions for the synthetic model ladders, and the code
that judges them on the frozen run.

The predictions file is WRITTEN BY CODE (`python -m cardsegments.ladder
preregister`) from results/ladder_trial.json, before the frozen run, and the
frozen JSON pins a sha256 prefix of it; tests/test_ladder.py fails if the
file changes afterwards or no longer matches what this module writes from the
committed trial JSON. Each prediction's verdict is decided by a judge
function below, never by hand: a loss or a null ships as printed.
"""

from cardsegments import synth
from cardsegments.supervised import SHIP_MIN_DELTA

GLM_GAP_LIMIT = 0.03
COMPLEXITY_BAR = 0.01

PREDICTIONS = [
    {"id": "P1", "kind": "leakage control",
     "text": "Under row-level shuffled labels (grouped folds kept), every non-constant "
             "rung's fold-mean AUC lies inside the band and its fold-mean R^2 is at or "
             "below the bound, at both signals."},
    {"id": "P2", "kind": "fold integrity",
     "text": "Every rung of both ladders is scored on identical outer folds, no set group "
             "sits in both the train and test side of any fold, and no rung makes an "
             "internal validation split, at both signals."},
    {"id": "P3", "kind": "oracle bound",
     "text": "No rung beats the planted truth: fold-mean AUC and R^2 never exceed the "
             "oracle's by more than 0.01 and no count model's deviance falls more than 0.01 "
             "below it, at both signals; at signal 0 (where the planted log-rate is linear "
             "in trailing sales) the Poisson GLM's fold-mean deviance is within 0.03 of "
             "the oracle's."},
    {"id": "P4", "kind": "null (signal 0)",
     "text": "At cluster_signal 0, no classification rung beats the velocity logistic "
             "under the rule (paired CI excludes 0 AND delta AUC >= +0.01)."},
    {"id": "P5", "kind": "planted effect (signal 0.9)",
     "text": "At cluster_signal 0.9, the full logistic (L2, C=1) beats the velocity "
             "logistic under the same rule."},
    {"id": "P6", "kind": "headline KPI, null (signal 0)",
     "text": "At cluster_signal 0, the paired CI of parked(velocity) - parked(model) at "
             "t* = 2/3 includes 0 (descriptive; no gate)."},
    {"id": "P7", "kind": "headline KPI, planted (signal 0.9)",
     "text": "At cluster_signal 0.9, the paired CI of parked(velocity) - parked(model) at "
             "t* lies above 0: the model's buy list parks less capital (descriptive)."},
    {"id": "P8", "kind": "regression family",
     "text": "At both signals the Poisson GLM beats velocity OLS on log1p(sold count) "
             "under the R^2 gate, and it has a higher fold-mean R^2 than OLS on the same "
             "features (the planted count is Poisson with a log link, so a linear fit to "
             "log1p misses its curvature)."},
    {"id": "P9", "kind": "regression null (signal 0)",
     "text": "At cluster_signal 0, OLS on the full features does not beat velocity OLS "
             "under the R^2 gate."},
    {"id": "P10", "kind": "complexity",
     "text": "At cluster_signal 0.9, no rung's fold-mean AUC exceeds the full logistic "
             "(L2, C=1)'s by 0.01 or more: boosting buys less than the +0.01 bar."},
]


def fmt(value, decimals=4):
    if value is None:
        return "n/a"
    return ("%." + str(decimals) + "f") % value


def predictions_markdown(trial):
    """The predictions file text, built only from the trial JSON."""
    band = trial["band"]
    lines = ["# Pre-registered predictions: synthetic model ladders", ""]
    lines.append("Registered %s, before the frozen run. Written by "
                 "`python -m cardsegments.ladder preregister` from "
                 "`results/ladder_trial.json`; the frozen results pin a sha256 prefix "
                 "of this file and a test fails if it changes." % _registration_date())
    lines.append("")
    lines.append("All data here is SYNTHETIC (the illustrative preset: round values, "
                 "never the private cohort). The trial run used a different data draw "
                 "(seed %d) and CV seeds %s; the frozen run uses the preset's data seed "
                 "(%d) and CV seeds 0-9."
                 % (trial["data_seed"], _seed_text(trial["cv_seeds"]),
                    synth.ILLUSTRATIVE_SEED))
    lines.append("")
    lines.append("## Frozen protocol")
    lines.append("")
    lines.append("- StratifiedGroupKFold, 5 folds x 10 seeds, groups = set ids; impute + "
                 "scale refit in every training fold; the same folds for every rung of "
                 "both ladders.")
    lines.append("- Classification: majority, velocity logistic, full logistic "
                 "unpenalized (C=inf), full logistic L2 C=1, gbc; oracle row.")
    lines.append("- Count regression on log1p(sold count): mean, velocity OLS, OLS, "
                 "Huber, Poisson GLM (raw count), gbr; oracle row.")
    lines.append("- Beats rule: paired group-bootstrap 95%% CI (B = 1000) of the pooled "
                 "delta excludes 0 AND delta >= +%s AUC, or >= the R^2 gate."
                 % fmt(SHIP_MIN_DELTA, 2))
    lines.append("- Headline KPI: parked-capital rate at t* = 2/3 (m = h = 0.10), full "
                 "logistic vs velocity logistic, paired group bootstrap; descriptive.")
    lines.append("")
    lines.append("## Numbers fixed from the trial seeds")
    lines.append("")
    lines.append("- Leakage band (fold-mean shuffled AUC): [%s, %s] = 0.5 +/- 3 x %s "
                 "(pooled within-rung sd of %d single-fold shuffled AUCs, rounded up)."
                 % (fmt(band["auc_low"], 3), fmt(band["auc_high"], 3),
                    fmt(band["single_fold_auc_sd"], 4), band["n_auc_cells"]))
    lines.append("- Shuffled R^2 bound: fold-mean R^2 <= %s = 3 x %s (pooled within-rung sd of %d "
                 "single-fold shuffled R^2s, rounded up)."
                 % (fmt(band["r2_high"], 3), fmt(band["single_fold_r2_sd"], 4),
                    band["n_r2_cells"]))
    lines.append("- R^2 gate: delta R^2 vs velocity OLS >= %s (the larger trial fold sd of "
                 "velocity OLS R^2 across the two signals, rounded up)."
                 % fmt(trial["r2_gate"], 3))
    lines.append("- Constant rungs (majority, mean) are left out of the band: a per-fold "
                 "constant has fold AUC 0.5 by definition and a biased pooled AUC.")
    lines.append("")
    lines.append("## Predictions")
    lines.append("")
    for prediction in PREDICTIONS:
        lines.append("- **%s (%s).** %s" % (prediction["id"], prediction["kind"],
                                            prediction["text"]))
    lines.append("")
    return "\n".join(lines)


def _registration_date():
    from cardsegments.ladder import REGISTRATION_DATE
    return REGISTRATION_DATE


def _seed_text(seeds):
    parts = []
    for seed in seeds:
        parts.append(str(seed))
    return ", ".join(parts)


def find_row(rows, name):
    for row in rows:
        if row["name"] == name:
            return row
    raise KeyError(name)


def verdict(confirmed, statistic):
    if confirmed:
        label = "confirmed"
    else:
        label = "not confirmed"
    return {"verdict": label, "deciding_statistic": statistic}


def judge_p1(frozen):
    passes = True
    parts = []
    keys = sorted(frozen["signals"].keys())
    for key in keys:
        check = frozen["signals"][key]["leakage"]["band_check"]
        if not check["passes"]:
            passes = False
        aucs = frozen["signals"][key]["leakage"]["auc_by_rung"]
        names = sorted(aucs.keys())
        lowest = None
        highest = None
        for name in names:
            value = aucs[name]["mean"]
            if lowest is None or value < lowest:
                lowest = value
            if highest is None or value > highest:
                highest = value
        parts.append("signal %s: shuffled fold-mean AUC %s to %s, outside band: %d rung(s), "
                     "R^2 above bound: %d" % (key, fmt(lowest), fmt(highest),
                                              len(check["auc_outside_band"]),
                                              len(check["r2_above_bound"])))
    return verdict(passes, "; ".join(parts))


def judge_p2(frozen):
    passes = True
    parts = []
    keys = sorted(frozen["signals"].keys())
    for key in keys:
        folds = frozen["signals"][key]["folds"]
        ok = (folds["identical_folds_all_rungs"] and folds["shared_group_pairs_outer"] == 0
              and len(folds["internal_split_rungs"]) == 0)
        if not ok:
            passes = False
        parts.append("signal %s: identical folds %s, shared groups %d, internal splits %d"
                     % (key, folds["identical_folds_all_rungs"],
                        folds["shared_group_pairs_outer"], len(folds["internal_split_rungs"])))
    return verdict(passes, "; ".join(parts))


def judge_p3(frozen):
    passes = True
    parts = []
    keys = sorted(frozen["signals"].keys())
    for key in keys:
        block = frozen["signals"][key]
        n_violations = len(block["oracle_violations"])
        if n_violations > 0:
            passes = False
        parts.append("signal %s: %d oracle-bound violation(s)" % (key, n_violations))
    gap = frozen["signals"]["0.0"]["glm_oracle_deviance_gap"]
    if gap is None or gap > GLM_GAP_LIMIT:
        passes = False
    parts.append("GLM - oracle deviance at signal 0.0: %s" % fmt(gap))
    return verdict(passes, "; ".join(parts))


def beating_rungs(rows):
    names = []
    for row in rows:
        if row["beats_velocity"] is True:
            names.append(row["name"])
    return names


def best_delta_row(rows):
    best = None
    for row in rows:
        delta = row.get("delta_auc_vs_velocity")
        if delta is None:
            continue
        if best is None or delta["value"] > best["delta_auc_vs_velocity"]["value"]:
            best = row
    return best


def judge_p4(frozen):
    rows = frozen["signals"]["0.0"]["classification"]
    winners = beating_rungs(rows)
    best = best_delta_row(rows)
    delta = best["delta_auc_vs_velocity"]
    statistic = ("largest delta AUC vs velocity: %s %s [%s, %s]; rungs that beat: %d"
                 % (best["name"], fmt(delta["value"]), fmt(delta["ci_low"]),
                    fmt(delta["ci_high"]), len(winners)))
    return verdict(len(winners) == 0, statistic)


def judge_p5(frozen):
    row = find_row(frozen["signals"]["0.9"]["classification"], "logistic_l2_c1")
    delta = row["delta_auc_vs_velocity"]
    statistic = ("logistic_l2_c1 delta AUC vs velocity %s [%s, %s]"
                 % (fmt(delta["value"]), fmt(delta["ci_low"]), fmt(delta["ci_high"])))
    return verdict(row["beats_velocity"] is True, statistic)


def kpi_statistic(kpi):
    delta = kpi["delta_parked"]
    return ("parked model %s vs velocity %s; delta %s [%s, %s]"
            % (fmt(kpi["model_parked"]), fmt(kpi["velocity_parked"]), fmt(delta["value"]),
               fmt(delta["ci_low"]), fmt(delta["ci_high"])))


def judge_p6(frozen):
    kpi = frozen["signals"]["0.0"]["kpi"]
    delta = kpi["delta_parked"]
    includes_zero = delta["ci_low"] <= 0.0 <= delta["ci_high"]
    return verdict(includes_zero, kpi_statistic(kpi))


def judge_p7(frozen):
    kpi = frozen["signals"]["0.9"]["kpi"]
    delta = kpi["delta_parked"]
    return verdict(delta["ci_low"] > 0.0, kpi_statistic(kpi))


def judge_p8(frozen):
    passes = True
    parts = []
    keys = sorted(frozen["signals"].keys())
    for key in keys:
        rows = frozen["signals"][key]["regression"]
        glm = find_row(rows, "poisson_glm")
        ols = find_row(rows, "ols")
        glm_r2 = glm["r2_folds"]["mean"]
        ols_r2 = ols["r2_folds"]["mean"]
        if glm["beats_velocity"] is not True or glm_r2 <= ols_r2:
            passes = False
        delta = glm["delta_r2_vs_velocity"]
        parts.append("signal %s: GLM delta R^2 vs velocity OLS %s [%s, %s]; fold-mean R^2 "
                     "GLM %s vs OLS %s" % (key, fmt(delta["value"]), fmt(delta["ci_low"]),
                                           fmt(delta["ci_high"]), fmt(glm_r2), fmt(ols_r2)))
    parts.append("gate %s" % fmt(frozen["r2_gate"], 3))
    return verdict(passes, "; ".join(parts))


def judge_p9(frozen):
    row = find_row(frozen["signals"]["0.0"]["regression"], "ols")
    delta = row["delta_r2_vs_velocity"]
    statistic = ("OLS delta R^2 vs velocity OLS %s [%s, %s], gate %s"
                 % (fmt(delta["value"]), fmt(delta["ci_low"]), fmt(delta["ci_high"]),
                    fmt(frozen["r2_gate"], 3)))
    return verdict(row["beats_velocity"] is not True, statistic)


def judge_p10(frozen):
    rows = frozen["signals"]["0.9"]["classification"]
    reference = find_row(rows, "logistic_l2_c1")["auc_folds"]["mean"]
    best_name = None
    best_gap = None
    for row in rows:
        if row["name"] in ("oracle", "logistic_l2_c1") or row["constant"]:
            continue
        gap = row["auc_folds"]["mean"] - reference
        if best_gap is None or gap > best_gap:
            best_gap = gap
            best_name = row["name"]
    statistic = ("largest fold-mean AUC gain over logistic_l2_c1: %s %s"
                 % (best_name, fmt(best_gap)))
    return verdict(best_gap < COMPLEXITY_BAR, statistic)


JUDGES = {"P1": judge_p1, "P2": judge_p2, "P3": judge_p3, "P4": judge_p4, "P5": judge_p5,
          "P6": judge_p6, "P7": judge_p7, "P8": judge_p8, "P9": judge_p9, "P10": judge_p10}


def judge_all(frozen):
    """Verdict rows for every registered prediction, in registration order."""
    rows = []
    for prediction in PREDICTIONS:
        judge = JUDGES[prediction["id"]]
        result = judge(frozen)
        rows.append({"id": prediction["id"], "kind": prediction["kind"],
                     "text": prediction["text"], **result})
    return rows
