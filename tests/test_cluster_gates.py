"""The repo's hard acceptance gates on planted synthetic structure.

Gates (fixed; REAL-data stability is reported descriptively, never gated):
  - planted-truth mean ARI >= 0.9 at separation >= 3 within-cluster SDs;
  - mean ARI >= 0.6 at moderate separation (2.0);
  - planted k recovered in >= 80% of seeds at moderate separation;
  - the null controls find no structure (gap one-SE picks k=1).
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cardsegments import nullcontrol
from cardsegments.cluster import complete_case_matrix, fit_solution, run_selection
from cardsegments.synth import generate_segmented

GATE_SEEDS = [0, 1, 2, 3, 4]
GATE_K_VALUES = [1, 2, 3, 4, 5, 6]
PLANTED_K = 3
HIGH_SEPARATION = 3.0
MODERATE_SEPARATION = 2.0


def selection_run(separation, seed):
    frame, manifest, truth = generate_segmented(
        n_rows=800, n_clusters=PLANTED_K, separation=separation, seed=seed,
        missingness=True)
    feature_cols = manifest["feature_cols"]
    complete, X = complete_case_matrix(frame, feature_cols)
    selection, _tables = run_selection("gate", X, k_values=GATE_K_VALUES,
                                       n_boot=10, gap_b=20)
    solution = selection["solution"]
    complete_mask = frame[feature_cols].notna().all(axis=1).to_numpy()
    planted = truth["labels"][complete_mask]
    _model, labels = fit_solution(X, solution)
    ari = float(adjusted_rand_score(planted, labels))
    return solution, ari


def test_gate_high_separation_ari():
    ari_values = []
    for seed in GATE_SEEDS:
        _solution, ari = selection_run(HIGH_SEPARATION, seed)
        ari_values.append(ari)
    mean_ari = float(np.mean(ari_values))
    print("high-separation mean ARI over %d seeds: %.3f"
          % (len(GATE_SEEDS), mean_ari))
    assert mean_ari >= 0.9


def test_gate_moderate_separation_ari_and_k_recovery():
    ari_values = []
    k_hits = 0
    for seed in GATE_SEEDS:
        solution, ari = selection_run(MODERATE_SEPARATION, seed)
        ari_values.append(ari)
        if solution["k"] == PLANTED_K:
            k_hits += 1
    mean_ari = float(np.mean(ari_values))
    recovery = k_hits / len(GATE_SEEDS)
    print("moderate-separation mean ARI %.3f, k recovery %.0f%%"
          % (mean_ari, recovery * 100))
    assert mean_ari >= 0.6
    assert recovery >= 0.8


def test_gate_null_controls_find_no_structure():
    for seed in (0, 1):
        frame, manifest, _truth = generate_segmented(
            n_rows=800, n_clusters=PLANTED_K,
            separation=MODERATE_SEPARATION, seed=seed, missingness=True)
        complete, _X = complete_case_matrix(frame, manifest["feature_cols"])
        raw = complete[manifest["feature_cols"]].to_numpy(dtype=float)
        summaries, _tables = nullcontrol.run_null_control(
            raw, GATE_K_VALUES, gap_b=15)
        assert summaries["real"]["gap_one_se_k"] == PLANTED_K
        assert summaries["mvn_null"]["no_structure_by_gap"]
        assert summaries["shuffle_null"]["no_structure_by_gap"]
