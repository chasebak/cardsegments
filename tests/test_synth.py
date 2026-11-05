"""Synthetic generator: determinism, missingness, selection effect, schema,
planted truth, threshold mode and the illustrative preset."""

import hashlib
import io
import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from cardsegments.synth import (
    FEATURES,
    ILLUSTRATIVE_GROUPS_PER_CLUSTER,
    ILLUSTRATIVE_N_ROWS,
    LIQUID_MIN_SOLD,
    STRUCTURED_DIMS,
    THRESHOLD_TARGET_BASE_RATE,
    draw_centers,
    generate_segmented,
    generate_supervised,
    illustrative_preset,
)

PRESET_FINGERPRINT = "2d5f8bad490b8f4c"


def test_deterministic_for_same_seed():
    frame_a, manifest_a, truth_a = generate_segmented(n_rows=200, seed=9)
    frame_b, manifest_b, truth_b = generate_segmented(n_rows=200, seed=9)
    assert frame_a.equals(frame_b)
    assert manifest_a == manifest_b
    assert np.array_equal(truth_a["labels"], truth_b["labels"])


def test_center_geometry_matches_separation():
    rng = np.random.default_rng(0)
    separation = 2.5
    centers = draw_centers(rng, 3, len(FEATURES), separation)
    target = separation * np.sqrt(len(STRUCTURED_DIMS))
    for i in range(3):
        for j in range(i + 1, 3):
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            assert abs(distance - target) < 1e-9
    # nuisance dims carry no centers
    assert np.allclose(centers[:, 2], 0.0)
    assert np.allclose(centers[:, 5], 0.0)


def test_missingness_targets_low_pop_cluster():
    frame, _manifest, truth = generate_segmented(n_rows=4000, n_clusters=3,
                                                 separation=3.0, seed=2,
                                                 missingness=True)
    low_pop_cluster = int(np.argmin(truth["centers"][:, 3]))
    missing = frame["f_gem_rate_pooled"].isna().to_numpy()
    labels = truth["labels"]
    in_low = labels == low_pop_cluster
    rate_low = missing[in_low].mean()
    rate_rest = missing[~in_low].mean()
    assert rate_low > 0.15
    assert rate_rest < 0.08


def test_selection_effect_drops_cheap_rows():
    frame_full, _m, truth_full = generate_segmented(n_rows=2000, seed=3,
                                                    selection_effect=False)
    frame_sel, manifest_sel, truth_sel = generate_segmented(
        n_rows=2000, seed=3, selection_effect=True)
    assert len(frame_sel) < len(frame_full)
    assert manifest_sel["n_rows"] == len(frame_sel)
    assert len(truth_sel["labels"]) == len(frame_sel)
    # the dropped rows come from the cheap quartile, so the observed
    # price distribution shifts upward
    assert frame_sel["f_log_raw_price"].median() \
        > frame_full["f_log_raw_price"].median()


def test_supervised_variant_schema():
    frame, manifest, truth = generate_supervised(n_rows=500, seed=4)
    assert manifest["outcome_cols"] == ["y_liquid", "n_sold_outcome",
                                        "fwd_log_price_change"]
    assert manifest["baseline_cols"] == ["b_log1p_sold_90d", "b_log1p_sold_365d"]
    assert set(frame["y_liquid"].unique()) <= {0, 1}
    assert frame["f_gem_rate_pooled"].isna().mean() > 0.2
    assert frame["fwd_log_price_change"].isna().mean() > 0.5


# ---- threshold mode, planted truth and the illustrative preset (Tier 1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def assert_frame_matches_committed(frame, csv_name):
    """Strings and integers exactly; floats to 1e-12.

    Why not bytes: the committed CSVs were written on another BLAS build and
    already differ from what the unchanged generator writes here by at most
    a couple of float ulps (seen in the QR rotation); the new code was
    checked byte-identical against the previous commit's code in one
    environment. A real change to the random stream moves values by far
    more than 1e-12.
    """
    committed = pd.read_csv(os.path.join(REPO_ROOT, "synth", csv_name))
    generated = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    assert list(committed.columns) == list(generated.columns)
    assert len(committed) == len(generated)
    columns = list(committed.columns)
    for column in columns:
        left = committed[column]
        right = generated[column]
        if pd.api.types.is_float_dtype(left):
            both_nan = left.isna() & right.isna()
            assert both_nan.sum() == left.isna().sum() == right.isna().sum(), column
            difference = np.abs(left[~both_nan].to_numpy() - right[~both_nan].to_numpy())
            assert difference.max() < 1e-12, column
        else:
            assert left.equals(right), column


def test_default_generators_still_match_committed_examples():
    frame, _m, _t = generate_segmented(n_rows=1200, n_clusters=4, separation=2.5, seed=0,
                                       missingness=True, name="example_segments")
    assert_frame_matches_committed(frame, "example_segments.csv")
    frame, _m, _t = generate_supervised(n_rows=1200, n_clusters=3, separation=2.5, seed=0,
                                        name="example_supervised")
    assert_frame_matches_committed(frame, "example_supervised.csv")


def test_truth_p_liquid_is_the_exact_vector_the_generator_drew():
    """Replays the generator's own random stream: the y it drew must be
    Bernoulli(truth['p_liquid']) and the count Poisson(truth['lambda_sold'])."""
    seed = 4
    frame, _manifest, truth = generate_supervised(n_rows=800, seed=seed)
    rng = np.random.default_rng(seed + 90001)
    n_rows = len(frame)
    rng.normal(0.0, 1.0, size=n_rows)  # the b90 noise draw
    replay_y = rng.binomial(1, truth["p_liquid"])
    replay_count = rng.poisson(truth["lambda_sold"])
    assert np.array_equal(replay_y, frame["y_liquid"].to_numpy())
    assert np.array_equal(replay_count, frame["n_sold_outcome"].to_numpy())
    shifted = rng_replay_with(seed, n_rows, truth["p_liquid"] * 0.9)
    assert not np.array_equal(shifted, frame["y_liquid"].to_numpy())  # control


def rng_replay_with(seed, n_rows, probabilities):
    rng = np.random.default_rng(seed + 90001)
    rng.normal(0.0, 1.0, size=n_rows)
    return rng.binomial(1, probabilities)


def test_threshold_mode_label_is_the_thresholded_count():
    seed = 6
    frame, _manifest, truth = generate_supervised(n_rows=900, seed=seed,
                                                  outcome_mode="threshold",
                                                  target_base_rate=0.23)
    counts = frame["n_sold_outcome"].to_numpy()
    assert np.array_equal(frame["y_liquid"].to_numpy(), (counts >= LIQUID_MIN_SOLD).astype(int))
    expected_p = poisson.sf(LIQUID_MIN_SOLD - 1, truth["lambda_sold"])
    assert np.array_equal(truth["p_liquid"], expected_p)
    assert truth["p_liquid"].mean() == pytest.approx(0.23, abs=1e-9)
    rng = np.random.default_rng(seed + 90001)
    rng.normal(0.0, 1.0, size=len(frame))
    rng.binomial(1, 0.5, size=len(frame))  # the unused independent-mode draw
    assert np.array_equal(rng.poisson(truth["lambda_sold"]), counts)


def test_threshold_signal_zero_rate_depends_on_trailing_sales_only():
    frame, _manifest, truth = generate_supervised(n_rows=600, seed=2, cluster_signal=0.0,
                                                  outcome_mode="threshold")
    sales = frame["f_log1p_sales_365"].to_numpy()
    order = np.argsort(sales)
    assert np.all(np.diff(truth["lambda_sold"][order]) >= 0)


def test_unknown_outcome_mode_is_refused():
    with pytest.raises(ValueError):
        generate_supervised(n_rows=100, outcome_mode="both")


def test_illustrative_preset_uses_round_labelled_values():
    for signal in (0.0, 0.9):
        frame, manifest, truth = illustrative_preset(signal)
        assert len(frame) == ILLUSTRATIVE_N_ROWS == 12000
        assert frame["group_id"].nunique() == 3 * ILLUSTRATIVE_GROUPS_PER_CLUSTER == 150
        assert truth["p_liquid"].mean() == pytest.approx(THRESHOLD_TARGET_BASE_RATE, abs=1e-9)
        realized = frame["y_liquid"].mean()
        assert 0.21 < realized < 0.25
        assert truth["outcome_mode"] == "threshold"
        assert manifest["name"].startswith("illustrative")


def test_illustrative_preset_golden_fingerprint():
    """Frozen-number guard: the preset frame (rounded to 10 significant
    digits, which absorbs float ulp noise across BLAS builds) must not
    drift; a change to the generator's stream changes this prefix."""
    frame, _manifest, _truth = illustrative_preset(0.9)
    text = frame.to_csv(index=False, float_format="%.10g")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert digest[:16] == PRESET_FINGERPRINT
