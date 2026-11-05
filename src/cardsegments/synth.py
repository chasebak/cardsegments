"""Seeded synthetic data with PLANTED latent segments, in the adapter schema.

Moment choices (documented, fixed):
  Latent space: 6 dims. Cluster centers sit at the vertices of a regular
  simplex inside the 4 STRUCTURED dims (price, multiplier, pop, sales),
  randomly rotated per seed, with pairwise center distance equal to
  separation * sqrt(4): "separation" therefore reads as the average
  per-structured-dimension center gap in within-cluster SD units. The two
  bounded/positive dims (gem rate, volatility) carry NO planted centers -
  they are realistic nuisance axes the pipeline must see through.
  Within-cluster noise is standard Gaussian per dim ON THE LOG SCALE; the
  observed dollar-scale quantities (exp of the log features) are therefore
  lognormal-mixture distributed, i.e. heavy-tailed where real market data
  is heavy-tailed. (Student-t latent noise was tried first and rejected:
  the mixture criteria kept inventing an extra component for the tails
  even at high separation - see Design decisions in the README.)
  Feature mapping from latent dims (log-scale locations/spreads chosen to
  look like real graded-card magnitudes):
    f_log_raw_price   = 3.4 + 1.1 * z0        (log dollars)
    f_log_multiplier  = 1.4 + 0.7 * z1        (log price ratio)
    f_gem_rate_pooled = sigmoid(0.45 * z2)    (rate in (0,1); nuisance)
    f_log_pop_total   = 7.0 + 1.3 * z3        (log population)
    f_log1p_sales_365 = 4.8 + 1.2 * z4        (log1p yearly sales)
    f_tcg_vol_365     = exp(-1.6 + 0.30 * z5) (lognormal volatility; nuisance)
  Era labels: each cluster has a planted era mixture; group_id = planted set
  ids owned by one cluster each (grouped CV has real group structure).

Planted missingness (toggle, default on): f_gem_rate_pooled is missing with
probability 0.25 inside the cluster with the LOWEST population center and
0.03 elsewhere (mirrors low-pop cards lacking a stable gem rate).

Submission-selection effect (toggle, default off): rows in the bottom price
quartile are dropped with probability 0.5 (cheap cards are less likely to be
graded, so the observed roster over-represents expensive cards).
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import poisson

FEATURES = [
    "f_log_raw_price",
    "f_log_multiplier",
    "f_gem_rate_pooled",
    "f_log_pop_total",
    "f_log1p_sales_365",
    "f_tcg_vol_365",
]
METADATA_COLS = ["card_id", "group_id", "era_label", "release_year", "pop_fresh"]
STRUCTURED_DIMS = [0, 1, 3, 4]  # price, multiplier, pop, sales carry centers
ERA_ORDER = ["vintage", "mid", "modern"]
GROUPS_PER_CLUSTER = 10
MISS_GEM_LOW_POP = 0.25
MISS_GEM_BASE = 0.03
SELECTION_DROP_PROB = 0.5

# Threshold outcome mode and the illustrative preset (DECISIONS.md,
# 2026-11-04 — Illustrative preset values). Every value is a round
# illustrative choice, labelled so wherever it is shown; none is the
# private cohort's n or base rate.
LIQUID_MIN_SOLD = 5
THRESHOLD_LOG_LAMBDA_SLOPE = 1.0
THRESHOLD_CLUSTER_PATTERN = (-1.0, 1.0, 0.0)
THRESHOLD_TARGET_BASE_RATE = 0.23
ILLUSTRATIVE_N_ROWS = 12000
ILLUSTRATIVE_GROUPS_PER_CLUSTER = 50
ILLUSTRATIVE_SEED = 1104

# era mixture per cluster position (cycled when k > 4)
ERA_MIXES = [
    (0.70, 0.20, 0.10),
    (0.05, 0.15, 0.80),
    (0.10, 0.60, 0.30),
    (0.02, 0.08, 0.90),
]
ERA_YEAR_RANGES = {"vintage": (1999, 2003), "mid": (2004, 2016),
                   "modern": (2017, 2026)}


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def draw_centers(rng, n_clusters, n_dims, separation):
    """Regular-simplex centers in the structured dims, randomly rotated.

    Pairwise center distance = separation * sqrt(len(STRUCTURED_DIMS)), so
    "separation" is the average per-structured-dimension gap in
    within-cluster SD units (noise SD is 1 per dim).
    """
    n_structured = len(STRUCTURED_DIMS)
    if n_clusters > n_structured:
        raise ValueError("n_clusters must be <= %d (structured dims)"
                         % n_structured)
    identity = np.eye(n_clusters)
    simplex = identity - identity.mean(axis=0)
    gauss = rng.normal(size=(n_structured, n_clusters))
    q, _r = np.linalg.qr(gauss)
    coords = simplex @ q[:, :n_clusters].T
    base_distance = float(np.linalg.norm(coords[0] - coords[1]))
    target = separation * np.sqrt(n_structured)
    coords = coords * (target / base_distance)
    centers = np.zeros((n_clusters, n_dims))
    for position in range(n_structured):
        centers[:, STRUCTURED_DIMS[position]] = coords[:, position]
    return centers


def latent_noise(rng, n_rows, n_dims):
    """Standard Gaussian log-scale noise (unit SD per dim)."""
    return rng.normal(size=(n_rows, n_dims))


def map_latent_to_features(latent):
    """Latent z (n x 6) -> adapter-schema feature columns (documented above)."""
    columns = {}
    columns["f_log_raw_price"] = 3.4 + 1.1 * latent[:, 0]
    columns["f_log_multiplier"] = 1.4 + 0.7 * latent[:, 1]
    columns["f_gem_rate_pooled"] = sigmoid(0.45 * latent[:, 2])
    columns["f_log_pop_total"] = 7.0 + 1.3 * latent[:, 3]
    columns["f_log1p_sales_365"] = 4.8 + 1.2 * latent[:, 4]
    columns["f_tcg_vol_365"] = np.exp(-1.6 + 0.30 * latent[:, 5])
    return pd.DataFrame(columns)


def assign_eras(rng, labels, n_clusters):
    """Planted era mixture per cluster; returns (era_label, release_year)."""
    n_rows = len(labels)
    era_values = []
    year_values = []
    for position in range(n_rows):
        cluster_id = int(labels[position])
        mix = ERA_MIXES[cluster_id % len(ERA_MIXES)]
        era = rng.choice(ERA_ORDER, p=mix)
        year_low, year_high = ERA_YEAR_RANGES[era]
        year = int(rng.integers(year_low, year_high + 1))
        era_values.append(era)
        year_values.append(year)
    return np.array(era_values), np.array(year_values)


def assign_groups(rng, labels, n_clusters, groups_per_cluster=GROUPS_PER_CLUSTER):
    """Each cluster owns groups_per_cluster planted set ids.

    The default (10) draws exactly the same random stream as before the
    parameter existed, so default frames stay byte-identical."""
    n_rows = len(labels)
    group_values = []
    for position in range(n_rows):
        cluster_id = int(labels[position])
        slot = int(rng.integers(0, groups_per_cluster))
        group_values.append("gsyn%02d%02d" % (cluster_id, slot))
    return np.array(group_values)


def generate_segmented(n_rows=1200, n_clusters=4, separation=2.0, seed=0,
                       missingness=True, selection_effect=False,
                       name="synth_segments",
                       groups_per_cluster=GROUPS_PER_CLUSTER):
    """Planted-segment dataset in the adapter schema.

    Returns (frame, manifest, truth) where truth carries the planted labels
    (aligned with the returned rows) and the latent centers.
    """
    rng = np.random.default_rng(seed)
    n_dims = len(FEATURES)
    centers = draw_centers(rng, n_clusters, n_dims, separation)
    labels = rng.integers(0, n_clusters, size=n_rows)
    latent = centers[labels] + latent_noise(rng, n_rows, n_dims)
    frame = map_latent_to_features(latent)

    era_values, year_values = assign_eras(rng, labels, n_clusters)
    group_values = assign_groups(rng, labels, n_clusters, groups_per_cluster)
    card_numbers = np.arange(n_rows)
    card_ids = []
    for number in card_numbers:
        card_ids.append("csyn%06d" % number)
    frame.insert(0, "card_id", card_ids)
    frame.insert(1, "group_id", group_values)
    frame.insert(2, "era_label", era_values)
    frame.insert(3, "release_year", year_values)
    pop_fresh = rng.binomial(1, 0.85, size=n_rows)
    frame.insert(4, "pop_fresh", pop_fresh)

    if missingness:
        pop_center_per_cluster = centers[:, 3]
        low_pop_cluster = int(np.argmin(pop_center_per_cluster))
        miss_prob = np.where(labels == low_pop_cluster, MISS_GEM_LOW_POP,
                             MISS_GEM_BASE)
        miss_draw = rng.uniform(size=n_rows) < miss_prob
        gem_col = frame["f_gem_rate_pooled"].to_numpy(dtype=float).copy()
        gem_col[miss_draw] = np.nan
        frame["f_gem_rate_pooled"] = gem_col

    labels = np.asarray(labels)
    if selection_effect:
        price = frame["f_log_raw_price"].to_numpy(dtype=float)
        cutoff = float(np.quantile(price, 0.25))
        cheap = price < cutoff
        drop_draw = rng.uniform(size=n_rows) < SELECTION_DROP_PROB
        dropped = cheap & drop_draw
        frame = frame[~dropped].reset_index(drop=True)
        labels = labels[~dropped]

    manifest = build_manifest(name, frame, FEATURES,
                              notes="synthetic planted-segment data; "
                                    "separation=%.2f, k=%d, seed=%d"
                                    % (separation, n_clusters, seed))
    truth = {"labels": labels, "centers": centers,
             "separation": separation, "n_clusters": n_clusters}
    return frame, manifest, truth


def generate_supervised(n_rows=2400, n_clusters=3, separation=2.5, seed=0,
                        cluster_signal=0.9, name="synth_supervised",
                        outcome_mode="independent", groups_per_cluster=GROUPS_PER_CLUSTER,
                        target_base_rate=None):
    """Supervised variant: planted outcome structure where cluster membership
    adds signal BEYOND trailing velocity and era (so the nested gate has a
    true positive to find). ~30% of rows lack a cluster feature (exercises
    the no_cluster path).

    outcome_mode:
      "independent" (default, unchanged): y_liquid ~ Bernoulli(sigmoid(
        -1.1 + 0.8 sales_z + cluster effect + 0.5 vintage)) and
        n_sold_outcome ~ Poisson(exp(0.3 + 0.9 sales_z)), drawn
        independently, so the two outcomes do not agree card by card.
      "threshold" (the illustrative preset): n_sold_outcome ~ Poisson(lambda)
        with log lambda = a + THRESHOLD_LOG_LAMBDA_SLOPE * sales_z
        + cluster_signal * THRESHOLD_CLUSTER_PATTERN[cluster], and
        y_liquid = 1{n_sold_outcome >= LIQUID_MIN_SOLD}, the way the real
        export builds its label, so both model ladders share one outcome.
        a is solved by bisection so the PLANTED mean P(liquid) equals
        target_base_rate (THRESHOLD_TARGET_BASE_RATE when None).

    truth gains "p_liquid" (the exact probability vector y_liquid was drawn
    from, or P(N >= 5) in threshold mode) and "lambda_sold" (the exact
    Poisson rate n_sold_outcome was drawn from). The default call draws the
    same random stream as before, so default frames stay byte-identical
    (tests/test_synth.py checks this against the committed synth/ files).
    Failure mode: in threshold mode the label is a deterministic function of
    the count, so a model fed the count would score AUC 1; the count is an
    outcome column and never enters X.
    """
    if outcome_mode not in ("independent", "threshold"):
        raise ValueError("outcome_mode must be 'independent' or 'threshold'")
    frame, manifest, truth = generate_segmented(
        n_rows=n_rows, n_clusters=n_clusters, separation=separation,
        seed=seed, missingness=False, selection_effect=False, name=name,
        groups_per_cluster=groups_per_cluster)
    rng = np.random.default_rng(seed + 90001)
    labels = truth["labels"]
    n_out = len(frame)

    sales = frame["f_log1p_sales_365"].to_numpy(dtype=float)
    sales_z = (sales - sales.mean()) / sales.std(ddof=0)
    b90 = np.log1p(np.maximum(0.0, np.expm1(sales / 2.0)
                              + rng.normal(0.0, 1.0, size=n_out)))
    frame["b_log1p_sold_90d"] = b90
    frame["b_log1p_sold_365d"] = sales

    cluster_effects = []
    for cluster_id in range(n_clusters):
        effect = cluster_signal * ((cluster_id % 2) * 2.0 - 1.0)
        cluster_effects.append(effect)
    cluster_effect = np.zeros(n_out)
    for position in range(n_out):
        cluster_effect[position] = cluster_effects[int(labels[position])]
    era_effect = np.where(frame["era_label"].to_numpy() == "vintage", 0.5, 0.0)
    logit = -1.1 + 0.8 * sales_z + cluster_effect + era_effect
    probability = sigmoid(logit)
    y = rng.binomial(1, probability)
    if outcome_mode == "independent":
        lam = np.exp(0.3 + 0.9 * sales_z)
        n_sold = rng.poisson(lam)
        p_liquid = probability
    else:
        if target_base_rate is None:
            target_base_rate = THRESHOLD_TARGET_BASE_RATE
        threshold_effect = threshold_cluster_effect(labels, cluster_signal)
        log_lambda_core = THRESHOLD_LOG_LAMBDA_SLOPE * sales_z + threshold_effect
        intercept = solve_threshold_intercept(log_lambda_core, target_base_rate)
        lam = np.exp(intercept + log_lambda_core)
        n_sold = rng.poisson(lam)
        y = (n_sold >= LIQUID_MIN_SOLD).astype(int)
        p_liquid = poisson_at_least(lam, LIQUID_MIN_SOLD)
        truth["threshold_intercept"] = float(intercept)
    frame["y_liquid"] = y
    frame["n_sold_outcome"] = n_sold
    fwd = rng.normal(0.0, 0.25, size=n_out)
    eligible = rng.uniform(size=n_out) < 0.2
    fwd_col = np.where(eligible, fwd, np.nan)
    frame["fwd_log_price_change"] = fwd_col

    # planted missing cluster features: ~30% of rows lose gem rate
    miss_draw = rng.uniform(size=n_out) < 0.30
    gem_col = frame["f_gem_rate_pooled"].to_numpy(dtype=float).copy()
    gem_col[miss_draw] = np.nan
    frame["f_gem_rate_pooled"] = gem_col

    manifest = build_manifest(
        name, frame, FEATURES,
        baseline_cols=["b_log1p_sold_90d", "b_log1p_sold_365d"],
        outcome_cols=["y_liquid", "n_sold_outcome", "fwd_log_price_change"],
        notes="synthetic supervised data with planted cluster value-add; "
              "seed=%d" % seed)
    truth["p_liquid"] = np.asarray(p_liquid, dtype=float)
    truth["lambda_sold"] = np.asarray(lam, dtype=float)
    truth["outcome_mode"] = outcome_mode
    truth["cluster_signal"] = float(cluster_signal)
    return frame, manifest, truth


def poisson_at_least(lam, min_count):
    """P(N >= min_count) for N ~ Poisson(lam), elementwise (scipy's sf)."""
    return poisson.sf(min_count - 1, lam)


def threshold_cluster_effect(labels, cluster_signal):
    """cluster_signal * THRESHOLD_CLUSTER_PATTERN[label] per row.

    The pattern (-1, +1, 0) sums to zero, so turning the signal on moves
    cards between clusters' sale rates instead of shifting the whole market.
    """
    n_rows = len(labels)
    effect = np.zeros(n_rows)
    n_pattern = len(THRESHOLD_CLUSTER_PATTERN)
    positions = range(n_rows)
    for position in positions:
        pattern_value = THRESHOLD_CLUSTER_PATTERN[int(labels[position]) % n_pattern]
        effect[position] = cluster_signal * pattern_value
    return effect


def solve_threshold_intercept(log_lambda_core, target_base_rate, n_steps=60):
    """Bisection for a with mean P(Poisson(exp(a + core)) >= 5) = target.

    Why: the planted base rate should be the round illustrative value at
    every cluster_signal; one fixed intercept gives 0.16 at signal 0 and
    0.23 at 0.9 (convexity of the sale rate). The mean is increasing in a,
    so bisection on [-10, 10] converges; 60 halvings reach ~1e-17.
    """
    low = -10.0
    high = 10.0
    steps = range(n_steps)
    for _step in steps:
        middle = 0.5 * (low + high)
        lam = np.exp(middle + log_lambda_core)
        planted_rate = float(poisson_at_least(lam, LIQUID_MIN_SOLD).mean())
        if planted_rate < target_base_rate:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def illustrative_preset(cluster_signal, seed=ILLUSTRATIVE_SEED):
    """The ILLUSTRATIVE public-demo dataset (DECISIONS.md, 2026-11-04 —
    Illustrative preset values): round values, never the private cohort's
    n or base rate.

    n = 12,000 cards, 3 planted clusters x 50 set groups, planted base rate
    0.23, y_liquid = 1{61-day-style sold count >= 5}. Shown at
    cluster_signal = 0 (no planted segment effect, like the real data's
    null) and 0.9 (planted effect). Returns (frame, manifest, truth).
    """
    name = "illustrative_signal_%s" % str(cluster_signal).replace(".", "p")
    return generate_supervised(
        n_rows=ILLUSTRATIVE_N_ROWS, n_clusters=3, separation=2.5, seed=seed,
        cluster_signal=cluster_signal, name=name, outcome_mode="threshold",
        groups_per_cluster=ILLUSTRATIVE_GROUPS_PER_CLUSTER,
        target_base_rate=THRESHOLD_TARGET_BASE_RATE)


def build_manifest(name, frame, feature_cols, baseline_cols=None,
                   outcome_cols=None, notes=""):
    """Adapter-schema manifest; counts are computed from the frame."""
    if baseline_cols is None:
        baseline_cols = []
    if outcome_cols is None:
        outcome_cols = []
    manifest = {
        "name": name,
        "n_rows": int(len(frame)),
        "anchor_date": None,
        "outcome_window": None,
        "feature_cols": list(feature_cols),
        "baseline_cols": list(baseline_cols),
        "metadata_cols": list(METADATA_COLS),
        "outcome_cols": list(outcome_cols),
        "notes": notes,
    }
    return manifest


def write_dataset(frame, manifest, out_dir):
    """Write <name>.csv + <name>.manifest.json under out_dir."""
    csv_path = str(out_dir) + "/" + manifest["name"] + ".csv"
    manifest_path = str(out_dir) + "/" + manifest["name"] + ".manifest.json"
    frame.to_csv(csv_path, index=False)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return csv_path, manifest_path
