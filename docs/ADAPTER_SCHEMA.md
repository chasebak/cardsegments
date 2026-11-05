# cardsegments BYO-CSV adapter schema (the local/public seam — single source of truth)
The public repo consumes feature matrices ONLY in this vendor-neutral form; the local lab exports real data into it. Two files per dataset:

## 1. <name>.csv — one row per card
- card_id: opaque string (local exports use "c" + a stable hash fragment — NEVER a real catalog id)
- group_id: opaque string for grouped CV (locally = a hashed set id; synthetic = planted set id)
- era_label: string, profiling/validation ONLY (vintage | mid | modern — three buckets, never raw internal era codes)
- release_year: int, profiling/validation ONLY
- pop_fresh: 0/1 metadata (stratifier, never a feature)
- f_*: numeric feature columns (float, NaN allowed). Canonical names: f_log_raw_price, f_log_multiplier, f_log_psa9_price, f_gem_rate_pooled, f_log_pop_total, f_pop_delta_30d_rate, f_log1p_sales_365, f_log1p_sales_90, f_log_sellers, f_psa10_vol_365, f_tcg_vol_365, f_momentum_90_pct. (A matrix may carry any subset; the manifest declares what's active.)
- Optional supervised columns (T0-anchored exports only): y_liquid (0/1), n_sold_outcome (int), fwd_log_price_change (float, NaN when ineligible), b_log1p_sold_90d, b_log1p_sold_365d (baseline-velocity features at T0).

## 2. <name>.manifest.json
{"name", "n_rows", "anchor_date", "outcome_window": [start, end] | null, "feature_cols": [...], "baseline_cols": [...], "metadata_cols": [...], "outcome_cols": [...], "notes"}

Rules: no data vendor or marketplace is named in the schema or prose; grade-scale abbreviations (PSA 9/10) and the generic hobby abbreviation TCG are nominative domain terminology. Counts in the manifest are code-printed; local per-card exports NEVER enter the public repo tree, and neither do aggregates computed on them — the repo ships only synthetic matrices in this same schema plus a synthetic aggregates bundle (reports/results_bundle.json, provenance label "synthetic").
