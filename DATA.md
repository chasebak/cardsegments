# Data

**Every committed dataset, number and figure in this repository is
synthetic.** The method was built for a segmentation study on a licensed
third-party market-data feed; that run, its records and its aggregates
stay private and are not published until the feed terms are read
(see `DECISIONS.md`, 2026-11-04 — Findings regenerated on synthetic
data only).

What that means in practice:

- **No real data ships, per-card or aggregate.** `reports/results_bundle.json`
  is a synthetic aggregates bundle written by `python -m
  cardsegments.synth_findings build` (labelled `provenance.label =
  "synthetic"`): medians, bootstrap CIs, counts, lifts and out-of-fold
  metrics on planted-truth data, every cell carrying its `n`, with any
  per-group cell of `n < 50` suppressed in code (`cardsegments/report.py`).
  `report.py` refuses to render a bundle without that label into
  `reports/`.
- **The repo runs end to end on synthetic data.** `python -m
  cardsegments.demo` generates planted-segment data in the same adapter
  schema (`docs/ADAPTER_SCHEMA.md`) and exercises the full pipeline;
  `python -m pytest` gates on recovering the planted structure.
- **Feature matrices enter only through the BYO-CSV adapter**
  (`cardsegments/adapters/byo_csv.py`). If you have your own per-item
  market data, export it to the adapter schema and the whole pipeline runs
  on it unchanged.
- Column names are vendor-neutral by design (e.g. `f_log_raw_price`,
  `f_gem_rate_pooled`). No data vendor or marketplace is named in the
  schema or prose; grade-scale abbreviations (PSA 9/10) and the generic
  hobby abbreviation TCG are nominative domain terminology.
