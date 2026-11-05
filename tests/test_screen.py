"""Feature screening: blocks, pruning, VIF, price ranking, residualization."""

import numpy as np
import pandas as pd

from cardsegments import screen


def make_frame(n_rows=400, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n_rows)
    frame = pd.DataFrame({
        "f_a": base,
        "f_a_twin": base + rng.normal(0.0, 0.05, size=n_rows),
        "f_b": rng.normal(size=n_rows),
        "f_c": rng.normal(size=n_rows),
    })
    return frame


def test_blocks_group_near_duplicates():
    frame = make_frame()
    feature_cols = list(frame.columns)
    spearman, _pearson = screen.correlation_matrices(frame, feature_cols)
    blocks = screen.feature_blocks(spearman)
    assert blocks["f_a"] == blocks["f_a_twin"]
    assert blocks["f_a"] != blocks["f_b"]
    assert blocks["f_b"] != blocks["f_c"]


def test_prune_keeps_highest_coverage():
    frame = make_frame()
    frame.loc[:49, "f_a"] = np.nan  # f_a_twin now has better coverage
    feature_cols = list(frame.columns)
    spearman, _pearson = screen.correlation_matrices(frame, feature_cols)
    blocks = screen.feature_blocks(spearman)
    kept, drops = screen.prune_blocks(frame, feature_cols, blocks)
    assert "f_a_twin" in kept
    assert "f_a" not in kept
    dropped_names = []
    for row in drops:
        dropped_names.append(row["dropped"])
    assert dropped_names == ["f_a"]


def test_vif_flags_collinear_feature():
    frame = make_frame()
    vif = screen.vif_table(frame, list(frame.columns))
    by_feature = vif.set_index("feature")["vif"]
    assert by_feature["f_a"] > 10
    assert by_feature["f_b"] < 2


def test_r2_on_price_ranks_price_proxy_first():
    rng = np.random.default_rng(4)
    price = rng.normal(size=500)
    frame = pd.DataFrame({
        "f_log_raw_price": price,
        "f_proxy": 2.0 * price + rng.normal(0.0, 0.1, size=500),
        "f_noise": rng.normal(size=500),
    })
    table = screen.r2_on_price(frame, list(frame.columns))
    assert table.iloc[0]["feature"] == "f_proxy"
    assert table.iloc[0]["r2_on_price"] > 0.9
    assert "f_log_raw_price" not in list(table["feature"])


def test_residualization_removes_price_and_keeps_nan():
    rng = np.random.default_rng(5)
    price = rng.normal(size=500)
    proxy = 1.5 * price + rng.normal(0.0, 0.3, size=500)
    frame = pd.DataFrame({"f_log_raw_price": price, "f_proxy": proxy})
    frame.loc[0, "f_proxy"] = np.nan
    fits = screen.fit_price_residualization(frame, ["f_proxy"])
    result = screen.apply_price_residualization(frame, fits)
    resid = result["f_resid_proxy"].to_numpy(dtype=float)
    assert np.isnan(resid[0])
    clean = ~np.isnan(resid)
    corr = np.corrcoef(price[clean], resid[clean])[0, 1]
    assert abs(corr) < 0.05
    assert fits["f_proxy"]["n"] == 499
