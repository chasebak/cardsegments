"""The Buy List Explorer page data (docs/data/*.json) and static files.

Why: the page computes nothing, so every check on its numbers lives here.
- Freshness: the committed page data and the README / DECISIONS / METHOD
  explorer blocks equal a build from the committed results files.
- Parity: the page's 49-point columns and per-setting cells equal the
  buylist functions' output on the recomputed probabilities, at the page's
  rounding; the default setting's t* cells equal the frozen ladder's KPI.
- Privacy: SYNTHETIC labels everywhere (the build refuses others), a key
  allowlist, no card / group / release-year keys, no list longer than 64
  entries, data under 150 KB.
- Static: tokens.css is the kit's byte copy, .nojekyll exists, the Chart.js
  tag is exactly the one in the page kit's CONVENTIONS.md.
Each privacy and freshness check has a control that must fail.
"""

import json
import os
import shutil

import numpy as np
import pytest

from cardsegments import buylist, explorer, ladder, ladder_report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Typed here, not imported from explorer, so a typo there cannot pass itself.
CONVENTIONS_TAG = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" '
    'integrity="sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6djDQZjh3Y8SZ1d+6I+1zze6Z7k'
    'HXO7q3UyZAWw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
)
CONVENTIONS_FOOTER = (
    "Every number was computed by this repo's tested Python and copied; the page computes nothing."
)
KIT_TOKENS_SHA256_B64 = "J3texCt1out7QFMk3jA+rC1vsZ5U8WlE3PnDZFY8OO8="
SOURCE_FILES = (buylist.GRID_PATH_REL, ladder_report.FROZEN_PATH_REL, "README.md",
                "DECISIONS.md", "docs/METHOD.md")


def rounded(value):
    return json.loads(json.dumps(ladder.clean_for_json(value), sort_keys=True))


def read_text(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path)) as handle:
        return handle.read()


@pytest.fixture(scope="module")
def outputs():
    return explorer.expected_outputs(REPO_ROOT)


@pytest.fixture(scope="module")
def page(outputs):
    return json.loads(outputs[explorer.EXPLORER_PATH])


def copy_sources(target):
    for rel_path in SOURCE_FILES:
        destination = os.path.join(target, rel_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, rel_path), destination)


# ---------------------------------------------------------------- freshness

def test_committed_page_data_and_blocks_are_fresh():
    assert explorer.check(REPO_ROOT) == []


def test_control_check_catches_a_stale_page_file(outputs):
    stale = dict(outputs)
    text = outputs[explorer.EXPLORER_PATH]
    stale[explorer.EXPLORER_PATH] = text.replace('"default_signal": "0.9"',
                                                 '"default_signal": "0.0"', 1)
    assert stale[explorer.EXPLORER_PATH] != text
    problems = explorer.check(REPO_ROOT, outputs=stale)
    assert ("%s is stale (run %s)" % (explorer.EXPLORER_PATH, explorer.GENERATED_BY)
            in problems)


# ---------------------------------------------------------------- parity

def test_page_columns_equal_the_grid_functions(page, buylist_recompute):
    """Exact, or at most one card moved per cell after a cross-platform
    lbfgs difference (buylist.recompute_problems; moved cells reported)."""
    _payload, probs = buylist_recompute
    for key in explorer.SIGNAL_ORDER:
        y, _groups, model_probs, velocity_probs = probs[key]
        by_model = {ladder.HEADLINE_MODEL: model_probs, ladder.BASELINE_CLASSIFIER: velocity_probs}
        for name in buylist.MODEL_NAMES:
            rows = buylist.confusion_grid(y, by_model[name])
            columns = page["signals"][key]["grid"][name]
            expected = {}
            for column in explorer.GRID_COLUMNS:
                values = []
                for row in rows:
                    values.append(row[column])
                expected[column] = rounded(values)
            committed = {}
            for column in explorer.GRID_COLUMNS:
                committed[column] = columns[column]
            result = buylist.recompute_problems(expected, committed)
            assert result["problems"] == [], (key, name, result)


def test_page_setting_cells_equal_threshold_regret(page, buylist_recompute):
    _payload, probs = buylist_recompute
    for key in explorer.SIGNAL_ORDER:
        y, _groups, model_probs, velocity_probs = probs[key]
        by_model = {ladder.HEADLINE_MODEL: model_probs, ladder.BASELINE_CLASSIFIER: velocity_probs}
        settings = page["signals"][key]["settings"]
        positions = range(len(settings))
        for position in positions:
            meta = page["settings"][position]
            for name in buylist.MODEL_NAMES:
                block = buylist.threshold_regret(y, by_model[name], meta["m"], meta["h"])
                cell = settings[position]["models"][name]
                expected = rounded({"regret": block["regret"], "cost_curve": block["grid_costs"],
                                    "fp": block["at_t_star"]["fp"],
                                    "fn": block["at_t_star"]["fn"],
                                    "parked": block["at_t_star"]["parked"]})
                committed = {"regret": cell["regret"], "cost_curve": cell["cost_curve"],
                             "fp": cell["fp"], "fn": cell["fn"], "parked": cell["parked"]}
                result = buylist.recompute_problems(expected, committed)
                assert result["problems"] == [], (key, position, name, result)


def test_page_t_star_tiles_equal_the_frozen_headline(page):
    frozen, _text = ladder_report.load_frozen(REPO_ROOT)
    for key in explorer.SIGNAL_ORDER:
        kpi = frozen["signals"][key]["kpi"]
        setting = page["signals"][key]["settings"][explorer.DEFAULT_SETTING]
        model = setting["models"][ladder.HEADLINE_MODEL]
        velocity = setting["models"][ladder.BASELINE_CLASSIFIER]
        assert model["parked"] == kpi["model_parked"]
        assert velocity["parked"] == kpi["velocity_parked"]
        assert model["fn"] == kpi["model_confusion"]["fn"]
        for part in explorer.CI_KEYS:
            assert setting["delta_parked"][part] == kpi["delta_parked"][part]
            assert page["signals"][key]["delta_auc"][part] == kpi["delta_auc"][part]
        assert page["signals"][key]["auc_beats_velocity"] == kpi["auc_beats_velocity"]


def test_slider_stops_and_t_star_markers(page):
    thresholds = page["thresholds"]
    assert len(thresholds) == 49
    assert thresholds == rounded(list(buylist.THRESHOLD_GRID))
    texts = []
    for meta in page["settings"]:
        gaps = np.abs(np.array(thresholds) - meta["t_star"])
        assert meta["nearest_index"] == int(np.argmin(gaps))
        texts.append(meta["t_star_text"])
    assert texts == ["2/3", "3/5", "3/4"]
    assert page["default_signal"] in page["signal_order"]
    assert page["settings"][page["default_setting"]]["t_star_text"] == "2/3"


# ---------------------------------------------------------------- privacy and size

def test_bundle_passes_allowlist_labels_list_length_and_size(outputs):
    assert explorer.bundle_problems(outputs) == []
    total = 0
    for path in explorer.DATA_PATHS:
        total += len(outputs[path].encode("utf-8"))
    assert total <= 150_000


def test_control_bundle_check_catches_card_keys_long_lists_labels_and_size(outputs):
    page = json.loads(outputs[explorer.EXPLORER_PATH])
    page["signals"]["0.9"]["grid"][ladder.HEADLINE_MODEL]["card_id"] = [1, 2]
    page["signals"]["0.9"]["grid"][ladder.HEADLINE_MODEL]["tp"] = list(
        range(explorer.MAX_LIST_LENGTH + 1))
    page["signals"]["0.0"]["label"] = ""
    page["label"] = "private aggregate"
    planted = dict(outputs)
    planted[explorer.EXPLORER_PATH] = json.dumps(page) + " " * explorer.DATA_BUDGET_BYTES
    problems = explorer.bundle_problems(planted)
    joined = "\n".join(problems)
    assert "keys outside the allowlist: ['card_id']" in joined
    assert "card-level keys: ['card_id']" in joined
    assert "a list of %d entries" % (explorer.MAX_LIST_LENGTH + 1) in joined
    assert "unlabelled blocks: ['top label', 'signal 0.0']" in joined
    assert "budget" in joined


def test_build_refuses_an_unlabelled_source(tmp_path):
    copy_sources(str(tmp_path))
    grid_path = os.path.join(str(tmp_path), buylist.GRID_PATH_REL)
    with open(grid_path) as handle:
        grid = json.load(handle)
    grid["label"] = "real cohort"
    with open(grid_path, "w") as handle:
        json.dump(grid, handle)
    with pytest.raises(ValueError, match="is not labelled SYNTHETIC"):
        explorer.expected_outputs(str(tmp_path))
    problems = explorer.check(str(tmp_path))
    assert len(problems) == 1 and "is not labelled SYNTHETIC" in problems[0]


def test_build_refuses_a_grid_that_misses_the_frozen_run(tmp_path):
    copy_sources(str(tmp_path))
    grid_path = os.path.join(str(tmp_path), buylist.GRID_PATH_REL)
    with open(grid_path) as handle:
        grid = json.load(handle)
    cell = grid["signals"]["0.9"]["cost_settings"][0]["models"][
        ladder.HEADLINE_MODEL]["at_t_star"]
    cell["tp"] = cell["tp"] - 1
    with open(grid_path, "w") as handle:
        json.dump(grid, handle)
    with pytest.raises(ValueError, match="logistic_l2_c1 tp differs"):
        explorer.expected_outputs(str(tmp_path))


def test_meta_records_the_measured_size_and_sources(outputs):
    meta = json.loads(outputs[explorer.META_PATH])
    size = len(outputs[explorer.EXPLORER_PATH].encode("utf-8"))
    assert meta["data_bytes"] == {"explorer.json": size}
    assert meta["data_budget_bytes"] == 150_000
    assert meta["sources"] == [buylist.GRID_PATH_REL, ladder_report.FROZEN_PATH_REL]
    assert meta["tokens_css_sha256_base64"] == KIT_TOKENS_SHA256_B64


# ---------------------------------------------------------------- static files and text

def test_static_files_tokens_nojekyll_and_chart_tag():
    assert explorer.static_problems(REPO_ROOT) == []
    assert explorer.CHART_JS_TAG == CONVENTIONS_TAG
    assert explorer.FOOTER == CONVENTIONS_FOOTER
    assert explorer.TOKENS_SHA256_B64 == KIT_TOKENS_SHA256_B64
    index_text = read_text(explorer.INDEX_PATH)
    assert index_text.count(CONVENTIONS_TAG) == 1
    with open(os.path.join(REPO_ROOT, explorer.TOKENS_PATH), "rb") as handle:
        tokens = handle.read()
    assert explorer.sha256_base64(tokens) == KIT_TOKENS_SHA256_B64
    assert os.path.isfile(os.path.join(REPO_ROOT, explorer.NOJEKYLL_PATH))


def test_control_static_check_catches_an_edited_tokens_copy(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    with open(os.path.join(REPO_ROOT, explorer.TOKENS_PATH), "rb") as handle:
        tokens = handle.read()
    (docs / "tokens.css").write_bytes(tokens + b"\n")
    (docs / "index.html").write_text("<html></html>", encoding="utf-8")
    problems = explorer.static_problems(str(tmp_path))
    assert "%s is not the page kit's tokens.css" % explorer.TOKENS_PATH in problems
    assert "%s is missing" % explorer.NOJEKYLL_PATH in problems
    assert "%s lacks the pinned Chart.js tag" % explorer.INDEX_PATH in problems
    assert "%s is missing" % explorer.APP_PATH in problems


def test_page_why_text_is_the_shared_dict(page):
    index_text = read_text(explorer.INDEX_PATH)
    for key in explorer.PAGE_WHY_KEYS:
        assert page["why"][key] == ladder_report.WHY_THIS_METRIC[key]
        assert 'data-why="%s"' % key in index_text
    readme = read_text("README.md")
    for key in explorer.README_WHY_KEYS:
        assert ladder_report.why_box(key) in readme


def cards_without_why(index_text, card_ids):
    """Chart/table card ids whose <section> holds no data-why box."""
    missing = []
    for card_id in card_ids:
        marker = '<section class="card" id="%s">' % card_id
        start = index_text.find(marker)
        if start < 0:
            missing.append(card_id)
            continue
        stop = index_text.find("</section>", start)
        section = index_text[start:stop]
        if "data-why=" not in section:
            missing.append(card_id)
    return missing


def test_every_chart_and_table_card_has_a_why_box():
    index_text = read_text(explorer.INDEX_PATH)
    assert cards_without_why(index_text, explorer.CHART_AND_TABLE_CARDS) == []
    # every section holding a canvas, a table or the 2x2 lists is in the list
    holders = []
    pieces = index_text.split('<section class="card" id="')[1:]
    for piece in pieces:
        card_id = piece.split('"', 1)[0]
        body = piece.split("</section>", 1)[0]
        if "<canvas" in body or "<table" in body or 'id="lists"' in body:
            holders.append(card_id)
    assert sorted(holders) == sorted(explorer.CHART_AND_TABLE_CARDS)


def test_control_why_box_check_catches_a_missing_box():
    index_text = read_text(explorer.INDEX_PATH)
    stripped = index_text.replace('<div class="why" data-why="grid_table"></div>', "", 1)
    assert stripped != index_text
    assert cards_without_why(stripped, explorer.CHART_AND_TABLE_CARDS) == ["table-card"]


def test_every_provenance_slot_has_a_pill(page):
    index_text = read_text(explorer.INDEX_PATH)
    for key in page["provenance"]:
        assert 'data-prov="%s"' % key in index_text
        assert page["provenance"][key].startswith("synthetic · "), key
    assert page["provenance"]["slider"] == ("synthetic · precomputed at 49 thresholds; stops "
                                            "show as ticks")


def test_readme_try_it_line_names_the_hosted_page(page):
    thresholds = page["thresholds"]
    line = explorer.readme_try_it_line(len(thresholds), thresholds[0], thresholds[-1])
    readme = read_text("README.md")
    assert line in readme
    assert explorer.PAGE_URL in line
    assert "http.server" in line


def test_app_refuses_unlabelled_data_and_reads_only_the_bundle():
    app_text = read_text(explorer.APP_PATH)
    assert 'data.label.indexOf("SYNTHETIC") !== 0' in app_text
    assert 'loadJson("data/explorer.json")' in app_text
    assert app_text.count("fetch(") == 1
