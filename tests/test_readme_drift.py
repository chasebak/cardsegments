"""README-numbers drift test: every generated block in README.md,
DECISIONS.md, docs/METHOD.md and results/LADDER.md must equal what
ladder_report renders from the committed results/ladder_frozen.json,
results/buylist_grid.json (KPI rows for decision cost, regret, Brier
skill and ECE) and results/leak_detection.json (the band's detection
limit), and
every synthetic-findings block (README headline findings, DECISIONS,
METHOD assumptions) plus reports/findings.md must equal what
synth_findings renders from the committed reports/results_bundle.json,
and every Buy List Explorer block (README table and Try it line,
DECISIONS, METHOD assumptions) must equal what `python -m cardsegments.viz
build` renders from results/buylist_grid.json.

Why: numbers are computed by tested scripts and only copied into prose; a
hand edit, or a rerun whose JSON was committed without re-rendering, shows
up here as a stale file. The controls prove the checks can fail: one digit
changed in a copy of the README, or of findings.md, must be caught.
"""

import os
import re
import shutil

from cardsegments import buylist, explorer, ladder_report, synth_findings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = ["README.md", "DECISIONS.md", "docs/METHOD.md", "results/LADDER.md",
             "results/ladder_frozen.json", "results/buylist_grid.json",
             "results/leak_detection.json"]


def test_readme_numbers_match_bundle():
    assert ladder_report.stale_files(REPO_ROOT) == []


def copy_tree(target):
    for rel_path in GENERATED:
        destination = os.path.join(target, rel_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, rel_path), destination)


def test_drift_check_catches_one_changed_digit(tmp_path):
    copy_tree(str(tmp_path))
    readme_path = os.path.join(str(tmp_path), "README.md")
    with open(readme_path) as handle:
        text = handle.read()
    block = ladder_report.extract_block(text, ladder_report.README_HEADLINE)
    match = re.search(r"\d", block)
    assert match is not None
    digit = block[match.start()]
    replacement = "7" if digit != "7" else "3"
    tampered_block = block[:match.start()] + replacement + block[match.start() + 1:]
    tampered = text.replace(block, tampered_block, 1)
    with open(readme_path, "w") as handle:
        handle.write(tampered)
    assert ladder_report.stale_files(str(tmp_path)) == ["README.md"]


def tamper_kpi_row(tmp_path, row_start):
    """Change the first digit after 'value now' starts in one KPI row."""
    copy_tree(str(tmp_path))
    readme_path = os.path.join(str(tmp_path), "README.md")
    with open(readme_path) as handle:
        text = handle.read()
    start = text.index(row_start)
    stop = text.index("\n", start)
    row = text[start:stop]
    value_start = row.index("signal 0.9:")
    tampered_row = row[:value_start] + change_first_digit(row[value_start:])
    with open(readme_path, "w") as handle:
        handle.write(text.replace(row, tampered_row, 1))
    return ladder_report.stale_files(str(tmp_path))


def test_drift_check_catches_a_changed_calibration_number(tmp_path):
    assert tamper_kpi_row(tmp_path, "| Calibration |") == ["README.md"]


def test_drift_check_catches_a_changed_regret_number(tmp_path):
    assert tamper_kpi_row(tmp_path, "| Decision cost and threshold regret |") == ["README.md"]


def test_kpi_rows_copy_the_grid_file():
    """The calibration and regret rows carry the grid's numbers, formatted."""
    with open(os.path.join(REPO_ROOT, "README.md")) as handle:
        readme = handle.read()
    grid, _text = ladder_report.load_results(ladder_report.GRID_PATH_REL, REPO_ROOT)
    model = grid["signals"]["0.9"]["models"]["logistic_l2_c1"]["calibration"]
    assert "skill %.3f, ECE %.3f" % (model["brier_skill"], model["ece"]) in readme
    cell = grid["signals"]["0.9"]["cost_settings"][0]["models"]["logistic_l2_c1"]
    assert "model cost(t*) %.5f" % cell["cost_t_star"] in readme
    assert "to measure" not in readme.split("| Calibration |", 1)[1].split("\n", 1)[0]


def test_headline_is_labelled_synthetic_and_bold():
    with open(os.path.join(REPO_ROOT, "README.md")) as handle:
        text = handle.read()
    block = ladder_report.extract_block(text, ladder_report.README_HEADLINE)
    assert block.startswith("**SYNTHETIC")
    assert block.endswith("**")


def test_why_this_metric_has_three_lines_per_box():
    for key in ladder_report.WHY_THIS_METRIC:
        box = ladder_report.WHY_THIS_METRIC[key]
        assert sorted(box.keys()) == ["kind", "lies", "question"]
        for part in box:
            assert len(box[part]) > 20


FINDINGS_GENERATED = ["README.md", "DECISIONS.md", "docs/METHOD.md",
                      synth_findings.BUNDLE_REL, synth_findings.FINDINGS_REL]


def copy_findings_tree(target):
    for rel_path in FINDINGS_GENERATED:
        destination = os.path.join(target, rel_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, rel_path), destination)


def change_first_digit(text):
    match = re.search(r"\d", text)
    assert match is not None
    digit = text[match.start()]
    replacement = "7" if digit != "7" else "3"
    return text[:match.start()] + replacement + text[match.start() + 1:]


def test_findings_blocks_match_bundle():
    assert synth_findings.stale_files(REPO_ROOT) == []


def test_findings_drift_check_catches_one_changed_readme_digit(tmp_path):
    copy_findings_tree(str(tmp_path))
    readme_path = os.path.join(str(tmp_path), "README.md")
    with open(readme_path) as handle:
        text = handle.read()
    block = ladder_report.extract_block(text, synth_findings.README_BLOCK)
    tampered = text.replace(block, change_first_digit(block), 1)
    with open(readme_path, "w") as handle:
        handle.write(tampered)
    assert synth_findings.stale_files(str(tmp_path)) == ["README.md"]


def test_findings_drift_check_catches_one_changed_report_digit(tmp_path):
    copy_findings_tree(str(tmp_path))
    findings_path = os.path.join(str(tmp_path), synth_findings.FINDINGS_REL)
    with open(findings_path) as handle:
        text = handle.read()
    marker = "## Supervised validation"
    position = text.index(marker)
    tampered = text[:position] + change_first_digit(text[position:])
    with open(findings_path, "w") as handle:
        handle.write(tampered)
    assert synth_findings.stale_files(str(tmp_path)) == [synth_findings.FINDINGS_REL]


def test_findings_headline_is_labelled_synthetic():
    with open(os.path.join(REPO_ROOT, "README.md")) as handle:
        text = handle.read()
    block = ladder_report.extract_block(text, synth_findings.README_BLOCK)
    assert block.startswith("## Headline findings (SYNTHETIC v2")


EXPLORER_GENERATED = ["README.md", "DECISIONS.md", "docs/METHOD.md",
                      buylist.GRID_PATH_REL, ladder_report.FROZEN_PATH_REL,
                      explorer.EXPLORER_PATH, explorer.META_PATH]


def copy_explorer_tree(target):
    for rel_path in EXPLORER_GENERATED:
        destination = os.path.join(target, rel_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, rel_path), destination)


def test_explorer_blocks_match_the_buylist_grid():
    """The README explorer table, DECISIONS entries and METHOD assumptions
    equal what `python -m cardsegments.viz build` renders from
    results/buylist_grid.json and results/ladder_frozen.json."""
    assert explorer.stale_files(REPO_ROOT) == []


def test_explorer_drift_check_catches_one_changed_readme_digit(tmp_path):
    copy_explorer_tree(str(tmp_path))
    readme_path = os.path.join(str(tmp_path), "README.md")
    with open(readme_path) as handle:
        text = handle.read()
    block = ladder_report.extract_block(text, explorer.README_BLOCK)
    table_start = block.index("| signal 0 |")
    tampered_block = block[:table_start] + change_first_digit(block[table_start:])
    tampered = text.replace(block, tampered_block, 1)
    with open(readme_path, "w") as handle:
        handle.write(tampered)
    assert explorer.stale_files(str(tmp_path)) == ["README.md"]


def test_explorer_drift_check_catches_one_changed_decision_digit(tmp_path):
    copy_explorer_tree(str(tmp_path))
    decisions_path = os.path.join(str(tmp_path), "DECISIONS.md")
    with open(decisions_path) as handle:
        text = handle.read()
    block = ladder_report.extract_block(text, explorer.DECISIONS_BLOCK)
    body_start = block.index("measurement:")
    tampered_block = block[:body_start] + change_first_digit(block[body_start:])
    tampered = text.replace(block, tampered_block, 1)
    with open(decisions_path, "w") as handle:
        handle.write(tampered)
    assert explorer.stale_files(str(tmp_path)) == ["DECISIONS.md"]
