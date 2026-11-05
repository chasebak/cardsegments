"""Adapter-schema round trip and strict validation failures."""

import json

import pytest

from cardsegments.adapters import byo_csv
from cardsegments.synth import generate_segmented, write_dataset


@pytest.fixture()
def dataset_paths(tmp_path):
    frame, manifest, _truth = generate_segmented(n_rows=120, n_clusters=3,
                                                 separation=3.0, seed=5)
    csv_path, manifest_path = write_dataset(frame, manifest, tmp_path)
    return csv_path, manifest_path


def test_round_trip(dataset_paths):
    csv_path, manifest_path = dataset_paths
    frame, manifest = byo_csv.load_dataset(csv_path, manifest_path)
    assert len(frame) == manifest["n_rows"] == 120
    assert manifest["feature_cols"][0] == "f_log_raw_price"
    for column in byo_csv.REQUIRED_METADATA:
        assert column in frame.columns


def test_manifest_path_inferred(dataset_paths):
    csv_path, _manifest_path = dataset_paths
    frame, _manifest = byo_csv.load_dataset(csv_path)
    assert len(frame) == 120


def test_non_csv_path_rejected():
    with pytest.raises(byo_csv.AdapterError):
        byo_csv.manifest_path_for("data.parquet")


def test_missing_manifest_key(dataset_paths, tmp_path):
    csv_path, manifest_path = dataset_paths
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    del manifest["feature_cols"]
    broken_path = tmp_path / "broken.manifest.json"
    with open(broken_path, "w") as handle:
        json.dump(manifest, handle)
    with pytest.raises(byo_csv.AdapterError):
        byo_csv.load_dataset(csv_path, str(broken_path))


def test_row_count_mismatch_rejected(dataset_paths, tmp_path):
    csv_path, manifest_path = dataset_paths
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    manifest["n_rows"] = 999
    broken_path = tmp_path / "rows.manifest.json"
    with open(broken_path, "w") as handle:
        json.dump(manifest, handle)
    with pytest.raises(byo_csv.AdapterError):
        byo_csv.load_dataset(csv_path, str(broken_path))


def test_bad_era_rejected(dataset_paths, tmp_path):
    csv_path, manifest_path = dataset_paths
    frame, manifest = byo_csv.load_dataset(csv_path, manifest_path)
    frame.loc[0, "era_label"] = "antique"
    bad_csv = tmp_path / "bad_era.csv"
    frame.to_csv(bad_csv, index=False)
    with pytest.raises(byo_csv.AdapterError):
        byo_csv.load_dataset(str(bad_csv), manifest_path)


def test_duplicate_card_id_rejected(dataset_paths, tmp_path):
    csv_path, manifest_path = dataset_paths
    frame, manifest = byo_csv.load_dataset(csv_path, manifest_path)
    frame.loc[1, "card_id"] = frame.loc[0, "card_id"]
    bad_csv = tmp_path / "dup.csv"
    frame.to_csv(bad_csv, index=False)
    with pytest.raises(byo_csv.AdapterError):
        byo_csv.load_dataset(str(bad_csv), manifest_path)


def test_feature_prefix_enforced(dataset_paths, tmp_path):
    csv_path, manifest_path = dataset_paths
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    manifest["feature_cols"] = ["card_id"]
    broken_path = tmp_path / "prefix.manifest.json"
    with open(broken_path, "w") as handle:
        json.dump(manifest, handle)
    with pytest.raises(byo_csv.AdapterError):
        byo_csv.load_dataset(csv_path, str(broken_path))
