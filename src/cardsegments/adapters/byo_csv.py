"""BYO-CSV adapter: the one seam through which feature matrices enter.

Implements docs/ADAPTER_SCHEMA.md verbatim: a dataset is <name>.csv (one row
per card) plus <name>.manifest.json. The package consumes nothing else.
Validation is strict so a schema drift fails loudly, not silently.
"""

import json
import os

import pandas as pd

REQUIRED_METADATA = ["card_id", "group_id", "era_label", "release_year",
                     "pop_fresh"]
ERA_VALUES = ["vintage", "mid", "modern"]
MANIFEST_KEYS = ["name", "n_rows", "anchor_date", "outcome_window",
                 "feature_cols", "baseline_cols", "metadata_cols",
                 "outcome_cols", "notes"]


class AdapterError(ValueError):
    """Raised when a dataset does not satisfy the adapter schema."""


def manifest_path_for(csv_path):
    base, extension = os.path.splitext(str(csv_path))
    if extension != ".csv":
        raise AdapterError("dataset file must end in .csv: %s" % csv_path)
    return base + ".manifest.json"


def load_manifest(manifest_path):
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    for key in MANIFEST_KEYS:
        if key not in manifest:
            raise AdapterError("manifest missing key '%s': %s"
                               % (key, manifest_path))
    return manifest


def validate(frame, manifest):
    """Schema checks; raises AdapterError with a precise message."""
    for column in REQUIRED_METADATA:
        if column not in frame.columns:
            raise AdapterError("missing metadata column '%s'" % column)
    if int(manifest["n_rows"]) != len(frame):
        raise AdapterError("manifest n_rows=%d but csv has %d rows"
                           % (int(manifest["n_rows"]), len(frame)))
    era_seen = set(frame["era_label"].dropna().unique())
    bad_eras = era_seen - set(ERA_VALUES)
    if bad_eras:
        raise AdapterError("era_label values outside %s: %s"
                           % (ERA_VALUES, sorted(bad_eras)))
    declared = list(manifest["feature_cols"]) + list(manifest["baseline_cols"]) \
        + list(manifest["outcome_cols"])
    for column in declared:
        if column not in frame.columns:
            raise AdapterError("manifest declares column '%s' not in csv"
                               % column)
    for column in manifest["feature_cols"]:
        if not column.startswith("f_"):
            raise AdapterError("feature column '%s' must start with 'f_'"
                               % column)
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise AdapterError("feature column '%s' is not numeric" % column)
    if frame["card_id"].duplicated().any():
        raise AdapterError("card_id values are not unique")


def load_dataset(csv_path, manifest_path=None):
    """Load and validate one adapter-schema dataset.

    Returns (frame, manifest).
    """
    if manifest_path is None:
        manifest_path = manifest_path_for(csv_path)
    manifest = load_manifest(manifest_path)
    frame = pd.read_csv(csv_path)
    validate(frame, manifest)
    return frame, manifest
