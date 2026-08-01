# scripts/build_pickle_bundle.py
"""Build one deployment pickle from the notebook's saved artifact files."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib


REQUIRED_FILES = {
    "failure_type_xgboost.joblib",
    "time_to_failure_xgboost.joblib",
    "feature_imputer.joblib",
    "label_encoders.joblib",
    "feature_columns.json",
    "alert_thresholds.json",
    "preprocessing_config.json",
    "model_metadata.json",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def locate_artifact_directory(root: Path) -> Path:
    candidates = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        existing = {path.name for path in candidate.iterdir() if path.is_file()}
        if REQUIRED_FILES.issubset(existing):
            return candidate
    raise FileNotFoundError(
        "Could not find all required model artifacts. Missing expected files: "
        + ", ".join(sorted(REQUIRED_FILES))
    )


def build_bundle(artifact_directory: Path) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "classifier": joblib.load(
            artifact_directory / "failure_type_xgboost.joblib"
        ),
        "time_to_failure_model": joblib.load(
            artifact_directory / "time_to_failure_xgboost.joblib"
        ),
        "feature_imputer": joblib.load(
            artifact_directory / "feature_imputer.joblib"
        ),
        "label_encoders": joblib.load(
            artifact_directory / "label_encoders.joblib"
        ),
        "feature_columns": load_json(
            artifact_directory / "feature_columns.json"
        ),
        "alert_thresholds": load_json(
            artifact_directory / "alert_thresholds.json"
        ),
        "preprocessing_config": load_json(
            artifact_directory / "preprocessing_config.json"
        ),
        "model_metadata": load_json(
            artifact_directory / "model_metadata.json"
        ),
        "bundle_format_version": "1.0",
        "prediction_horizon_minutes": 30,
    }

    optional_joblib = {
        "random_forest_classifier": "failure_type_random_forest.joblib",
    }
    optional_json = {
        "split_boundaries": "split_boundaries.json",
    }

    for key, filename in optional_joblib.items():
        path = artifact_directory / filename
        if path.exists():
            bundle[key] = joblib.load(path)

    for key, filename in optional_json.items():
        path = artifact_directory / filename
        if path.exists():
            bundle[key] = load_json(path)

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        help="Artifact directory or generator_failure_artifacts.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/generator_failure_model_bundle.pkl"),
    )
    arguments = parser.parse_args()

    temporary_directory: Path | None = None
    source = arguments.source.resolve()

    try:
        if source.suffix.lower() == ".zip":
            temporary_directory = Path(
                tempfile.mkdtemp(prefix="generator-artifacts-")
            )
            with zipfile.ZipFile(source, "r") as archive:
                archive.extractall(temporary_directory)
            artifact_directory = locate_artifact_directory(
                temporary_directory
            )
        else:
            artifact_directory = locate_artifact_directory(source)

        bundle = build_bundle(artifact_directory)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("wb") as file:
            pickle.dump(
                bundle,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with arguments.output.open("rb") as file:
            verified = pickle.load(file)

        print(f"Saved: {arguments.output.resolve()}")
        print(f"Bundle keys: {sorted(verified)}")
        print(
            "Classifier classes:",
            list(verified["label_encoders"]["classifier"].classes_),
        )
    finally:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)


if __name__ == "__main__":
    main()
