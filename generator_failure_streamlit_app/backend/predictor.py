# backend/predictor.py
"""Model loading, file validation, causal feature engineering, and prediction."""

from __future__ import annotations

import io
import math
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xlsm"}
BOUNDARY_COLUMN_CANDIDATES = (
    "segment_id",
    "Segment ID",
    "run_id",
    "Run ID",
    "generator_id",
    "Generator ID",
    "source_dataset",
    "Source Dataset",
)
FAILURE_COLUMN_CANDIDATES = ("Failure Type", "failure_type")
DEFAULT_ALERT_THRESHOLD = 0.50


class AnalysisError(ValueError):
    """Raised when an uploaded file cannot be safely analyzed."""


@dataclass(frozen=True)
class LoadedFile:
    """Parsed file and workbook metadata."""

    frame: pd.DataFrame
    selected_sheet: str
    sheet_names: list[str]


def _json_number(value: Any) -> float | int | None:
    """Convert NumPy and pandas numbers to strict JSON-compatible values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    number = float(value)
    return number if math.isfinite(number) else None


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def _read_csv(content: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise AnalysisError(f"Unable to decode the CSV file: {last_error}")


def read_sensor_file(
    content: bytes,
    filename: str,
    required_columns: list[str],
) -> LoadedFile:
    """Read CSV or select the first Excel sheet containing all required columns."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise AnalysisError(
            "Unsupported file type. Upload a .csv, .xlsx, or .xlsm file."
        )

    if extension == ".csv":
        frame = _normalize_columns(_read_csv(content))
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise AnalysisError(f"The CSV file is missing required columns: {missing}")
        return LoadedFile(frame=frame, selected_sheet="CSV", sheet_names=["CSV"])

    try:
        workbook = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
    except Exception as error:
        raise AnalysisError(f"Unable to open the Excel workbook: {error}") from error

    matching_sheets: list[tuple[str, pd.DataFrame]] = []
    for sheet_name in workbook.sheet_names:
        try:
            candidate = pd.read_excel(workbook, sheet_name=sheet_name)
        except Exception:
            continue
        candidate = _normalize_columns(candidate)
        if set(required_columns).issubset(candidate.columns):
            matching_sheets.append((sheet_name, candidate))

    if not matching_sheets:
        raise AnalysisError(
            "No worksheet contains every required column: "
            + ", ".join(required_columns)
        )

    selected_sheet, frame = matching_sheets[0]
    return LoadedFile(
        frame=frame,
        selected_sheet=selected_sheet,
        sheet_names=list(workbook.sheet_names),
    )


def _select_latest_stream(
    frame: pd.DataFrame,
    datetime_column: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    selected = frame.copy()
    selected[datetime_column] = pd.to_datetime(
        selected[datetime_column], errors="coerce"
    )
    if selected[datetime_column].isna().all():
        raise AnalysisError("No valid timestamps were found.")

    latest_row_index = selected[datetime_column].idxmax()
    identifiers: dict[str, str] = {}

    for column in BOUNDARY_COLUMN_CANDIDATES:
        if column not in selected.columns:
            continue
        latest_value = selected.loc[latest_row_index, column]
        if pd.isna(latest_value):
            continue
        selected = selected.loc[selected[column].eq(latest_value)].copy()
        identifiers[column] = str(latest_value)

    return selected, identifiers


def _grouped_rolling_series(
    values: pd.Series,
    segment_ids: pd.Series,
    window: int,
    operation: str,
) -> pd.Series:
    grouped = values.groupby(segment_ids, sort=False)
    rolling = grouped.rolling(window=window, min_periods=window)

    if operation == "mean":
        result = rolling.mean()
    elif operation == "min":
        result = rolling.min()
    elif operation == "max":
        result = rolling.max()
    elif operation == "std":
        result = rolling.std(ddof=0)
    elif operation == "median":
        result = rolling.median()
    elif operation == "slope":
        centered_time = np.arange(window, dtype=float)
        centered_time -= centered_time.mean()
        denominator = float(np.dot(centered_time, centered_time))
        result = rolling.apply(
            lambda array: float(np.dot(array, centered_time) / denominator),
            raw=True,
        )
    else:
        raise AnalysisError(f"Unsupported rolling operation: {operation}")

    return (
        result.reset_index(level=0, drop=True)
        .reindex(values.index)
        .astype(float)
    )


def engineer_time_series_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    segment_column: str = "segment_id",
) -> pd.DataFrame:
    """Recreate the exact current-and-historical features used for training."""
    datetime_column = config["datetime_column"]
    feature_config = config["feature_config"]
    required = [
        datetime_column,
        segment_column,
        *feature_config["sensor_columns"],
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AnalysisError(f"Feature engineering is missing columns: {missing}")

    ordered = frame.sort_values(
        [segment_column, datetime_column], kind="mergesort"
    ).copy()
    feature_data: dict[str, pd.Series] = {}
    lag_cache: dict[tuple[str, int], pd.Series] = {}

    for sensor in feature_config["sensor_columns"]:
        current = pd.to_numeric(ordered[sensor], errors="coerce").astype(float)
        feature_data[f"{sensor}__current"] = current
        grouped = current.groupby(ordered[segment_column], sort=False)

        for lag_minutes in feature_config["lag_minutes"]:
            lagged = grouped.shift(lag_minutes)
            lag_cache[(sensor, lag_minutes)] = lagged
            feature_data[f"{sensor}__lag_{lag_minutes}m"] = lagged

        for change_minutes in feature_config["change_minutes"]:
            lagged = lag_cache[(sensor, change_minutes)]
            change = current - lagged
            feature_data[f"{sensor}__change_{change_minutes}m"] = change

            valid_denominator = lagged.abs().gt(
                feature_config["percentage_change_epsilon"]
            )
            percentage_change = pd.Series(
                np.nan, index=ordered.index, dtype=float
            )
            percentage_change.loc[valid_denominator] = (
                change.loc[valid_denominator]
                / lagged.loc[valid_denominator]
            )
            feature_data[
                f"{sensor}__pct_change_{change_minutes}m"
            ] = percentage_change

        for rolling_minutes in feature_config["rolling_minutes"]:
            rolling_minimum = _grouped_rolling_series(
                current, ordered[segment_column], rolling_minutes, "min"
            )
            rolling_maximum = _grouped_rolling_series(
                current, ordered[segment_column], rolling_minutes, "max"
            )
            feature_data[
                f"{sensor}__rolling_mean_{rolling_minutes}m"
            ] = _grouped_rolling_series(
                current, ordered[segment_column], rolling_minutes, "mean"
            )
            feature_data[
                f"{sensor}__rolling_min_{rolling_minutes}m"
            ] = rolling_minimum
            feature_data[
                f"{sensor}__rolling_max_{rolling_minutes}m"
            ] = rolling_maximum
            feature_data[
                f"{sensor}__rolling_std_{rolling_minutes}m"
            ] = _grouped_rolling_series(
                current, ordered[segment_column], rolling_minutes, "std"
            )
            feature_data[
                f"{sensor}__rolling_median_{rolling_minutes}m"
            ] = _grouped_rolling_series(
                current, ordered[segment_column], rolling_minutes, "median"
            )
            feature_data[
                f"{sensor}__rolling_range_{rolling_minutes}m"
            ] = rolling_maximum - rolling_minimum

        for slope_minutes in feature_config["slope_minutes"]:
            feature_data[
                f"{sensor}__slope_{slope_minutes}m"
            ] = _grouped_rolling_series(
                current, ordered[segment_column], slope_minutes, "slope"
            )

    features = pd.DataFrame(feature_data, index=ordered.index)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.reindex(frame.index)


def _align_model_probabilities(
    model: Any,
    features: Any,
    total_class_count: int,
) -> np.ndarray:
    raw_probabilities = model.predict_proba(features)
    aligned = np.zeros((len(features), total_class_count), dtype=float)
    model_classes = np.asarray(model.classes_, dtype=int)
    aligned[:, model_classes] = raw_probabilities
    return aligned


def _apply_alert_thresholds(
    probabilities: np.ndarray,
    label_encoder: Any,
    thresholds: dict[str, float],
    no_failure_label: str,
) -> tuple[str, float, float]:
    class_names = label_encoder.classes_
    failure_indices = [
        index
        for index, class_name in enumerate(class_names)
        if str(class_name) != no_failure_label
    ]
    if not failure_indices:
        return no_failure_label, 0.0, DEFAULT_ALERT_THRESHOLD

    best_relative_index = int(
        np.argmax(probabilities[0, failure_indices])
    )
    best_class_index = failure_indices[best_relative_index]
    best_class_name = str(class_names[best_class_index])
    best_probability = float(probabilities[0, best_class_index])
    required_threshold = float(
        thresholds.get(best_class_name, DEFAULT_ALERT_THRESHOLD)
    )

    if best_probability < required_threshold:
        return no_failure_label, best_probability, required_threshold
    return best_class_name, best_probability, required_threshold


def _create_failure_message(
    failure_type: str,
    estimated_minutes: int,
    confidence: float,
) -> str:
    estimated_minutes = int(np.clip(estimated_minutes, 1, 30))
    confidence_percentage = int(np.clip(np.rint(confidence * 100), 0, 100))

    wording = {
        "Fuel Level": "The fuel level is likely to reach a failure condition",
        "Low Oil Pressure": (
            "The oil pressure is likely to reach a low-pressure failure condition"
        ),
        "Temperature": (
            "The generator temperature is likely to reach a failure condition"
        ),
        "Frequency": "The frequency is likely to fail",
        "Voltage": "The voltage is likely to fail",
    }.get(
        failure_type,
        f"The {failure_type.lower()} condition is likely to fail",
    )

    return (
        f"{wording} in about {estimated_minutes} minutes. "
        f"Confidence: {confidence_percentage}%."
    )


class GeneratorFailurePredictor:
    """Thread-safe service that loads one trusted pickle model bundle."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._prediction_lock = threading.Lock()
        self._cooldown_lock = threading.Lock()
        self._cooldown_state: dict[str, tuple[str, pd.Timestamp]] = {}
        self.bundle = self._load_bundle()

    def _load_bundle(self) -> dict[str, Any]:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found: {self.model_path}"
            )
        with self.model_path.open("rb") as file:
            bundle = pickle.load(file)

        required_keys = {
            "classifier",
            "time_to_failure_model",
            "feature_imputer",
            "label_encoders",
            "feature_columns",
            "alert_thresholds",
            "preprocessing_config",
            "model_metadata",
        }
        missing = required_keys - set(bundle)
        if missing:
            raise RuntimeError(
                f"The model bundle is missing keys: {sorted(missing)}"
            )

        encoders = bundle["label_encoders"]
        if not {"classifier", "time_bucket"}.issubset(encoders):
            raise RuntimeError(
                "label_encoders must contain classifier and time_bucket."
            )
        return bundle

    @property
    def config(self) -> dict[str, Any]:
        return self.bundle["preprocessing_config"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.bundle["model_metadata"]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_path": str(self.model_path),
            "classifier_classes": [
                str(value)
                for value in self.bundle["label_encoders"][
                    "classifier"
                ].classes_
            ],
            "prediction_horizon_minutes": int(
                self.config["prediction_horizon_minutes"]
            ),
            "feature_count": len(self.bundle["feature_columns"]),
            "missing_known_failure_types": self.config.get(
                "missing_known_failure_types", []
            ),
        }

    def _prepare_history(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        datetime_column = self.config["datetime_column"]
        sensor_columns = self.config["sensor_columns"]
        required_columns = [datetime_column, *sensor_columns]

        missing = [
            column for column in required_columns if column not in frame.columns
        ]
        if missing:
            raise AnalysisError(f"Missing required columns: {missing}")

        selected, stream_identifiers = _select_latest_stream(
            frame, datetime_column
        )
        prepared = selected[required_columns].copy()
        prepared[datetime_column] = pd.to_datetime(
            prepared[datetime_column], errors="coerce"
        )
        invalid_timestamps = int(prepared[datetime_column].isna().sum())
        if invalid_timestamps:
            raise AnalysisError(
                f"{invalid_timestamps} timestamp values could not be parsed."
            )

        for sensor in sensor_columns:
            prepared[sensor] = pd.to_numeric(
                prepared[sensor], errors="coerce"
            )

        prepared = prepared.sort_values(
            datetime_column, kind="mergesort"
        ).reset_index(drop=True)
        duplicate_count = int(
            prepared[datetime_column].duplicated().sum()
        )
        if duplicate_count:
            raise AnalysisError(
                f"The selected stream has {duplicate_count} duplicate timestamps. "
                "Filter the upload to one generator/run or include a generator_id, "
                "run_id, source_dataset, or segment_id column."
            )

        interval_minutes = (
            prepared[datetime_column]
            .diff()
            .dt.total_seconds()
            .div(60)
        )
        gap_threshold = float(self.config["gap_threshold_minutes"])
        starts_new_segment = (
            interval_minutes.isna()
            | interval_minutes.le(0)
            | interval_minutes.gt(gap_threshold)
        )
        prepared["segment_id"] = (
            starts_new_segment.cumsum().astype(int) - 1
        )

        latest_segment_id = int(prepared["segment_id"].iloc[-1])
        latest_segment = prepared.loc[
            prepared["segment_id"].eq(latest_segment_id)
        ].copy()

        maximum_history = int(
            self.config["feature_config"]["max_history_minutes"]
        )
        minimum_observations = maximum_history + 1
        if len(latest_segment) < minimum_observations:
            raise AnalysisError(
                "The latest continuous segment contains "
                f"{len(latest_segment)} observations. At least "
                f"{minimum_observations} one-minute observations are required."
            )

        model_window = latest_segment.tail(
            minimum_observations
        ).reset_index(drop=True)
        available_span = (
            model_window[datetime_column].iloc[-1]
            - model_window[datetime_column].iloc[0]
        ).total_seconds() / 60

        if available_span < maximum_history:
            raise AnalysisError(
                f"The latest continuous segment spans {available_span:.2f} "
                f"minutes; at least {maximum_history} minutes are required."
            )

        expected_interval = float(
            self.config["expected_interval_minutes"]
        )
        recent_intervals = (
            model_window[datetime_column]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(60)
            .to_numpy()
        )
        if not np.allclose(
            recent_intervals,
            expected_interval,
            atol=0.05,
            rtol=0.0,
        ):
            observed = sorted(set(np.round(recent_intervals, 3)))
            raise AnalysisError(
                "The latest observations must use the training interval of "
                f"{expected_interval:g} minute(s). Observed intervals: {observed}"
            )

        latest_values = model_window[sensor_columns].iloc[-1]
        if latest_values.isna().any():
            missing_latest = latest_values.index[
                latest_values.isna()
            ].tolist()
            raise AnalysisError(
                f"The latest row has invalid sensor values: {missing_latest}"
            )

        context = {
            "stream_identifiers": stream_identifiers,
            "selected_rows": int(len(prepared)),
            "segment_count": int(prepared["segment_id"].nunique()),
            "latest_segment_rows": int(len(latest_segment)),
            "latest_segment_start": latest_segment[
                datetime_column
            ].iloc[0].isoformat(),
            "latest_segment_end": latest_segment[
                datetime_column
            ].iloc[-1].isoformat(),
        }
        return model_window, latest_segment, context

    def _predict_latest(
        self,
        model_window: pd.DataFrame,
    ) -> dict[str, Any]:
        features = engineer_time_series_features(
            model_window,
            self.config,
            segment_column="segment_id",
        )
        latest_features = features.iloc[[-1]].reindex(
            columns=self.bundle["feature_columns"]
        )
        for feature_name in self.config.get(
            "all_nan_training_features", []
        ):
            latest_features[feature_name] = 0.0

        transformed = self.bundle["feature_imputer"].transform(
            latest_features
        )
        classifier_encoder = self.bundle["label_encoders"]["classifier"]
        classifier_probabilities = _align_model_probabilities(
            self.bundle["classifier"],
            transformed,
            len(classifier_encoder.classes_),
        )
        predicted_type, confidence, threshold = _apply_alert_thresholds(
            classifier_probabilities,
            classifier_encoder,
            self.bundle["alert_thresholds"],
            self.config["no_failure_label"],
        )
        probability_details = {
            str(class_name): float(
                classifier_probabilities[0, index]
            )
            for index, class_name in enumerate(
                classifier_encoder.classes_
            )
        }

        no_failure_label = self.config["no_failure_label"]
        if predicted_type == no_failure_label:
            return {
                "status": "no_alert",
                "failure_type": no_failure_label,
                "confidence": float(confidence),
                "required_threshold": float(threshold),
                "time_bucket": None,
                "estimated_minutes": None,
                "lead_time_within_30_minutes": None,
                "message": (
                    "No failure is predicted within the next 30 minutes."
                ),
                "failure_type_probabilities": probability_details,
            }

        time_encoder = self.bundle["label_encoders"]["time_bucket"]
        time_probabilities = _align_model_probabilities(
            self.bundle["time_to_failure_model"],
            transformed,
            len(time_encoder.classes_),
        )
        bucket_index = int(np.argmax(time_probabilities[0]))
        predicted_bucket = str(time_encoder.classes_[bucket_index])
        midpoints = np.array(
            [
                float(self.config["time_bucket_midpoints"][str(bucket)])
                for bucket in time_encoder.classes_
            ],
            dtype=float,
        )
        expected_minutes = float(time_probabilities[0] @ midpoints)
        estimated_minutes = int(
            np.clip(np.rint(expected_minutes), 1, 30)
        )
        if not 1 <= estimated_minutes <= 30:
            raise RuntimeError(
                "The predicted lead time exceeded the 1–30 minute limit."
            )

        return {
            "status": "alert",
            "failure_type": predicted_type,
            "confidence": float(confidence),
            "required_threshold": float(threshold),
            "time_bucket": predicted_bucket,
            "estimated_minutes": estimated_minutes,
            "lead_time_within_30_minutes": True,
            "message": _create_failure_message(
                predicted_type, estimated_minutes, confidence
            ),
            "failure_type_probabilities": probability_details,
            "time_bucket_probabilities": {
                str(bucket): float(time_probabilities[0, index])
                for index, bucket in enumerate(time_encoder.classes_)
            },
        }

    def _apply_cooldown(
        self,
        prediction: dict[str, Any],
        client_id: str,
        latest_timestamp: pd.Timestamp,
        cooldown_minutes: int,
    ) -> dict[str, Any]:
        if prediction["status"] != "alert":
            return prediction

        with self._cooldown_lock:
            previous = self._cooldown_state.get(client_id)
            if previous is not None:
                previous_type, previous_timestamp = previous
                elapsed = (
                    latest_timestamp - previous_timestamp
                ).total_seconds() / 60
                if (
                    previous_type == prediction["failure_type"]
                    and 0 <= elapsed < cooldown_minutes
                ):
                    suppressed = dict(prediction)
                    suppressed["status"] = "cooldown_suppressed"
                    suppressed["message"] = (
                        f"A {prediction['failure_type'].lower()} warning was "
                        f"already issued within the {cooldown_minutes}-minute "
                        "cooldown period."
                    )
                    return suppressed

            self._cooldown_state[client_id] = (
                prediction["failure_type"],
                latest_timestamp,
            )
        return prediction

    def _build_inspection(
        self,
        original_frame: pd.DataFrame,
        latest_segment: pd.DataFrame,
        selected_sheet: str,
        sheet_names: list[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        datetime_column = self.config["datetime_column"]
        sensor_columns = self.config["sensor_columns"]
        parsed_timestamps = pd.to_datetime(
            original_frame[datetime_column], errors="coerce"
        )
        sorted_timestamps = parsed_timestamps.dropna().sort_values()
        intervals = (
            sorted_timestamps.diff().dt.total_seconds().div(60).dropna()
        )
        positive_intervals = intervals.loc[intervals.gt(0)]

        missing_values = {
            column: int(original_frame[column].isna().sum())
            for column in [datetime_column, *sensor_columns]
        }
        invalid_sensor_values = {
            sensor: int(
                pd.to_numeric(
                    original_frame[sensor], errors="coerce"
                ).isna().sum()
                - original_frame[sensor].isna().sum()
            )
            for sensor in sensor_columns
        }

        sensor_summary: dict[str, dict[str, float | None]] = {}
        for sensor in sensor_columns:
            values = pd.to_numeric(
                latest_segment[sensor], errors="coerce"
            )
            sensor_summary[sensor] = {
                "latest": _json_number(values.iloc[-1]),
                "minimum": _json_number(values.min()),
                "maximum": _json_number(values.max()),
                "mean": _json_number(values.mean()),
                "standard_deviation": _json_number(values.std(ddof=0)),
            }

        known_failure_types = self.config.get("known_failure_types", [])
        available_failure_types: list[str] = []
        failure_counts: dict[str, int] = {}
        for column in FAILURE_COLUMN_CANDIDATES:
            if column not in original_frame.columns:
                continue
            normalized = (
                original_frame[column]
                .dropna()
                .astype(str)
                .str.strip()
            )
            normalized = normalized.loc[
                ~normalized.str.lower().isin(
                    {"", "none", "no failure", "nan"}
                )
            ]
            counts = normalized.value_counts()
            failure_counts = {
                str(label): int(count)
                for label, count in counts.items()
            }
            available_failure_types = sorted(failure_counts)
            break

        missing_failure_types = [
            failure_type
            for failure_type in known_failure_types
            if failure_type not in available_failure_types
        ]

        return {
            "selected_sheet": selected_sheet,
            "sheet_names": sheet_names,
            "row_count": int(len(original_frame)),
            "column_count": int(len(original_frame.columns)),
            "columns": [str(column) for column in original_frame.columns],
            "start_timestamp": (
                sorted_timestamps.iloc[0].isoformat()
                if not sorted_timestamps.empty
                else None
            ),
            "end_timestamp": (
                sorted_timestamps.iloc[-1].isoformat()
                if not sorted_timestamps.empty
                else None
            ),
            "median_timestamp_interval_minutes": (
                _json_number(positive_intervals.median())
                if not positive_intervals.empty
                else None
            ),
            "duplicate_timestamp_count": int(
                parsed_timestamps.duplicated().sum()
            ),
            "invalid_timestamp_count": int(
                parsed_timestamps.isna().sum()
            ),
            "missing_values": missing_values,
            "invalid_sensor_values": invalid_sensor_values,
            "available_failure_types": available_failure_types,
            "missing_known_failure_types": missing_failure_types,
            "failure_counts": failure_counts,
            "sensor_summary": sensor_summary,
            **context,
        }

    def analyze(
        self,
        content: bytes,
        filename: str,
        *,
        client_id: str = "anonymous",
        apply_cooldown: bool = False,
        cooldown_minutes: int | None = None,
        trend_points: int = 240,
    ) -> dict[str, Any]:
        datetime_column = self.config["datetime_column"]
        sensor_columns = self.config["sensor_columns"]
        required_columns = [datetime_column, *sensor_columns]
        loaded = read_sensor_file(content, filename, required_columns)
        model_window, latest_segment, context = self._prepare_history(
            loaded.frame
        )

        with self._prediction_lock:
            prediction = self._predict_latest(model_window)

        latest_timestamp = pd.Timestamp(
            model_window[datetime_column].iloc[-1]
        )
        if apply_cooldown:
            effective_cooldown = (
                int(cooldown_minutes)
                if cooldown_minutes is not None
                else int(self.config["alert_cooldown_minutes"])
            )
            if effective_cooldown < 0:
                raise AnalysisError("cooldown_minutes cannot be negative.")
            prediction = self._apply_cooldown(
                prediction,
                client_id=client_id,
                latest_timestamp=latest_timestamp,
                cooldown_minutes=effective_cooldown,
            )

        inspection = self._build_inspection(
            loaded.frame,
            latest_segment,
            loaded.selected_sheet,
            loaded.sheet_names,
            context,
        )

        trend_limit = max(31, min(int(trend_points), 2_000))
        trend_frame = latest_segment[
            [datetime_column, *sensor_columns]
        ].tail(trend_limit).copy()
        trend_frame[datetime_column] = trend_frame[
            datetime_column
        ].dt.strftime("%Y-%m-%dT%H:%M:%S")

        trend_records = []
        for record in trend_frame.to_dict(orient="records"):
            trend_records.append(
                {
                    key: (
                        value
                        if key == datetime_column
                        else _json_number(value)
                    )
                    for key, value in record.items()
                }
            )

        return {
            "filename": filename,
            "analyzed_at_timestamp": latest_timestamp.isoformat(),
            "prediction": prediction,
            "inspection": inspection,
            "trend_data": trend_records,
            "model": {
                "prediction_horizon_minutes": int(
                    self.config["prediction_horizon_minutes"]
                ),
                "feature_count": len(self.bundle["feature_columns"]),
                "classifier_classes": [
                    str(value)
                    for value in self.bundle["label_encoders"][
                        "classifier"
                    ].classes_
                ],
                "thresholds": {
                    str(key): float(value)
                    for key, value in self.bundle[
                        "alert_thresholds"
                    ].items()
                },
                "metadata": self.metadata,
            },
        }
