# frontend/app.py
"""Streamlit frontend for the generator failure analyzer API."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))

st.set_page_config(
    page_title="Generator Failure Analyzer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
    .app-kicker {
        color: #5f6b7a;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .result-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin: 0.6rem 0 1rem 0;
        background: rgba(128, 128, 128, 0.05);
    }
    .small-note {color: #687385; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "client_id" not in st.session_state:
    st.session_state.client_id = str(uuid.uuid4())
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analyzed_filename" not in st.session_state:
    st.session_state.analyzed_filename = None


def api_get(api_url: str, path: str) -> dict[str, Any]:
    response = requests.get(
        f"{api_url}{path}",
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def parse_api_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    detail = payload.get("detail", payload)
    if isinstance(detail, list):
        return "; ".join(
            str(item.get("msg", item))
            if isinstance(item, dict)
            else str(item)
            for item in detail
        )
    return str(detail)


def analyze_upload(
    api_url: str,
    uploaded_file: Any,
    apply_cooldown: bool,
    cooldown_minutes: int,
    trend_points: int,
) -> dict[str, Any]:
    uploaded_file.seek(0)
    response = requests.post(
        f"{api_url}/analyze",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        data={
            "client_id": st.session_state.client_id,
            "apply_cooldown": str(apply_cooldown).lower(),
            "cooldown_minutes": str(cooldown_minutes),
            "trend_points": str(trend_points),
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(parse_api_error(response))
    return response.json()


def percentage(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1%}"


def value_or_dash(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{decimals}f}"


def render_prediction(result: dict[str, Any]) -> None:
    prediction = result["prediction"]
    status = prediction["status"]

    if status == "alert":
        st.error(prediction["message"], icon="🚨")
    elif status == "cooldown_suppressed":
        st.warning(prediction["message"], icon="⏱️")
    else:
        st.success(prediction["message"], icon="✅")

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Prediction",
        prediction["failure_type"],
    )
    metric_columns[1].metric(
        "Confidence",
        percentage(prediction["confidence"]),
    )
    metric_columns[2].metric(
        "Estimated lead time",
        (
            f"{prediction['estimated_minutes']} min"
            if prediction["estimated_minutes"] is not None
            else "—"
        ),
    )
    metric_columns[3].metric(
        "Time bucket",
        prediction["time_bucket"] or "—",
    )

    probabilities = pd.DataFrame(
        [
            {
                "Failure type": label,
                "Probability": probability,
            }
            for label, probability in prediction[
                "failure_type_probabilities"
            ].items()
        ]
    ).sort_values("Probability", ascending=False)

    st.subheader("Failure probability analysis")
    probability_chart = probabilities.set_index("Failure type")
    st.bar_chart(probability_chart, y="Probability")

    display_probabilities = probabilities.copy()
    display_probabilities["Probability"] = display_probabilities[
        "Probability"
    ].map(lambda value: f"{value:.2%}")
    st.dataframe(
        display_probabilities,
        hide_index=True,
        use_container_width=True,
    )

    if prediction.get("time_bucket_probabilities"):
        bucket_probabilities = pd.DataFrame(
            [
                {
                    "Time bucket": label,
                    "Probability": probability,
                }
                for label, probability in prediction[
                    "time_bucket_probabilities"
                ].items()
            ]
        )
        bucket_order = {
            "1-5 minutes": 1,
            "6-10 minutes": 2,
            "11-15 minutes": 3,
            "16-20 minutes": 4,
            "21-25 minutes": 5,
            "26-30 minutes": 6,
        }
        bucket_probabilities["order"] = bucket_probabilities[
            "Time bucket"
        ].map(bucket_order)
        bucket_probabilities = bucket_probabilities.sort_values(
            "order"
        ).drop(columns="order")

        st.subheader("Time-to-failure bucket analysis")
        st.bar_chart(
            bucket_probabilities.set_index("Time bucket"),
            y="Probability",
        )


def render_sensor_trends(result: dict[str, Any]) -> None:
    trend_data = pd.DataFrame(result["trend_data"])
    if trend_data.empty:
        st.info("No trend data were returned.")
        return

    trend_data["datetime"] = pd.to_datetime(
        trend_data["datetime"], errors="coerce"
    )
    trend_data = trend_data.dropna(subset=["datetime"]).set_index(
        "datetime"
    )

    sensors = [
        column
        for column in trend_data.columns
        if column != "datetime"
    ]
    selected_sensors = st.multiselect(
        "Sensors to display",
        options=sensors,
        default=sensors,
    )

    if not selected_sensors:
        st.info("Select at least one sensor to display its trend.")
        return

    for sensor in selected_sensors:
        st.markdown(f"**{sensor}**")
        st.line_chart(trend_data[[sensor]])


def render_file_inspection(result: dict[str, Any]) -> None:
    inspection = result["inspection"]

    overview_columns = st.columns(4)
    overview_columns[0].metric(
        "Rows",
        f"{inspection['row_count']:,}",
    )
    overview_columns[1].metric(
        "Columns",
        inspection["column_count"],
    )
    overview_columns[2].metric(
        "Detected segments",
        inspection["segment_count"],
    )
    overview_columns[3].metric(
        "Median interval",
        (
            f"{inspection['median_timestamp_interval_minutes']:.2f} min"
            if inspection["median_timestamp_interval_minutes"] is not None
            else "—"
        ),
    )

    st.write(
        {
            "Selected sheet": inspection["selected_sheet"],
            "Workbook sheets": inspection["sheet_names"],
            "Start timestamp": inspection["start_timestamp"],
            "End timestamp": inspection["end_timestamp"],
            "Latest continuous segment": (
                f"{inspection['latest_segment_start']} to "
                f"{inspection['latest_segment_end']}"
            ),
            "Latest segment rows": inspection["latest_segment_rows"],
            "Selected stream": inspection["stream_identifiers"] or "Not provided",
        }
    )

    quality_rows = []
    for column, missing_count in inspection["missing_values"].items():
        quality_rows.append(
            {
                "Column": column,
                "Missing values": missing_count,
                "Invalid numeric values": inspection[
                    "invalid_sensor_values"
                ].get(column, 0),
            }
        )
    st.subheader("Data quality")
    st.dataframe(
        pd.DataFrame(quality_rows),
        hide_index=True,
        use_container_width=True,
    )

    warning_columns = st.columns(2)
    warning_columns[0].metric(
        "Duplicate timestamps",
        inspection["duplicate_timestamp_count"],
    )
    warning_columns[1].metric(
        "Invalid timestamps",
        inspection["invalid_timestamp_count"],
    )

    st.subheader("Failure classes found in the uploaded file")
    failure_counts = inspection["failure_counts"]
    if failure_counts:
        failure_frame = pd.DataFrame(
            [
                {"Failure type": key, "Count": value}
                for key, value in failure_counts.items()
            ]
        )
        st.dataframe(
            failure_frame,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "No Failure Type column was supplied. This is valid for inference."
        )

    missing_classes = inspection["missing_known_failure_types"]
    if missing_classes:
        st.warning(
            "Known failure classes absent from this uploaded file: "
            + ", ".join(missing_classes)
        )

    st.subheader("Latest-segment sensor summary")
    sensor_summary = pd.DataFrame.from_dict(
        inspection["sensor_summary"],
        orient="index",
    ).reset_index(names="Sensor")
    st.dataframe(
        sensor_summary,
        hide_index=True,
        use_container_width=True,
    )


def render_model_details(result: dict[str, Any]) -> None:
    model = result["model"]
    metadata = model["metadata"]

    st.write(
        {
            "Prediction horizon": (
                f"{model['prediction_horizon_minutes']} minutes"
            ),
            "Engineered features": model["feature_count"],
            "Classifier classes": model["classifier_classes"],
            "Alert thresholds": model["thresholds"],
            "Model created UTC": metadata.get("created_utc"),
            "Test macro F1": metadata.get("test_macro_f1"),
        }
    )

    event_metrics = metadata.get("event_level_metrics", {})
    if event_metrics:
        st.subheader("Training test-set event metrics")
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Events detected",
            (
                f"{event_metrics.get('percentage_of_failure_events_detected', 0):.2f}%"
            ),
        )
        metric_columns[1].metric(
            "Correct type rate",
            (
                f"{event_metrics.get('correct_failure_type_rate_among_alerted_events', 0):.2f}%"
            ),
        )
        metric_columns[2].metric(
            "Median warning",
            (
                f"{event_metrics.get('median_warning_time_minutes', 0):.0f} min"
            ),
        )
        metric_columns[3].metric(
            "False alerts/day",
            (
                f"{event_metrics.get('false_alerts_per_day', 0):.2f}"
            ),
        )

    st.warning(
        "This model is restricted to failures occurring strictly after the "
        "latest timestamp and within the next 30 minutes. It cannot guarantee "
        "that every real-world prediction is correct. Operational decisions "
        "must include engineering checks and established safety procedures."
    )


with st.sidebar:
    st.header("Analyzer settings")
    api_url = st.text_input(
        "API URL",
        value=DEFAULT_API_URL,
        help="FastAPI service address.",
    ).rstrip("/")

    apply_cooldown = st.toggle(
        "Apply alert cooldown",
        value=False,
        help=(
            "Useful for repeated real-time uploads. Leave disabled for "
            "one-off historical analysis."
        ),
    )
    cooldown_minutes = st.number_input(
        "Cooldown minutes",
        min_value=0,
        max_value=120,
        value=10,
        step=1,
        disabled=not apply_cooldown,
    )
    trend_points = st.slider(
        "Trend rows",
        min_value=31,
        max_value=1_000,
        value=240,
        step=1,
    )

    st.divider()
    try:
        health = api_get(api_url, "/health")
        st.success("API connected")
        st.caption(
            f"{health['feature_count']} features · "
            f"{health['prediction_horizon_minutes']}-minute horizon"
        )
        if health.get("missing_known_failure_types"):
            st.warning(
                "Untrained class: "
                + ", ".join(health["missing_known_failure_types"])
            )
    except requests.RequestException:
        st.error("API is not reachable")
        st.caption(
            "Start FastAPI first or update the API URL."
        )

st.markdown(
    '<div class="app-kicker">Upload-only predictive analyzer</div>',
    unsafe_allow_html=True,
)
st.title("Generator Failure Analyzer")
st.write(
    "Upload a CSV or Excel file. The backend validates the latest continuous "
    "sensor history, reconstructs the trained time-series features, and "
    "predicts a failure occurring within the next 30 minutes."
)

with st.expander("Required input columns", expanded=False):
    st.code(
        "\n".join(
            [
                "datetime",
                "Voltage",
                "Current",
                "Temperature",
                "Frequency",
                "Oil Pressure (PSI)",
                "Fuel level",
            ]
        )
    )
    st.caption(
        "At least 31 continuous one-minute rows are required, representing "
        "t-30 through the latest timestamp. Failure and Failure Type columns "
        "may be present but are never used as model features."
    )

uploaded_file = st.file_uploader(
    "Upload sensor readings",
    type=["csv", "xlsx", "xlsm"],
    accept_multiple_files=False,
)

analyze_button = st.button(
    "Analyze uploaded file",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
)

if analyze_button and uploaded_file is not None:
    with st.spinner("Validating data and running the models..."):
        try:
            result = analyze_upload(
                api_url=api_url,
                uploaded_file=uploaded_file,
                apply_cooldown=apply_cooldown,
                cooldown_minutes=int(cooldown_minutes),
                trend_points=int(trend_points),
            )
        except requests.ConnectionError:
            st.error(
                "The API could not be reached. Start the FastAPI backend "
                "and verify the API URL."
            )
        except requests.Timeout:
            st.error(
                "The API request timed out. Try a smaller file or increase "
                "REQUEST_TIMEOUT_SECONDS."
            )
        except (requests.RequestException, RuntimeError) as error:
            st.error(str(error))
        else:
            st.session_state.analysis_result = result
            st.session_state.analyzed_filename = uploaded_file.name

result = st.session_state.analysis_result
if result is not None:
    st.divider()
    st.caption(
        f"Analysis result for {st.session_state.analyzed_filename} · "
        f"latest timestamp {result['analyzed_at_timestamp']}"
    )

    render_prediction(result)

    tabs = st.tabs(
        [
            "Sensor trends",
            "File inspection",
            "Model details",
            "Export",
        ]
    )

    with tabs[0]:
        render_sensor_trends(result)

    with tabs[1]:
        render_file_inspection(result)

    with tabs[2]:
        render_model_details(result)

    with tabs[3]:
        st.download_button(
            "Download complete analysis JSON",
            data=json.dumps(result, indent=2),
            file_name="generator_failure_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

        trend_frame = pd.DataFrame(result["trend_data"])
        st.download_button(
            "Download analyzed trend data CSV",
            data=trend_frame.to_csv(index=False).encode("utf-8"),
            file_name="generator_sensor_trend.csv",
            mime="text/csv",
            use_container_width=True,
        )

        prediction_export = pd.DataFrame(
            [
                {
                    "timestamp": result["analyzed_at_timestamp"],
                    "status": result["prediction"]["status"],
                    "failure_type": result["prediction"]["failure_type"],
                    "confidence": result["prediction"]["confidence"],
                    "required_threshold": result["prediction"][
                        "required_threshold"
                    ],
                    "time_bucket": result["prediction"]["time_bucket"],
                    "estimated_minutes": result["prediction"][
                        "estimated_minutes"
                    ],
                    "message": result["prediction"]["message"],
                }
            ]
        )
        st.download_button(
            "Download prediction summary CSV",
            data=prediction_export.to_csv(index=False).encode("utf-8"),
            file_name="generator_failure_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info(
        "Upload a supported sensor file and select **Analyze uploaded file**."
    )
