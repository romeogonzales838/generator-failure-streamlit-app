# Generator Failure Analyzer Web App

A complete upload-only analyzer with:

- **FastAPI backend** that loads `generator_failure_model_bundle.pkl`
- **Streamlit frontend** that accepts `.csv`, `.xlsx`, or `.xlsm`
- Exact causal feature reconstruction from the training notebook
- Class-specific validation thresholds
- Six time-to-failure buckets constrained to **1–30 minutes**
- File inspection, missing-value reporting, timestamp checks, class discovery,
  probability charts, sensor trends, and downloadable results

## Architecture

```text
Browser
  │ uploads CSV/XLSX
  ▼
Streamlit frontend :8501
  │ multipart/form-data
  ▼
FastAPI backend :8000
  │ loads trusted pickle at startup
  ▼
XGBoost classifier + XGBoost time model
```

The frontend never loads the pickle. Only the backend has access to the model.

## Required input columns

```text
datetime
Voltage
Current
Temperature
Frequency
Oil Pressure (PSI)
Fuel level
```

`Failure` and `Failure Type` may be present for file inspection, but they are
never used as model inputs.

The latest continuous segment must contain at least **31 one-minute rows**,
covering `t-30` through the latest timestamp.

## Project layout

```text
generator_failure_streamlit_app/
├── backend/
│   ├── main.py
│   └── predictor.py
├── frontend/
│   └── app.py
├── models/
│   └── generator_failure_model_bundle.pkl
├── scripts/
│   └── build_pickle_bundle.py
├── .streamlit/
│   └── config.toml
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Run locally

Python 3.13 is recommended because it matches the model training environment.

### 1. Create an environment

Linux/macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start the API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive API documentation:

```text
http://localhost:8000/docs
```

### 3. Start Streamlit

In a second terminal, with the same environment active:

```bash
streamlit run frontend/app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

## Run with Docker

```bash
docker compose up --build
```

Then open:

- Streamlit: `http://localhost:8501`
- API documentation: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`

## API example

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@new_sensor_readings.xlsx" \
  -F "client_id=generator-001" \
  -F "apply_cooldown=false" \
  -F "cooldown_minutes=10" \
  -F "trend_points=240"
```

The response contains:

- Latest failure prediction
- Failure-class probabilities
- Time-bucket probabilities when an alert is issued
- Estimated lead time constrained to 1–30 minutes
- Workbook and data-quality inspection
- Sensor summaries
- Recent trend rows
- Model metadata

## Use a newly generated pickle

Replace:

```text
models/generator_failure_model_bundle.pkl
```

Or set:

```bash
export MODEL_PATH=/absolute/path/to/your/model.pkl
```

The pickle must have the same bundle structure used by this project.

To rebuild it from `generator_failure_artifacts.zip`:

```bash
python scripts/build_pickle_bundle.py \
  generator_failure_artifacts.zip \
  --output models/generator_failure_model_bundle.pkl
```

Restart the API after replacing the model.

## Deployment variables

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_PATH` | `models/generator_failure_model_bundle.pkl` | Trusted model bundle |
| `MAX_UPLOAD_MB` | `50` | API upload limit |
| `FRONTEND_ORIGINS` | Local Streamlit URLs | CORS allowlist |
| `API_URL` | `http://localhost:8000` | Backend URL used by Streamlit |
| `REQUEST_TIMEOUT_SECONDS` | `180` | Streamlit API timeout |

## Important security note

Python pickle files can execute code while loading. Only deploy a pickle created
by you or received from a fully trusted source. Never allow users to upload and
replace the model pickle through the public web interface.

## Model limitations

- The model predicts only failures occurring strictly after the latest timestamp
  and within the next 30 minutes.
- A displayed lead time is always clamped to 1–30 minutes.
- Low Oil Pressure was absent from the original training examples and therefore
  is not a trained output class in this model.
- Predictions are probabilistic and cannot guarantee every real-world failure
  will be detected correctly.
- The application is an analyzer, not a generator control or shutdown system.
  Safety-critical decisions require engineering validation and established
  operational procedures.
