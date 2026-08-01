# backend/main.py
"""FastAPI application exposing the generator failure pickle as an API."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor import AnalysisError, GeneratorFailurePredictor


BASE_DIRECTORY = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIRECTORY / "models" / "generator_failure_model_bundle.pkl"),
    )
)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

predictor = GeneratorFailurePredictor(MODEL_PATH)

app = FastAPI(
    title="Generator Failure Analyzer API",
    description=(
        "Upload a CSV or Excel file containing at least 30 minutes of "
        "one-minute generator sensor history. The API predicts whether a "
        "failure is likely strictly after the latest timestamp and within "
        "the next 30 minutes."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    """Describe the API and its interactive documentation."""
    return {
        "name": "Generator Failure Analyzer API",
        "health": "/health",
        "metadata": "/metadata",
        "analyze": "/analyze",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict:
    """Return API and model readiness."""
    return predictor.health()


@app.get("/metadata")
def metadata() -> dict:
    """Return saved model metadata and preprocessing configuration."""
    return {
        "model_metadata": predictor.metadata,
        "preprocessing_config": predictor.config,
        "health": predictor.health(),
    }


@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    client_id: str = Form("anonymous"),
    apply_cooldown: bool = Form(False),
    cooldown_minutes: int | None = Form(None),
    trend_points: int = Form(240),
) -> dict:
    """Analyze one uploaded CSV or Excel file."""
    filename = file.filename or "uploaded_file"
    extension = Path(filename).suffix.lower()
    if extension not in {".csv", ".xlsx", ".xlsm"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a .csv, .xlsx, or .xlsm file.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"The file exceeds the {MAX_UPLOAD_MB} MB upload limit.",
        )

    try:
        return predictor.analyze(
            content=content,
            filename=filename,
            client_id=client_id,
            apply_cooldown=apply_cooldown,
            cooldown_minutes=cooldown_minutes,
            trend_points=trend_points,
        )
    except AnalysisError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model analysis failed: {error}",
        ) from error
