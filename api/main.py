"""FastAPI server for HAM10000 skin lesion classifier.

Loads the model once at startup, exposes:
  GET  /health   — liveness check + model metadata
  POST /predict  — upload an image, get a prediction

Run locally:
  uv run uvicorn api.main:app --reload
"""
import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.model import ClassifierService


# Populated at startup, used by every request
service: ClassifierService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts."""
    global service
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pt")
    print(f"Loading model from {checkpoint_path}...")
    service = ClassifierService(config_path, checkpoint_path)
    print(f"Model ready. Epoch {service.checkpoint_epoch}, "
          f"val F1={service.checkpoint_val_f1:.4f}")
    yield
    # No teardown needed


app = FastAPI(
    title="HAM10000 Skin Lesion Classifier",
    description="Fine-tuned ResNet-50 on the HAM10000 dermatoscopic dataset. "
                "Not a medical device. Not for diagnostic use.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Liveness check. Returns model metadata for verifying deployed version."""
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model": "resnet50",
        "checkpoint_epoch": service.checkpoint_epoch,
        "checkpoint_val_f1": round(service.checkpoint_val_f1, 4),
        "classes": service.class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify a single dermatoscopic image.

    Accepts an image file (JPEG or PNG). Returns predicted class,
    confidence, and probabilities for all seven classes.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode file '{file.filename}' as an image.",
        )

    result = service.predict(image)
    result["filename"] = file.filename
    return result