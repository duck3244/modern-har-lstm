"""FastAPI application entry point. Run with: uvicorn backend.api.app:app --port 8000"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from .deps import setup_environment

# Must run before importing tensorflow anywhere in the process.
setup_environment()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..config import Config  # noqa: E402
from ..data_loader import DataLoader  # noqa: E402
from .deps import PROJECT_ROOT, resolve_model_path, state  # noqa: E402
from .routes import health, metrics, predict, samples  # noqa: E402

logger = logging.getLogger("backend.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    n_classes = len(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    return {
        "summary": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f),
            "total_samples": int(len(y_true)),
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.astype(int).tolist(),
            "normalized": cm_norm.astype(float).tolist(),
        },
        "per_class": {
            "rows": [
                {
                    "label": labels[i],
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(n_classes)
            ]
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config()
    cfg.data_dir = PROJECT_ROOT / cfg.data_dir
    cfg.dataset_dir = PROJECT_ROOT / cfg.dataset_dir

    model_path = resolve_model_path()
    logger.info("Loading model from %s", model_path)
    model = tf.keras.models.load_model(str(model_path))

    logger.info("Loading test data from %s", cfg.dataset_dir)
    dl = DataLoader(cfg)
    X_test = dl.load_signals("test")
    y_test = dl.load_labels("test")

    # Warm-up: first predict is slow due to graph building.
    model.predict(X_test[:1], verbose=0)

    logger.info("Evaluating test set (%d samples)…", len(y_test))
    proba = model.predict(X_test, verbose=0, batch_size=256)
    y_pred = np.argmax(proba, axis=1)

    state.model = model
    state.labels = cfg.labels
    state.X_test = X_test
    state.y_test = y_test
    state.metrics = _compute_metrics(y_test, y_pred, cfg.labels)
    state.metrics["model_path"] = str(model_path)

    logger.info(
        "Ready. accuracy=%.4f samples=%d",
        state.metrics["summary"]["accuracy"],
        state.metrics["summary"]["total_samples"],
    )
    yield
    state.model = None


app = FastAPI(title="HAR Dashboard API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(metrics.router, prefix="/api/metrics")
app.include_router(samples.router, prefix="/api/samples")
app.include_router(predict.router, prefix="/api/predict")
