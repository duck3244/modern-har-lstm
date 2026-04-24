"""Shared resources (model, metrics cache, test data) loaded once at startup."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


@dataclass
class AppState:
    model: object | None = None
    labels: list[str] = field(default_factory=list)
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    # Cached metrics on the full test set, computed once at startup.
    metrics: dict = field(default_factory=dict)


state = AppState()


def resolve_model_path() -> Path:
    """Prefer best_model.keras, fall back to best_model.h5, inside backend/."""
    for name in ("best_model.keras", "best_model.h5"):
        p = BACKEND_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No model found. Looked for best_model.keras / best_model.h5 in {BACKEND_DIR}"
    )


def setup_environment() -> None:
    """Suppress TF chatter and force CPU-only inference."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
