"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, conlist

# UCI HAR window shape: 128 timesteps x 9 channels.
Timestep = conlist(float, min_length=9, max_length=9)
Window = conlist(Timestep, min_length=128, max_length=128)


class PredictRequest(BaseModel):
    signal: Window = Field(
        ...,
        description="2D array shaped (128, 9): 128 timesteps of 9 sensor channels.",
    )


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


class SampleResponse(BaseModel):
    index: int
    signal: list[list[float]]
    true_class: int
    true_label: str


class SamplesInfoResponse(BaseModel):
    total: int
    labels: list[str]


class SummaryMetrics(BaseModel):
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    total_samples: int


class ConfusionMatrixResponse(BaseModel):
    labels: list[str]
    matrix: list[list[int]]
    normalized: list[list[float]]


class PerClassRow(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class PerClassResponse(BaseModel):
    rows: list[PerClassRow]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    n_test_samples: int
    labels: list[str]
