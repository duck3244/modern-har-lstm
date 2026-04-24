from fastapi import APIRouter, HTTPException

from ..deps import state
from ..schemas import (
    ConfusionMatrixResponse,
    PerClassResponse,
    SummaryMetrics,
)

router = APIRouter()


def _require_ready() -> None:
    if not state.metrics:
        raise HTTPException(status_code=503, detail="Metrics not ready yet")


@router.get("/summary", response_model=SummaryMetrics)
def get_summary() -> SummaryMetrics:
    _require_ready()
    return SummaryMetrics(**state.metrics["summary"])


@router.get("/confusion-matrix", response_model=ConfusionMatrixResponse)
def get_confusion_matrix() -> ConfusionMatrixResponse:
    _require_ready()
    return ConfusionMatrixResponse(**state.metrics["confusion_matrix"])


@router.get("/per-class", response_model=PerClassResponse)
def get_per_class() -> PerClassResponse:
    _require_ready()
    return PerClassResponse(**state.metrics["per_class"])
