import random

from fastapi import APIRouter, HTTPException

from ..deps import state
from ..schemas import SampleResponse, SamplesInfoResponse

router = APIRouter()


def _require_ready() -> None:
    if state.X_test is None or state.y_test is None:
        raise HTTPException(status_code=503, detail="Test data not ready yet")


@router.get("", response_model=SamplesInfoResponse)
def get_samples_info() -> SamplesInfoResponse:
    _require_ready()
    return SamplesInfoResponse(total=int(len(state.y_test)), labels=state.labels)


@router.get("/random", response_model=SampleResponse)
def get_random_sample() -> SampleResponse:
    _require_ready()
    idx = random.randrange(len(state.y_test))
    return _build_sample(idx)


@router.get("/{idx}", response_model=SampleResponse)
def get_sample(idx: int) -> SampleResponse:
    _require_ready()
    if idx < 0 or idx >= len(state.y_test):
        raise HTTPException(status_code=404, detail=f"Sample index out of range: {idx}")
    return _build_sample(idx)


def _build_sample(idx: int) -> SampleResponse:
    signal = state.X_test[idx]
    true_cls = int(state.y_test[idx])
    return SampleResponse(
        index=idx,
        signal=signal.astype(float).tolist(),
        true_class=true_cls,
        true_label=state.labels[true_cls],
    )
