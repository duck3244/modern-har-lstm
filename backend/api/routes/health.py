from fastapi import APIRouter

from ..deps import state
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.model is not None else "loading",
        model_path=state.metrics.get("model_path", ""),
        n_test_samples=int(len(state.y_test)) if state.y_test is not None else 0,
        labels=state.labels,
    )
