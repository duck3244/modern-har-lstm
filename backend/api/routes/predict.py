import csv
import io
import json

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

from ..deps import state
from ..schemas import PredictRequest, PredictResponse

router = APIRouter()

EXPECTED_SHAPE = (128, 9)


def _require_ready() -> None:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")


def _predict_array(arr: np.ndarray) -> PredictResponse:
    if arr.shape != EXPECTED_SHAPE:
        raise HTTPException(
            status_code=422,
            detail=f"Expected shape {EXPECTED_SHAPE}, got {arr.shape}",
        )
    if not np.isfinite(arr).all():
        raise HTTPException(status_code=422, detail="Input contains NaN or inf values")

    proba = state.model.predict(arr[np.newaxis, ...], verbose=0)[0]
    cls = int(np.argmax(proba))
    return PredictResponse(
        predicted_class=cls,
        predicted_label=state.labels[cls],
        confidence=float(proba[cls]),
        probabilities={state.labels[i]: float(p) for i, p in enumerate(proba)},
    )


@router.post("", response_model=PredictResponse)
def predict_json(body: PredictRequest) -> PredictResponse:
    _require_ready()
    arr = np.asarray(body.signal, dtype=np.float32)
    return _predict_array(arr)


@router.post("/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile) -> PredictResponse:
    _require_ready()
    raw = await file.read()
    name = (file.filename or "").lower()

    try:
        if name.endswith(".json"):
            arr = _parse_json(raw)
        elif name.endswith(".csv"):
            arr = _parse_csv(raw)
        elif name.endswith(".txt"):
            arr = _parse_txt(raw)
        else:
            # Best-effort auto-detect.
            text = raw.decode("utf-8", errors="replace").lstrip()
            arr = _parse_json(raw) if text.startswith(("[", "{")) else _parse_txt(raw)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}") from e

    return _predict_array(arr)


def _parse_json(raw: bytes) -> np.ndarray:
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, dict):
        if "signal" not in data:
            raise HTTPException(status_code=422, detail='JSON object must have a "signal" key')
        data = data["signal"]
    return np.asarray(data, dtype=np.float32)


def _parse_csv(raw: bytes) -> np.ndarray:
    text = raw.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        raise HTTPException(status_code=422, detail="CSV is empty")

    # Skip header if the first row is non-numeric.
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]

    try:
        arr = np.asarray([[float(c) for c in row] for row in rows], dtype=np.float32)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"CSV has non-numeric values: {e}") from e
    return arr


def _parse_txt(raw: bytes) -> np.ndarray:
    """Whitespace-delimited (numpy savetxt default), with CSV fallback."""
    text = raw.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise HTTPException(status_code=422, detail="TXT is empty")

    # Skip header if the first token can't be parsed as a number.
    first_tokens = lines[0].split()
    try:
        float(first_tokens[0])
    except (ValueError, IndexError):
        lines = lines[1:]

    # Prefer whitespace split; if the file is actually comma-delimited, fall back.
    try:
        arr = np.asarray([[float(t) for t in ln.split()] for ln in lines], dtype=np.float32)
    except ValueError:
        return _parse_csv(raw)
    return arr
