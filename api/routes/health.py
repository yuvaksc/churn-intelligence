"""api/routes/health.py — GET /health"""

from fastapi import APIRouter, Depends
from api.schemas import HealthResponse
from api.dependencies import AppState, get_state

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(state: AppState = Depends(get_state)):
    high_risk = int((state.risk_scores >= state.threshold).sum())

    return HealthResponse(
        status="ok",
        model="xgb_pipeline.pkl",
        threshold=round(state.threshold, 4),
        test_set_size=len(state.eng_test_raw),
        high_risk_count=high_risk,
        chroma_collections=["profiles", "reasons"],  # in-memory NumPy index
    )