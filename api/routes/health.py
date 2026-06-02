"""
api/routes/health.py

GET /health — system status, model info, and model provenance.

Top-level fields (test_set_size, high_risk_count, threshold, chroma_collections)
are kept flat because the frontend dashboard reads them directly. The new
model_registry block is added alongside them — it shows which SageMaker Model
Registry version this Lambda container loaded at cold start.

On Lambda:  model_registry populated from registry metadata fetched at cold start.
Locally:    model_registry shows source=local, version_number=dev.
"""

from fastapi import APIRouter, Depends
from api.dependencies import AppState, get_state

router = APIRouter()


@router.get("/health")
def health(state: AppState = Depends(get_state)):
    high_risk = int((state.risk_scores >= state.threshold).sum())

    return {
        "status": "ok",
        "test_set_size": len(state.eng_test_raw),
        "high_risk_count": high_risk,
        "threshold": state.threshold,
        "chroma_collections": ["profiles", "reasons"],
        "model_registry": {
            **state.model_version,
            "group": "churn-xgboost",
        },
    }