"""
api/routes/health.py

GET /health — system status and model provenance.

The model_registry block now shows which SageMaker Model Registry version is
actually running in this Lambda container — version number, ARN, approval
status, and the performance metrics stored at registration time.

On Lambda:  populated from registry metadata fetched at cold start.
Locally:    shows source=local, version_number=dev.
"""

from fastapi import APIRouter, Depends
from api.dependencies import AppState, get_state

router = APIRouter()


@router.get("/health")
def health(state: AppState = Depends(get_state)):
    high_risk = int((state.risk_scores >= state.threshold).sum())

    return {
        "status": "ok",
        "model": {
            "pipeline":  "xgb_pipeline.pkl  (SMOTEENN + XGBClassifier, 35 features)",
            "threshold": state.threshold,
            "test_set_size": len(state.eng_test_raw),
            "high_risk_customers": high_risk,
        },
        "model_registry": {
            **state.model_version,
            "group": "churn-xgboost",
        },
        "rag": {
            "chroma_collections": ["profiles", "reasons"],
        },
    }