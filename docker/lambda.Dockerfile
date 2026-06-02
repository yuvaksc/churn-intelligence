# docker/lambda.Dockerfile
# Lambda-specific image — registry-managed models are downloaded at cold start
# via SageMaker Model Registry. Only files NOT in the registry are baked in:
#   - shap_explainer.pkl    (not registered — static per model version)
#   - shap_values_test.csv  (not registered — pre-computed, static)
#   - threshold.pkl         (local dev fallback only — Lambda reads from registry)
#   - telco.csv             (raw dataset for deterministic test split)
FROM public.ecr.aws/lambda/python:3.12

# System dependencies
RUN dnf install -y gcc g++ libgomp && dnf clean all

# Python dependencies
COPY requirements.txt requirements_agents.txt requirements_api.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r requirements_agents.txt \
    -r requirements_api.txt
RUN pip install --no-cache-dir mangum sagemaker sentence-transformers

# Application code
COPY src/        ${LAMBDA_TASK_ROOT}/src/
COPY api/        ${LAMBDA_TASK_ROOT}/api/
COPY agents/     ${LAMBDA_TASK_ROOT}/agents/
COPY rag/        ${LAMBDA_TASK_ROOT}/rag/
COPY mcp_server/ ${LAMBDA_TASK_ROOT}/mcp_server/
COPY warroom_handler.py ${LAMBDA_TASK_ROOT}/warroom_handler.py

# Files NOT managed by the registry — baked into image
# xgb_pipeline.pkl, xgboost_churn.pkl, encoders.pkl, feature_names_*.pkl
# are downloaded from S3 at cold start via the registry metadata.
COPY models/shap_explainer.pkl      ${LAMBDA_TASK_ROOT}/models/shap_explainer.pkl
COPY models/shap_values_test.csv    ${LAMBDA_TASK_ROOT}/models/shap_values_test.csv
COPY models/threshold.pkl           ${LAMBDA_TASK_ROOT}/models/threshold.pkl
COPY data/raw/telco.csv             ${LAMBDA_TASK_ROOT}/data/raw/telco.csv

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

CMD ["api.main.lambda_handler"]
