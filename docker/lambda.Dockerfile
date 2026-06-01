# docker/lambda.Dockerfile
# Lambda-specific image — models baked in, no bind mounts, no uvicorn
FROM public.ecr.aws/lambda/python:3.12

# System dependencies
RUN dnf install -y gcc g++ libgomp && dnf clean all


# Python dependencies
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

# Bake models directly into the image (fast cold start — no S3 download needed)
COPY models/xgb_pipeline.pkl        ${LAMBDA_TASK_ROOT}/models/xgb_pipeline.pkl
COPY models/xgboost_churn.pkl       ${LAMBDA_TASK_ROOT}/models/xgboost_churn.pkl
COPY models/encoders.pkl            ${LAMBDA_TASK_ROOT}/models/encoders.pkl
COPY models/feature_names_35.pkl    ${LAMBDA_TASK_ROOT}/models/feature_names_35.pkl
COPY models/feature_names_36.pkl    ${LAMBDA_TASK_ROOT}/models/feature_names_36.pkl
COPY models/threshold.pkl           ${LAMBDA_TASK_ROOT}/models/threshold.pkl
COPY models/shap_explainer.pkl      ${LAMBDA_TASK_ROOT}/models/shap_explainer.pkl
COPY data/raw/telco.csv    ${LAMBDA_TASK_ROOT}/data/raw/telco.csv
COPY models/shap_values_test.csv    ${LAMBDA_TASK_ROOT}/models/shap_values_test.csv
COPY warroom_handler.py  ${LAMBDA_TASK_ROOT}/warroom_handler.py

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

# Lambda handler entrypoint
CMD ["api.main.lambda_handler"]