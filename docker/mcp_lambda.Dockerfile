# docker/mcp_lambda.Dockerfile
# Lightweight MCP tools Lambda — no ML dependencies
FROM public.ecr.aws/lambda/python:3.12

# Install dependencies
COPY mcp_server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir mangum boto3

# Application code
COPY mcp_server/ ${LAMBDA_TASK_ROOT}/mcp_server/

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

# Lambda handler
CMD ["mcp_server.lambda_handler.handler"]