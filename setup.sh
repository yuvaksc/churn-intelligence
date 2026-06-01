#!/bin/bash
set -e

echo "==> Creating virtual environment..."
python3 -m venv .venv

echo "==> Activating venv..."
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip -q

echo "==> Installing dependencies..."
pip install -r requirements.txt -q

echo "==> Creating .env file..."
if [ ! -f .env ]; then
cat > .env <<EOF
# Kaggle credentials (from https://www.kaggle.com/settings -> API -> Create New Token)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# LLM (get from https://console.anthropic.com)
ANTHROPIC_API_KEY=your_anthropic_key

# LangSmith (get from https://smith.langchain.com)
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=churn-intelligence
EOF
echo "==> .env created — fill in your credentials before running the pipeline."
else
echo "==> .env already exists, skipping."
fi

echo ""
echo "Done! Next steps:"
echo "  1. Fill in .env with your Kaggle + API keys"
echo "  2. source .venv/bin/activate"
echo "  3. python run_pipeline.py"