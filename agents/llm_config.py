"""
agents/llm_config.py — LLM singleton factory.

Provider selected by env var LLM_PROVIDER:
  bedrock  → Amazon Bedrock (IAM role, no key needed)
  groq     → Groq (GROQ_API_KEY env var required)

Two temperature presets:
  get_analytical_llm() → 0.1  Agents 1 & 2: consistent scoring and evidence
  get_creative_llm()   → 0.5  Agent 3: natural, varied offer language
"""

import os

_analytical_llm = None
_creative_llm   = None

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")


def _build_llm(temperature: float):
    if LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(
            model=os.environ.get("MODEL_ANALYTICAL", "amazon.nova-lite-v1:0"),
            region_name=os.environ.get("BEDROCK_REGION", "us-east-1"),
            temperature=temperature,
            max_tokens=1000,
        )
    else:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=temperature,
            max_tokens=1000,
        )


def get_analytical_llm():
    """Temperature 0.1 — used by Agent 1 (risk scoring) and Agent 2 (evidence)."""
    global _analytical_llm
    if _analytical_llm is None:
        _analytical_llm = _build_llm(temperature=0.1)
    return _analytical_llm


def get_creative_llm():
    """Temperature 0.5 — used by Agent 3 (offer drafting)."""
    global _creative_llm
    if _creative_llm is None:
        _creative_llm = _build_llm(temperature=0.5)
    return _creative_llm