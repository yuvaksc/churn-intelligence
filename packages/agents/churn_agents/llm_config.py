"""
churn_agents/llm_config.py — LLM singleton factory.

Primary:   Groq (GROQ_API_KEY, model from GROQ_MODEL)
Fallback:  Google Gemini 1.5 Flash (GOOGLE_API_KEY)

Two temperature presets:
  get_analytical_llm() → 0.1  Agents 1 & 2: consistent scoring and evidence
  get_creative_llm()   → 0.5  Agent 3: natural, varied offer language
"""

import os
from dotenv import load_dotenv

load_dotenv()   # reads .env in project root

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

_analytical_llm = None
_creative_llm   = None


def _build_llm(temperature: float):
    # ── Primary: Groq (fast inference, lower cost) ─────────────────────
    from langchain_groq import ChatGroq
    from pydantic import SecretStr
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    # Reasoning models (gpt-oss) spend hidden "reasoning" tokens on every call, and those
    # count against the tokens-per-minute limit. Low effort is plenty for these prompts.
    extra = {}
    if "gpt-oss" in model or os.getenv("GROQ_REASONING_EFFORT"):
        extra["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low")
    return ChatGroq(
        # Configurable: Groq retires models (llama-3.3-70b-versatile is gone).
        model=model,
        temperature=temperature,
        # Groq free/on-demand tiers rate-limit by tokens per minute; back off and retry
        # 429s instead of failing the war room mid-run.
        max_retries=int(os.getenv("GROQ_MAX_RETRIES", "6")),
        api_key=SecretStr(key) if (key := os.getenv("GROQ_API_KEY")) else None,
        **extra,
    )

    # ── Alternative: Google Gemini 1.5 Flash ─────────────────────────────────────
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # return ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash",
    #     temperature=temperature,
    #     api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    # )

    # ── Alternative A: OpenRouter ─────────────────────────────────────────────
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(
    #     model="google/gemini-2.5-flash",
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.getenv("OPENROUTER_API_KEY"),
    #     temperature=temperature,
    #     model_kwargs={
    #         "extra_headers": {"X-Title": "Churn-Intelligence-WarRoom"}
    #     },
    # )


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
