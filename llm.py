"""Single place to choose the LLM provider for all scripts.

LLM_PROVIDER=groq   -> Groq openai/gpt-oss-120b (needs GROQ_API_KEY)
LLM_PROVIDER=ollama -> local Ollama (OLLAMA_MODEL, default gpt-oss:20b)
Unset: Groq if GROQ_API_KEY is set, otherwise local Ollama.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_model(temperature: float = 0.3, timeout: int = 300):
    provider = os.getenv("LLM_PROVIDER") or ("groq" if os.getenv("GROQ_API_KEY") else "ollama")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=temperature,
            timeout=timeout,
            max_retries=3,  # retries on 429 rate limits
        )

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
        temperature=temperature,
        timeout=timeout,
    )
