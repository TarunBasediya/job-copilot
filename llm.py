"""Single place to choose the LLM provider for all scripts.

LLM_PROVIDER=groq   -> Groq openai/gpt-oss-120b (needs GROQ_API_KEY)
LLM_PROVIDER=ollama -> local Ollama (OLLAMA_MODEL, default gpt-oss:20b)
Unset: Groq if GROQ_API_KEY is set, otherwise local Ollama.
"""
import os

from deepagents import HarnessProfile, register_harness_profile

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Groq free tier mein 8K tokens/minute ki limit hai. Deep agent ke built-in
# tools ke schemas har request ke saath jaate hain, isliye jo tools hamare
# project mein kaam ke nahi, unhe model se chhupa dete hain.
GROQ_EXCLUDED_TOOLS = frozenset({"grep", "glob", "delete"})


def get_model(temperature: float = 0.3, timeout: int = 300):
    provider = os.getenv("LLM_PROVIDER") or ("groq" if os.getenv("GROQ_API_KEY") else "ollama")
    print(f"[llm] provider = {provider}")

    if provider == "groq":
        from langchain_groq import ChatGroq

        # create_deep_agent se PEHLE register hona chahiye (get_model pehle call hota hai)
        register_harness_profile("groq", HarnessProfile(excluded_tools=GROQ_EXCLUDED_TOOLS))

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