"""
Shared LLM API client for data synthesis scripts.
Set LLM_PROVIDER=openai to use OpenAI-compatible API, otherwise defaults to Gemini.
"""

import os
import requests

# ---- Provider selection ----
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()

# Gemini-specific env vars
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://example.googleapis.com/v1:generateContent")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# OpenAI-specific env vars
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")


def _get_api_key():
    """Raise if the selected provider's API key is not set."""
    if LLM_PROVIDER == "openai":
        key = OPENAI_API_KEY
        if not key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set in environment or .env")
    else:
        key = GEMINI_API_KEY
        if not key:
            raise ValueError("Please set GEMINI_API_KEY in environment or .env before running.")
    return key


def request_llm(prompt, temperature=0.85, max_tokens=65535, top_p=None, return_raw=False):
    """
    Send a prompt to the configured LLM provider.

    Args:
        prompt: The text prompt.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        top_p: Nucleus sampling (OpenAI only; Gemini uses topP in generationConfig).
        return_raw: If True, return the raw response dict (for callers that
                    parse the Gemini response themselves).

    Returns:
        - If return_raw=True: the raw response dict from the API.
        - If return_raw=False: the extracted text string, or None on failure.
    """
    api_key = _get_api_key()

    if LLM_PROVIDER == "openai":
        return _call_openai(prompt, api_key, temperature, max_tokens, top_p, return_raw)
    else:
        return _call_gemini(prompt, api_key, temperature, max_tokens, top_p, return_raw)


# ---- Gemini backend ----

def _call_gemini(prompt, api_key, temperature, max_tokens, top_p, return_raw):
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    gen_config = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if top_p is not None:
        gen_config["topP"] = top_p

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }

    try:
        resp = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=240)
        if resp.status_code == 200:
            data = resp.json()
            if return_raw:
                return data
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        pass
    return None if not return_raw else {"error": "request failed"}


# ---- OpenAI backend ----

def _call_openai(prompt, api_key, temperature, max_tokens, top_p, return_raw):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p

    try:
        resp = requests.post(OPENAI_ENDPOINT, headers=headers, json=payload, timeout=240)
        if resp.status_code == 200:
            data = resp.json()
            if return_raw:
                return data
            return data["choices"][0]["message"]["content"]
    except Exception:
        pass
    return None if not return_raw else {"error": "request failed"}
