from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}

PROVIDER_ALIASES = {
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "google": "gemini",
    "google_genai": "gemini",
    "googleai": "gemini",
    "open_router": "openrouter",
    "openai_compatible": "custom",
    "openai_compat": "custom",
    "gpt": "openai",
}


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""
    v = (value or "").strip().lower().replace("-", "_")
    v = PROVIDER_ALIASES.get(v, v)
    if v not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Provider {value!r} not supported. Choose one of {sorted(SUPPORTED_PROVIDERS)}")
    return v


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    # Imports are lazy so offline mode does not require every provider SDK.
    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        if provider == "custom" and not config.base_url:
            raise ValueError("Provider 'custom' requires base_url (OpenAI-compatible endpoint).")
        return ChatOpenAI(**common, api_key=config.api_key, base_url=config.base_url)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**common, api_key=config.api_key)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**common, **kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(**common, base_url=config.base_url or "http://localhost:11434")

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(**common, api_key=config.api_key)

    raise ValueError(f"Unsupported provider: {provider}")