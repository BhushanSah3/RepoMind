"""Language-model factory with purpose-specific fallback chains."""

import logging
from typing import Any, Literal

from config.settings import Settings, get_settings
from exceptions import LLMError


logger = logging.getLogger(__name__)

LLMPurpose = Literal["generation", "evaluation", "expansion", "classification"]
LLMProviderName = Literal["gemini", "groq", "ollama"]

_FALLBACK_ORDERS: dict[LLMPurpose, tuple[LLMProviderName, ...]] = {
    "generation": ("gemini", "groq", "ollama"),
    "evaluation": ("groq", "gemini", "ollama"),
    "expansion": ("gemini", "groq", "ollama"),
    "classification": ("gemini", "groq", "ollama"),
}

_TEMPERATURES: dict[LLMPurpose, float] = {
    "generation": 0.3,
    "evaluation": 0.0,
    "expansion": 0.7,
    "classification": 0.0,
}


class LLMProvider:
    """Create LangChain chat models with the configured fallback order."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def provider_order(
        self,
        purpose: LLMPurpose,
        preferred_provider: LLMProviderName | None = None,
    ) -> tuple[LLMProviderName, ...]:
        """Return the fallback order, optionally prioritizing one provider."""

        order = _FALLBACK_ORDERS[purpose]
        if preferred_provider is None:
            return order
        return (preferred_provider, *(item for item in order if item != preferred_provider))

    def get_chat_model(
        self,
        purpose: LLMPurpose = "generation",
        preferred_provider: LLMProviderName | None = None,
    ) -> Any:
        """Return a chat model that falls back when the active provider fails."""

        if preferred_provider is None and purpose == "generation":
            preferred_provider = self.settings.default_llm_provider
        models: list[Any] = []
        active_providers: list[LLMProviderName] = []
        unavailable: list[str] = []

        for provider in self.provider_order(purpose, preferred_provider):
            try:
                models.append(self._create_chat_model(provider, purpose))
                active_providers.append(provider)
            except LLMError as error:
                unavailable.append(str(error))

        if not models:
            details = "; ".join(unavailable) or "No provider could be initialized."
            raise LLMError(f"No LLM provider is available. {details}")

        logger.info(
            "Created %s model chain with providers: %s",
            purpose,
            ", ".join(active_providers),
        )
        primary, *fallbacks = models
        return primary.with_fallbacks(fallbacks) if fallbacks else primary

    def _create_chat_model(self, provider: LLMProviderName, purpose: LLMPurpose) -> Any:
        temperature = _TEMPERATURES[purpose]

        if provider == "gemini":
            api_key = self._api_key("google_api_key", "Gemini")
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as error:
                raise LLMError("Gemini support is not installed.") from error
            return ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                google_api_key=api_key,
                temperature=temperature,
            )

        if provider == "groq":
            api_key = self._api_key("groq_api_key", "Groq")
            try:
                from langchain_groq import ChatGroq
            except ImportError as error:
                raise LLMError("Groq support is not installed.") from error
            return ChatGroq(
                model=self.settings.groq_model,
                api_key=api_key,
                temperature=temperature,
            )

        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as error:
            raise LLMError("Ollama support is not installed.") from error
        return ChatOllama(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
            temperature=temperature,
        )

    def _api_key(self, setting_name: str, provider_name: str) -> str:
        secret = getattr(self.settings, setting_name)
        if secret is None:
            raise LLMError(f"{provider_name} is not configured: set {setting_name.upper()}.")
        return secret.get_secret_value()
