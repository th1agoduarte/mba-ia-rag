from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

LlmProvider = Literal["openai", "anthropic", "google"]
# Anthropic has no embeddings API, so it can only be chosen for chat.
EmbeddingProvider = Literal["openai", "google"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    llm_provider: LlmProvider = Field("openai", alias="LLM_PROVIDER")
    embedding_provider: EmbeddingProvider = Field("openai", alias="EMBEDDING_PROVIDER")

    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field("gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    anthropic_chat_model: str = Field("claude-opus-5-5", alias="ANTHROPIC_CHAT_MODEL")

    google_api_key: str | None = Field(None, alias="GOOGLE_API_KEY")
    google_chat_model: str = Field("gemini-3.8-flash", alias="GOOGLE_CHAT_MODEL")
    google_embedding_model: str = Field(
        "gemini-embedding-001", alias="GOOGLE_EMBEDDING_MODEL"
    )

    database_url: str = Field(..., alias="DATABASE_URL")

    @model_validator(mode="after")
    def require_selected_keys(self) -> "Settings":
        # Only the providers actually selected need a key, so an OpenAI-only
        # setup does not have to fill in Anthropic or Google credentials.
        for provider in {self.llm_provider, self.embedding_provider}:
            if not getattr(self, f"{provider}_api_key"):
                raise ValueError(
                    f"{provider.upper()}_API_KEY is required when "
                    f"LLM_PROVIDER or EMBEDDING_PROVIDER is '{provider}'"
                )
        return self

    @property
    def chat_model(self) -> str:
        return getattr(self, f"{self.llm_provider}_chat_model")

    @property
    def embedding_model(self) -> str:
        return getattr(self, f"{self.embedding_provider}_embedding_model")

    @property
    def embedding_id(self) -> str:
        # Vectors from different embedding models are not comparable, so the
        # index remembers which one produced them.
        return f"{self.embedding_provider}:{self.embedding_model}"

    @property
    def psycopg_dsn(self) -> str:
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")


settings = Settings()
