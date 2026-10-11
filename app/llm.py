"""Model factories: the single place that knows which provider is configured.

LLM_PROVIDER picks the chat model (openai, anthropic or google) and
EMBEDDING_PROVIDER picks the embeddings (openai or google). Everything else in the
project asks for "the chat model" or "the embeddings" and never names a provider.
"""

from typing import TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import settings

PROVIDER_IDS = {"openai": "openai", "anthropic": "anthropic", "google": "google_genai"}

Schema = TypeVar("Schema", bound=BaseModel)


def get_chat_model(deterministic: bool = False) -> BaseChatModel:
    provider = settings.llm_provider
    kwargs = {}
    # Current Claude models reject a non-default temperature with a 400, so
    # determinism is only requested from the providers that accept it.
    if deterministic and provider != "anthropic":
        kwargs["temperature"] = 0
    return init_chat_model(
        settings.chat_model,
        model_provider=PROVIDER_IDS[provider],
        api_key=getattr(settings, f"{provider}_api_key"),
        **kwargs,
    )


def with_structured_output(
    model: BaseChatModel, schema: type[Schema]
) -> Runnable[object, Schema]:
    # LangChain's default for Anthropic forces a tool call, which current Claude
    # models reject; native JSON schema output gives the same validated object.
    if settings.llm_provider == "anthropic":
        return model.with_structured_output(schema, method="json_schema")
    return model.with_structured_output(schema)


def get_embeddings() -> Embeddings:
    if settings.embedding_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            api_key=settings.google_api_key,
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
