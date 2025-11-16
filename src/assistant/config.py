import os
from typing import ClassVar

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from assistant.domain.article_models import FeedItem


# -----------------------------
# Supabase Settings
# -----------------------------
class SupabaseDBSettings(BaseModel):
    table_name: str = Field(default="substack_articles", description="Supabase table name")
    host: str = Field(default="localhost", description="Database host")
    name: str = Field(default="postgres", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: SecretStr = Field(default=SecretStr("password"), description="Database password")
    port: int = Field(default=6543, description="Database port")
    test_database: str = Field(default="substack_test", description="Test database name")

# -----------------------------
# Qdrant Settings
# -----------------------------
class QdrantSettings(BaseModel):
    url: str = Field(default="", description="Qdrant API URL")
    api_key: SecretStr = Field(default=SecretStr(""), description="Qdrant API Key")
    collection_name: str = Field(default="substack_collection", description="Qdrant collection name")
    dense_model_name: str = Field(default="BAAI/bge-base-en", description="Dense model name for embeddings")
    sparse_model_name: str = Field(default="Qdrant/bm25", description="Sparse model name for embeddings")
    vector_dim: int = Field(default=768, description="Dimension of the embedding vectors")
    article_batch_size: int = Field(default=4, description="Number of articles to parse in a batch")
    sparse_batch_size: int = Field(default=32, description="Batch size for sparse embeddings")
    embed_batch_size: int = Field(default=32, description="Batch size for dense embeddings")
    upsert_batch_size: int = Field(default=64, description="Batch size for upserting to Qdrant")
    max_concurrent: int = Field(default=2, description="Maximum number of concurrent tasks")

# -----------------------------
# Text Splitter Settings
# -----------------------------
class TextSplitterSettings(BaseModel):
    chunk_size: int = Field(default=2000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between text chunks")
    separator: list[str] = Field(default_factory=lambda: [ "\n---\n",
            "\n\n",
            "\n```\n",
            "\n## ",
            "\n# ",
            "\n**",
            "\n",
            ". ",
            "! ",
            "? ",
            " ",
            "",],
            description="Separators for text splitting. Order of the separators matters.")

# -----------------------------
# Jina Settings
# -----------------------------
class JinaSettings(BaseModel):
    api_key: str = Field(default="", description="Jina API key")
    url: str = Field(default="https://api.jina.ai/v1/embeddings", description="Jina API URL")
    model: str = Field(default="jina-embeddings-v3", description="Jina model name")  # 1024


# -----------------------------
# Hugging Face Settings
# -----------------------------
# BAAI/bge-large-en-v1.5 (1024), BAAI/bge-base-en-v1.5 (768)
class HuggingFaceSettings(BaseModel):
    api_key: str = Field(default="", description="Hugging Face API key")
    model: str = Field(default="BAAI/bge-base-en-v1.5", description="Hugging Face model name")


# -----------------------------
# Openai Settings
# -----------------------------
class OpenAISettings(BaseModel):
    api_key: str | None = Field(default="", description="OpenAI API key")
    # model: str = Field(default="gpt-4o-mini", description="OpenAI model name")


# -----------------------------
# OpenRouter Settings
# -----------------------------
class OpenRouterSettings(BaseModel):
    api_key: str = Field(default="", description="OpenRouter API key")
    api_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API URL")


# -----------------------------
# Opik Observability Settings
# -----------------------------
class OpikObservabilitySettings(BaseModel):
    api_key: str = Field(default="", description="Opik Observability API key")
    project_name: str = Field(default="substack-pipeline", description="Opik project name")

class RSSSettings(BaseModel):
    feeds: list[FeedItem] = Field(
        default_factory=list[FeedItem], description="List of RSS feed items"
    )
    default_start_date: str = Field(default="2025-09-15", description="Default cutoff date")
    batch_size: int = Field(
        default=5, description="Number of articles to parse and ingest in a batch"
    )

def load_yaml_feeds(yaml_path: str) -> list[FeedItem]:
    """
    Load RSS feed items from a YAML file.

    Args:
        yaml_path (str): Path to the YAML file containing feed items.
    """
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        

class Settings(BaseSettings):
    supabase_db: SupabaseDBSettings = Field(default_factory=SupabaseDBSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    rss: RSSSettings = Field(default_factory=RSSSettings)
    text_splitter: TextSplitterSettings = Field(default_factory=TextSplitterSettings)

    jina: JinaSettings = Field(default_factory=JinaSettings)
    hugging_face: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    opik: OpikObservabilitySettings = Field(default_factory=OpikObservabilitySettings)

    rss_config_yaml_path: str = r"configs/feeds_rss.yaml"

    # Pydantic v2 model config
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=[".env"],
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    @model_validator(mode="after")
    def load_yaml_rss_feeds(self) -> "Settings":
        """
        Load RSS feeds from a YAML file after model initialization.
        If the file does not exist or is empty, the feeds list remains unchanged.

        Args:
            self (Settings): The settings instance.

        Returns:
            Settings: The updated settings instance.
        """
        yaml_feeds = load_yaml_feeds(self.rss_config_yaml_path)
        if yaml_feeds:
            self.rss.feeds = yaml_feeds
        return self



settings = Settings()