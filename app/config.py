from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    model_name: str = Field("gpt-3.5-turbo", alias="MODEL_NAME")
    chroma_dir: str = ".chroma"

    model_config = SettingsConfigDict(
        env_file=".env",
        protected_namespaces=("settings_",)  # tell Pydantic our prefix
    )


@lru_cache
def get_settings() -> Settings:
    """Singleton accessor so we parse .env only once."""
    return Settings()
