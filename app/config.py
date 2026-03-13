from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    aws_region: str = "ap-northeast-2"
    s3_bucket: str = "papa-poster-bucket"
    sqs_queue_url: str = "https://sqs.ap-northeast-2.amazonaws.com/533266984381/papa-poster-jobs"
    dynamodb_table: str = "papa-poster-jobs"

    openai_api_key: str = ""
    openai_org_id: str = ""
    openai_image_model: str = "gpt-image-1"
    openai_timeout_seconds: float = 120.0
    openai_max_retries: int = 2

    bfl_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
