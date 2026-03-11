from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    aws_region: str = "us-east-1"
    s3_bucket: str = "poster-engine-bucket"
    sqs_queue_url: str = "https://sqs.us-east-1.amazonaws.com/000000000000/poster-jobs"
    openai_api_key: str = ""
    bfl_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
