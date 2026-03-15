from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    aws_profile: str = ""
    aws_region: str = "ap-northeast-2"
    s3_bucket: str = "papa-poster-bucket"
    sqs_queue_url: str = "https://sqs.ap-northeast-2.amazonaws.com/533266984381/papa-poster-jobs"
    dynamodb_table: str = "papa-poster-jobs"

    openai_api_key: str = ""
    openai_org_id: str = ""
    openai_image_model: str = "gpt-image-1"
    openai_timeout_seconds: float = 120.0
    openai_max_retries: int = 2

    akool_client_id: str = ""
    akool_api_key: str = ""
    akool_face_enhance: int = 1
    akool_poll_interval_seconds: float = 2.0
    akool_poll_max_attempts: int = 30

    bfl_api_key: str = ""
    bfl_base_url: str = "https://api.bfl.ai/v1"
    bfl_flux_model: str = "flux-pro-1.1-ultra"
    bfl_image_prompt_strength: float = 0.15
    bfl_submit_timeout_seconds: float = 60.0
    bfl_submit_max_retries: int = 2
    bfl_poll_interval_seconds: float = 2.0
    bfl_poll_max_attempts: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
