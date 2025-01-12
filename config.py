from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    openai_api_key: str
    openai_model: str
    db_url: str
    pinecone_api_key: str
    pinecone_environment: str
    aws_access_key: str
    aws_secret_key: str
    bucket_name: str


    def __init__(self, **data):
        super().__init__(**data)

def load_settings():
    try:
        return Settings()
    except ValidationError as e:
        exit(str(e))


settings = load_settings()
