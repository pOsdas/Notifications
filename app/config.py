from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from django.conf import settings


class RunModel(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8010


class ApiV1Prefix(BaseModel):
    prefix: str = "/v1"


class ApiPrefix(BaseModel):
    prefix: str = "api"
    v1: ApiV1Prefix = ApiV1Prefix()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env-template", ".env"),
        case_sensitive=False,
    )
    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    pgdata: str

    allow_db_create: int
    db_create_retry_delay: int
    db_create_retries: int

    debug: bool = False
    secret_key: str
    allowed_hosts: str
    celery_broker_url: str

    api_id: str
    sms_from: str

    telegram_api: str
    bot_token: str

    email_host: str
    email_port: str
    email_use_ssl: str
    email_use_tls: str
    email_host_user: str
    email_host_password: str
    default_from_email: str

    redis_host: str
    redis_port: int
    redis_db: int

    run: RunModel = RunModel()
    api: ApiPrefix = ApiPrefix()


pydantic_settings = Settings()
# pprint(pydantic_settings.model_dump())
