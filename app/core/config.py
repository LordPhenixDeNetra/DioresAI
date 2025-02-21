from pydantic_settings import BaseSettings
from functools import lru_cache


class Config(BaseSettings):  # Renommé de Settings à Config
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    PROJECT_NAME: str = "DioresAI API"

    class Config:
        env_file = ".env"


@lru_cache()
def get_config():  # Renommé de get_settings à get_config
    return Config()
