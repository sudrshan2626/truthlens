from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic automatically reads from .env file.
    """

    #App metadata
    app_name: str = "TruthLens"
    app_version: str = "1.0.0"
    debug: bool = True

    #Gemini API
    gemini_api_key: str = ""


    #CORS
    frontend_url: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """
    Returns cahched settings instance. 
    lru_cache ensures this is only created once (singleton pattern).
    """
    return Settings()