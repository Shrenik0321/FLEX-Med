"""Configuration module for FLEX-Med FastAPI application."""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from supabase import create_client, Client as SupabaseClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Supabase configuration
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    
    # Application configuration
    app_name: str = "FLEX-Med API"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Orchestrator URLs
    local_train_orchestrator_url: str = os.getenv(
        "LOCAL_TRAIN_ORCHESTRATOR_URL",
        "https://intraspinal-agape-deidra.ngrok-free.dev"
    )
    federated_training_orchestrator_url: str = os.getenv(
        "FEDERATED_TRAINING_ORCHESTRATOR_URL",
        "https://40ff4102b2e7.ngrok-free.app"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_supabase_client() -> SupabaseClient:
    """
    Create and return a Supabase client instance.
    
    This is a dependency that can be injected into route handlers.
    """
    settings = get_settings()
    
    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError(
            "Supabase URL and Key must be set in environment variables. "
            "Please check your .env file."
        )
    
    return create_client(settings.supabase_url, settings.supabase_key)
