from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from supabase import create_client, Client

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_path: Path = PROJECT_ROOT / "models" / "model_client_0.pt"
    public_data_path: Optional[Path] = None
    device: str = "cuda"
    class_names: List[str] = ["ALL (Leukemia)", "Hem (Healthy)"]

    supabase_url: str
    supabase_key: str

    # Orchestrator URLs (ngrok tunnels from Colab)
    federated_training_orchestrator_url: str = Field(
        default="http://localhost:8001",
        description="URL for federated learning orchestrator (ngrok tunnel)"
    )
    inference_orchestrator_url: str = Field(
        default="http://localhost:8002",
        description="URL for inference orchestrator with xAI (ngrok tunnel)"
    )

    class Config:
        case_sensitive = False
        env_file = ".env"  # <-- important


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@lru_cache()
def get_supabase_client() -> Client:
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)
