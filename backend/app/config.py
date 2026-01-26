"""Configuration module for FLEX-Med FastAPI application."""
import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, List
from pydantic_settings import BaseSettings
from supabase import create_client, Client as SupabaseClient
from dotenv import load_dotenv

# Load environment variables from .env file
def _load_env():
    """Load environment variables from common locations."""
    current_file = Path(__file__).resolve()
    # backend/app/config.py -> backend/
    backend_root = current_file.parent.parent
    # backend/ -> project_root/
    project_root = backend_root.parent
    
    env_locations = [
        backend_root / ".env",
        project_root / ".env",
        Path.cwd() / ".env"
    ]
    
    env_loaded = False
    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path)
            env_loaded = True
            break
            
    if not env_loaded:
        load_dotenv() # Fallback to default search

_load_env()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Supabase configuration
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")

    # Application configuration
    app_name: str = "FLEX-Med API"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Path Configuration
    # Base path is the backend directory
    base_path: Path = Path(__file__).parent.parent
    
    # Federated Learning Paths
    fl_dir: Path = base_path / "federated_learning"
    project_dir: Path = fl_dir / "flex-med" # Virtual project dir structure if needed, or just fl_dir
    
    # Logs
    log_file_path: Path = fl_dir / "flex_med" / "utils" / "server_debug.log"
    simulation_log_path: Path = fl_dir / "flex_med" / "utils" / "simulation_run.log"
    
    # Data & Config Files
    client_info_file_path: Path = fl_dir / "flex_med" / "utils" / "data.json"
    data_json_path: Path = fl_dir / "cnmc_data.json"
    pyproject_path: Path = fl_dir / "pyproject.toml"
    
    # Datasets (Assume at same level as backend or configured via env)
    # Check for 'datasets' folder in backend directory first, then fallback to project root
    dataset_base_path: Path = Path(os.getenv("BASE_PATH", 
        str(Path(__file__).parent.parent if (Path(__file__).parent.parent / "datasets").exists() 
            else Path(__file__).parent.parent.parent)
    ))
    dataset_file_path: Path = dataset_base_path / "datasets" / "cnmc"
    public_anchor_dataset_path: Path = dataset_file_path / "cnmc_public_anchor"
    public_test_dataset_path: Path = dataset_file_path / "cnmc_public_test"
    
    # Model Checkpoints
    model_checkpoint_file_path: Path = dataset_base_path / "checkpoints"
    
    # Metrics & output
    round_metrics_file_path: Path = dataset_base_path / "round_metrics.json"
    graphs_output_dir: Path = dataset_base_path / "graphical_visualisation"
    
    # Simulation Runner
    run_simulation_shell_file_path: Path = fl_dir / "flex_med" / "utils" / "run_simulation.sh"
    simulation_status_json_file_path: Path = fl_dir / "flex_med" / "utils" / "simulation_status.json"

    # Orchestrator URLs
    local_train_orchestrator_url: str = os.getenv(
        "LOCAL_TRAIN_ORCHESTRATOR_URL",
        "https://intraspinal-agape-deidra.ngrok-free.dev"
    )
    federated_training_orchestrator_url: str = os.getenv(
        "FEDERATED_TRAINING_ORCHESTRATOR_URL",
        "https://40ff4102b2e7.ngrok-free.app"
    )

    # Model inference configuration
    model_path: Path = Path(os.getenv(
        "MODEL_PATH",
        str(base_path / "models" / "Durdans-mobilenet-cnmc.pt")
    ))
    device: str = os.getenv("DEVICE", "cpu")
    class_names: List[str] = ["Healthy", "ALL (Leukemia)"]
    public_data_path: str = os.getenv("PUBLIC_DATA_PATH", "")
    
    # FL Model Parameters
    num_classes: int = 2
    img_size: int = 224
    consensus_momentum: float = 0.5
    train_loss_weight: float = 0.7
    distill_loss_weight: float = 0.3
    
    # Model Suitability Scores
    model_suitability_scores: Dict[str, float] = {
        # Tier 1
        'resnet18': 1.10, 'resnet34': 1.10, 'resnet50': 1.10, 'resnet101': 1.10, 'resnet152': 1.10,
        'densenet121': 1.10, 'densenet161': 1.10, 'densenet169': 1.10, 'densenet201': 1.10,
        # Tier 2
        'efficientnet_b0': 1.05, 'efficientnet_b1': 1.05, 'efficientnet_b2': 1.05,
        'efficientnet_b3': 1.05, 'efficientnet_b4': 1.05, 'efficientnet_b5': 1.05,
        'efficientnet_b6': 1.05, 'efficientnet_b7': 1.05,
        'inception_v3': 1.05, 'googlenet': 1.05,
        # Tier 3
        'vgg16': 1.00, 'vgg19': 1.00,
        # Tier 4
        'mobilenet_v2': 0.95, 'mobilenet_v3_small': 0.90, 'mobilenet_v3_large': 0.95,
        'squeezenet': 0.90,
        # Tier 5
        'alexnet': 0.90, 'vgg11': 0.95, 'vgg13': 0.95,
    }

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "protected_namespaces": ()
    }


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
