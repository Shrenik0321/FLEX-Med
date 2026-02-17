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

    # ========================================================================
    # Path Configuration
    # ========================================================================
    # All paths can be overridden via environment variables for flexibility

    # Base paths
    # backend_root: The backend/ directory (where this config.py is located)
    backend_root: Path = Path(__file__).parent.parent

    # base_path: Root for datasets, models, checkpoints
    # Default: backend/ directory (or parent if backend/datasets doesn't exist)
    # Override with BASE_PATH environment variable
    @property
    def _default_base_path(self) -> Path:
        """Calculate default base path based on directory structure."""
        if (self.backend_root / "datasets").exists():
            return self.backend_root
        else:
            # If datasets not in backend/, use project root (backend's parent)
            return self.backend_root.parent

    base_path: Path = Path(os.getenv("BASE_PATH", ""))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set base_path to default if not provided via env
        if not self.base_path or str(self.base_path) == ".":
            self.base_path = self._default_base_path

    # ------------------------------------------------------------------------
    # Federated Learning Directory Structure
    # ------------------------------------------------------------------------
    fl_dir: Path = backend_root / "federated_learning"
    project_dir: Path = fl_dir

    # Logs
    simulation_log_path: Path = fl_dir / "flex_med" / "utils" / "simulation_run.log"

    # Data & Config Files
    client_info_file_path: Path = fl_dir / "flex_med" / "utils" / "client_data.json"
    pyproject_path: Path = fl_dir / "pyproject.toml"

    # Simulation Runner
    run_simulation_shell_file_path: Path = fl_dir / "flex_med" / "utils" / "run_simulation.sh"

    # ------------------------------------------------------------------------
    # Datasets (with environment variable overrides)
    # ------------------------------------------------------------------------
    @property
    def datasets_path(self) -> Path:
        """Root datasets directory - can be overridden via DATASETS_PATH env var."""
        return Path(os.getenv("DATASETS_PATH", str(self.base_path / "datasets")))

    @property
    def dataset_file_path(self) -> Path:
        """CNMC dataset directory."""
        return self.datasets_path / "cnmc"

    @property
    def public_anchor_dataset_path(self) -> Path:
        """Public anchor dataset for federated learning distillation."""
        env_path = os.getenv("PUBLIC_ANCHOR_DATASET_PATH")
        if env_path:
            return Path(env_path)
        return self.dataset_file_path / "cnmc_public_anchor"

    @property
    def public_test_dataset_path(self) -> Path:
        """Public test dataset for evaluation."""
        env_path = os.getenv("PUBLIC_TEST_DATASET_PATH")
        if env_path:
            return Path(env_path)
        return self.dataset_file_path / "cnmc_public_test"

    @property
    def local_train_dataset_path(self) -> Path:
        """
        Shared local training dataset (70% of data) for runtime Dirichlet partitioning.
        This dataset is partitioned in-memory across clients during FL initialization.
        """
        env_path = os.getenv("LOCAL_TRAIN_DATASET_PATH")
        if env_path:
            return Path(env_path)
        return self.dataset_file_path / "cnmc_local_train"

    # ------------------------------------------------------------------------
    # Models and Checkpoints (with environment variable overrides)
    # ------------------------------------------------------------------------
    @property
    def models_path(self) -> Path:
        """Root models directory - can be overridden via MODELS_PATH env var."""
        return Path(os.getenv("MODELS_PATH", str(self.base_path / "models")))

    @property
    def model_checkpoint_file_path(self) -> Path:
        """Checkpoints directory for FL intermediate checkpoints."""
        return Path(os.getenv("CHECKPOINTS_PATH", str(self.base_path / "checkpoints")))

    @property
    def default_model_dir(self) -> Path:
        """Default directory for new client models."""
        return Path(os.getenv("DEFAULT_MODEL_DIR", str(self.models_path)))

    # ------------------------------------------------------------------------
    # Metrics & Output
    # ------------------------------------------------------------------------
    @property
    def graphs_output_dir(self) -> Path:
        """Directory for graphical visualizations."""
        env_path = os.getenv("GRAPHS_OUTPUT_DIR")
        if env_path:
            return Path(env_path)
        return self.base_path / "graphical_visualisation"

    # ------------------------------------------------------------------------
    # Orchestrator URLs (no hardcoded defaults - must be set via env or will use localhost)
    # ------------------------------------------------------------------------
    federated_training_orchestrator_url: str = os.getenv(
        "FEDERATED_TRAINING_ORCHESTRATOR_URL",
        "https://eb474f08357f.ngrok-free.app"
    )

    # ------------------------------------------------------------------------
    # Model Inference Configuration
    # ------------------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        """Path to the model used for prediction endpoints."""
        env_path = os.getenv("MODEL_PATH")
        if env_path:
            return Path(env_path)
        return self.models_path / "Durdans-mobilenet-cnmc.pt"

    device: str = os.getenv("DEVICE", "cpu")
    class_names: List[str] = ["ALL (Leukemia)", "Healthy"]
    public_data_path: str = os.getenv("PUBLIC_DATA_PATH", "")
    
    # FL Model Parameters
    num_classes: int = 2
    img_size: int = 224
    consensus_momentum: float = 0.20
    train_loss_weight: float = 0.70
    distill_loss_weight: float = 0.3

    # Per-architecture Focal Loss alpha is now computed dynamically per client
    # fohcal_alpha_per_arch removed in favor of data-driven approach
    focal_alpha_default: float = 0.50
    focal_gamma: float = 2.0
    
    # Minority Class Boost (for Focal Loss clamping/adjustment)
    minority_boost: float = 0.70

    # Knowledge distillation configuration
    # Lowered to 0.45 so clients learn more from local data while still benefiting from consensus
    distill_weight_base: float = float(os.getenv("FLEX_MED_DISTILL_WEIGHT", "0.45"))
    distill_decay_rate: float = float(os.getenv("FLEX_MED_DISTILL_DECAY", "0.2"))  # Slower decay

    # ------------------------------------------------------------------------
    # Dirichlet Partitioning Configuration (Runtime Data Heterogeneity)
    # ------------------------------------------------------------------------
    dirichlet_alpha: float = float(os.getenv("FLEX_MED_DIRICHLET_ALPHA", "2.5"))
    dirichlet_seed: int = int(os.getenv("FLEX_MED_DIRICHLET_SEED", "42"))
    dirichlet_min_partition_size: int = int(os.getenv("FLEX_MED_DIRICHLET_MIN_PARTITION_SIZE", "400"))  # Minimum samples per client partition

    # Normalization statistics (must match training in task.py)
    img_norm_mean: List[float] = [
        float(x) for x in os.getenv("IMG_NORM_MEAN", "0.485,0.456,0.406").split(",")
    ]
    img_norm_std: List[float] = [
        float(x) for x in os.getenv("IMG_NORM_STD", "0.229,0.224,0.225").split(",")
    ]

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
