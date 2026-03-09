"""Pydantic schemas for system configuration."""
from pydantic import BaseModel, Field
from typing import Dict
from datetime import datetime


class SystemConfigData(BaseModel):
    """Configuration data stored in JSONB.
    
    Tuning parameters (minority_boost, focal_alpha, focal_gamma, etc.) are
    top-level fields — constant across all heterogeneity presets.
    
    Presets only contain dirichlet_alpha (the heterogeneity control).
    The active preset is identified by the `heterogeneity_preset` field.
    """
    # Active Heterogeneity Preset
    heterogeneity_preset: str = Field(
        default="moderate",
        description="Heterogeneity preset level: low, moderate, high"
    )

    # Heterogeneity Presets — only dirichlet_alpha per preset
    presets: Dict[str, dict] = Field(
        default_factory=dict,
        description="Heterogeneity preset configurations keyed by name (each contains only dirichlet_alpha)"
    )

    # Training Strategy Parameters (global — constant across presets)
    minority_boost: float = Field(default=0.78, ge=0.0, le=2.0, description="Minority class boost for WeightedRandomSampler")
    focal_alpha: float = Field(default=0.50, ge=0.0, le=1.0, description="Focal loss alpha (fallback; dynamic alpha used at runtime)")
    focal_gamma: float = Field(default=2.0, ge=0.0, le=5.0, description="Focal loss gamma")
    consensus_momentum: float = Field(default=0.35, ge=0.0, le=1.0, description="Consensus momentum smoothing")
    distill_weight_base: float = Field(default=0.80, ge=0.0, le=1.0, description="Knowledge distillation weight base")
    distill_decay_rate: float = Field(default=0.18, ge=0.0, le=1.0, description="Distillation weight decay rate")
    train_loss_weight: float = Field(default=0.72, ge=0.0, le=1.0, description="Training loss weight in combined loss")
    distill_loss_weight: float = Field(default=0.28, ge=0.0, le=1.0, description="Distillation loss weight in combined loss")
    lr_decay: float = Field(default=0.90, ge=0.5, le=1.0, description="Learning rate decay per round")
    learning_rate: float = Field(default=0.0005, ge=0.00001, le=1.0, description="Initial learning rate")

    # FL Training Configuration (global)
    num_rounds: int = Field(default=10, ge=1, le=100, description="Number of federated learning rounds")
    local_epochs: int = Field(default=1, ge=1, le=20, description="Local training epochs per round")
    batch_size: int = Field(default=32, ge=1, le=256, description="Training batch size")

    # Knowledge Distillation Configuration (global)
    distill_lr: float = Field(default=0.0005, ge=0.00001, le=1.0, description="Distillation learning rate")
    distill_epochs: int = Field(default=1, ge=1, le=20, description="Distillation epochs")
    temperature: float = Field(default=4.0, ge=1.0, le=10.0, description="Softmax temperature for distillation")

    # Dirichlet Partitioning (global — seed and min size don't vary by preset)
    dirichlet_seed: int = Field(default=42, ge=0, description="Random seed for Dirichlet partitioning")
    dirichlet_min_partition_size: int = Field(default=400, ge=50, description="Minimum samples per client")

    # Dataset Paths (global)
    public_anchor_dataset_path: str = Field(
        default="/content/datasets/cnmc/cnmc_public_anchor",
        description="Path to public anchor dataset for distillation"
    )
    public_test_dataset_path: str = Field(
        default="/content/datasets/cnmc/cnmc_public_test",
        description="Path to public test dataset for evaluation"
    )
    local_train_dataset_path: str = Field(
        default="/content/datasets/cnmc/cnmc_local_train",
        description="Path to local training dataset for Dirichlet partitioning"
    )

    # System Configuration (global)
    ngrok_url: str = Field(
        default="https://eb474f08357f.ngrok-free.app",
        description="NGROK URL for federated orchestrator"
    )

    def get_active_preset_config(self) -> dict:
        """Resolve the active preset config values.

        Returns the preset dict (dirichlet_alpha) merged with top-level tuning params.
        Falls back to the 'moderate' preset if the active key is not found.
        """
        preset = self.presets.get(self.heterogeneity_preset)
        if preset is None:
            preset = self.presets.get("moderate", {})
        # Merge top-level tuning params with preset-specific dirichlet_alpha
        return {
            **preset,
            "minority_boost": self.minority_boost,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "consensus_momentum": self.consensus_momentum,
            "distill_weight_base": self.distill_weight_base,
            "distill_decay_rate": self.distill_decay_rate,
            "train_loss_weight": self.train_loss_weight,
            "distill_loss_weight": self.distill_loss_weight,
            "lr_decay": self.lr_decay,
            "learning_rate": self.learning_rate,
        }


class SystemConfig(BaseModel):
    """System configuration database model."""
    id: int
    config: SystemConfigData
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SystemConfigUpdate(BaseModel):
    """Update model for system configuration."""
    config: SystemConfigData
