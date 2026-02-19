"""Pydantic schemas for system configuration."""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


# Default presets (fallback if not found in database)
DEFAULT_HETEROGENEITY_PRESETS: Dict[str, dict] = {
    "low": {
        "dirichlet_alpha": 5.0,
        "minority_boost": 1.0,
        "focal_alpha": 0.50,
        "focal_gamma": 2.0,
        "consensus_momentum": 0.10,
        "distill_weight_base": 0.55,
        "distill_decay_rate": 0.30,
        "train_loss_weight": 0.65,
        "distill_loss_weight": 0.35,
        "lr_decay": 0.92,
        "learning_rate": 0.001,
    },
    "moderate": {
        "dirichlet_alpha": 2.5,
        "minority_boost": 0.70,
        "focal_alpha": 0.50,
        "focal_gamma": 2.0,
        "consensus_momentum": 0.40,
        "distill_weight_base": 0.45,
        "distill_decay_rate": 0.20,
        "train_loss_weight": 0.70,
        "distill_loss_weight": 0.30,
        "lr_decay": 0.95,
        "learning_rate": 0.001,
    },
    "high": {
        "dirichlet_alpha": 1.0,
        "minority_boost": 0.65,
        "focal_alpha": 0.50,
        "focal_gamma": 2.5,
        "consensus_momentum": 0.30,
        "distill_weight_base": 0.35,
        "distill_decay_rate": 0.15,
        "train_loss_weight": 0.75,
        "distill_loss_weight": 0.25,
        "lr_decay": 0.97,
        "learning_rate": 0.001,
    },
}


class SystemConfigData(BaseModel):
    """Configuration data stored in JSONB.
    
    Preset-specific fields (lr_decay, learning_rate, dirichlet_alpha,
    minority_boost, focal_alpha, focal_gamma, consensus_momentum,
    distill_weight_base, distill_decay_rate, train_loss_weight,
    distill_loss_weight) live exclusively inside the preset objects.
    
    The active preset is identified by the `heterogeneity_preset` field.
    """
    # Active Heterogeneity Preset
    heterogeneity_preset: str = Field(
        default="moderate",
        description="Heterogeneity preset level: low, moderate, high, or custom"
    )

    # Heterogeneity Presets (stored in DB, editable via API)
    presets: Dict[str, dict] = Field(
        default=DEFAULT_HETEROGENEITY_PRESETS,
        description="Heterogeneity preset configurations keyed by name (low, moderate, high, custom)"
    )

    # FL Training Configuration (global — not preset-specific)
    num_rounds: int = Field(default=10, ge=1, le=100, description="Number of federated learning rounds")
    local_epochs: int = Field(default=5, ge=1, le=20, description="Local training epochs per round")
    batch_size: int = Field(default=32, ge=1, le=256, description="Training batch size")

    # Knowledge Distillation Configuration (global)
    distill_lr: float = Field(default=0.001, ge=0.00001, le=1.0, description="Distillation learning rate")
    distill_epochs: int = Field(default=2, ge=1, le=20, description="Distillation epochs")
    temperature: float = Field(default=3.0, ge=1.0, le=10.0, description="Softmax temperature for distillation")

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
        
        Returns the preset dict for the active heterogeneity_preset.
        Falls back to the 'moderate' preset if the preset key is not found.
        """
        preset = self.presets.get(self.heterogeneity_preset)
        if preset is not None:
            return preset
        # Fallback to moderate
        return self.presets.get("moderate", DEFAULT_HETEROGENEITY_PRESETS["moderate"])


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
