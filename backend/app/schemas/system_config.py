"""Pydantic schemas for system configuration."""
from pydantic import BaseModel, Field
from typing import Dict
from datetime import datetime


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
        default_factory=dict,
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
        Falls back to the 'moderate' preset if the active key is not found.
        """
        preset = self.presets.get(self.heterogeneity_preset)
        if preset is not None:
            return preset
        # Fallback to moderate preset from DB
        return self.presets.get("moderate", {})


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
