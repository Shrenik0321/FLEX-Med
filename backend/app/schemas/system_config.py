"""Pydantic schemas for system configuration."""
from pydantic import BaseModel, Field
from datetime import datetime


class SystemConfigData(BaseModel):
    """Configuration data stored in JSONB.

    Single configuration — no presets. dirichlet_alpha is a top-level field
    that the user can set directly.
    """
    # Dirichlet Alpha (data heterogeneity control)
    dirichlet_alpha: float = Field(default=1.5, ge=0.01, le=100.0, description="Dirichlet alpha for data partitioning (higher = more uniform)")

    # Training Strategy Parameters (global)
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

    # Dirichlet Partitioning (global)
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
