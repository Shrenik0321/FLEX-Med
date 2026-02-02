"""Pydantic schemas for system configuration."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SystemConfigData(BaseModel):
    """Configuration data stored in JSONB."""
    # FL Training Configuration
    num_rounds: int = Field(default=10, ge=1, le=100, description="Number of federated learning rounds")
    local_epochs: int = Field(default=5, ge=1, le=20, description="Local training epochs per round")
    batch_size: int = Field(default=32, ge=1, le=256, description="Training batch size")
    learning_rate: float = Field(default=0.0001, ge=0.00001, le=1.0, description="Initial learning rate")
    lr_decay: float = Field(default=0.99, ge=0.8, le=1.0, description="Learning rate decay per round")
    
    # Knowledge Distillation Configuration
    distill_lr: float = Field(default=0.001, ge=0.00001, le=1.0, description="Distillation learning rate")
    distill_epochs: int = Field(default=2, ge=1, le=20, description="Distillation epochs")
    temperature: float = Field(default=3.0, ge=1.0, le=10.0, description="Softmax temperature for distillation")
    
    # Data Heterogeneity Configuration
    dirichlet_alpha: float = Field(default=1.0, ge=0.1, le=10.0, description="Dirichlet concentration parameter")
    dirichlet_seed: int = Field(default=42, ge=0, description="Random seed for Dirichlet partitioning")
    dirichlet_min_partition_size: int = Field(default=400, ge=50, description="Minimum samples per client")
    
    # Dataset Paths
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
    
    # System Configuration
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
