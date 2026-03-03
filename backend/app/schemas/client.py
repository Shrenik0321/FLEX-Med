from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel

class Status(str, Enum):
    Active = "Active"
    Inactive = "Inactive"

class ModelType(str, Enum):
    # The "Big Four" Architectures for FLEX-Med
    ResNet50 = "resnet50"             # The Industry Standard
    MobileNetV2 = "mobilenet_v2"       # Mobile Optimized
    DenseNet121 = "densenet121"       # High Dense Connections

# ==========================================
# SCHEMAS FOR API (FastAPI Compatible)
# ==========================================
#
# NOTE: As of v3.0, all clients use runtime Dirichlet partitioning
# with shared LOCAL_TRAIN_DATASET_PATH configured in backend/app/config.py.
# Data heterogeneity is controlled by the DIRICHLET_ALPHA parameter.
# See backend/federated_learning/flex_med/task.py for implementation.
#
# ==========================================

class ClientBase(BaseModel):
    """Base schema for client (static configuration only)"""
    client_name: str
    status: Status
    model_type: ModelType
    model_path: str

    model_config = {
        "protected_namespaces": ()
    }

class ClientCreate(BaseModel):
    """Schema for creating new clients"""
    client_name: str
    model_type: ModelType
    status: Status = Status.Inactive
    model_path: Optional[str] = None  # Will be set by route handler using config

    model_config = {
        "protected_namespaces": ()
    }

class Client(ClientBase):
    """Complete client schema with ID"""
    id: int
    created_at: str  # ISO format datetime string
    
    model_config = {
        "from_attributes": True
    }

# ==========================================
# NOTE: Metrics are now stored in client_simulation_metrics table
# ==========================================
#
# Client metrics are no longer stored in this table.
# For FL simulation metrics, see:
# - backend/app/schemas/client_simulation_metrics.py
# - backend/app/routes/client_simulation_metrics.py
#

# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("Client Schema Examples")
    print("=" * 70)

    # Example 1: Create new client
    print("\nExample 1: Create New Client")
    print("-" * 70)

    client_create = ClientCreate(
        client_name="Hospital_A",
        model_type=ModelType.ResNet50,
        status=Status.Active,
        model_path="/path/to/model.pt"
    )

    print(f"Client Name: {client_create.client_name}")
    print(f"Model Type: {client_create.model_type}")

    # Example 2: Complete client object
    print("\nExample 2: Complete Client Object")
    print("-" * 70)

    complete_client = Client(
        id=1,
        client_name="Hospital_A",
        status=Status.Active,
        model_type=ModelType.ResNet50,
        model_path="/path/to/model.pt",
        created_at=datetime.now().isoformat()
    )

    print(f"Client ID: {complete_client.id}")
    print(f"Created At: {complete_client.created_at}")
    print(f"\nComplete Client JSON:")
    print(complete_client.model_dump_json(indent=2))

    print("\n" + "=" * 70)
    print("NOTE: Client metrics are stored in client_simulation_metrics table")
    print("=" * 70)