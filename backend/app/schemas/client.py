from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel

class Status(str, Enum):
    Active = "Active"
    Inactive = "Inactive"

class ModelType(str, Enum):
    EfficientNetB0 = "efficientnet_b0"
    EfficientNetB1 = "efficientnet_b1"
    EfficientNetB2 = "efficientnet_b2"

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
    model_path: Optional[str] = None

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

if __name__ == "__main__":
    print("=" * 70)
    print("Client Schema Examples")
    print("=" * 70)

    # Example 1: Create new client
    print("\nExample 1: Create New Client")
    print("-" * 70)

    client_create = ClientCreate(
        client_name="Hospital_A",
        model_type=ModelType.ResNet18,
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
        model_type=ModelType.ResNet18,
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