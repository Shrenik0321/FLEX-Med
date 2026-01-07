from datetime import datetime
from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, field_validator
import json

class Status(str, Enum):
    Active = "Active"
    Inactive = "Inactive"

class ModelType(str, Enum):
    # ResNet
    ResNet18 = "resnet18"
    ResNet34 = "resnet34"
    ResNet50 = "resnet50"
    ResNet101 = "resnet101"
    ResNet152 = "resnet152"
    
    # VGG
    VGG11 = "vgg11"
    VGG13 = "vgg13"
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    
    # DenseNet
    DenseNet121 = "densenet121"
    DenseNet161 = "densenet161"
    DenseNet169 = "densenet169"
    DenseNet201 = "densenet201"
    
    # EfficientNet
    EfficientNetB0 = "efficientnet_b0"
    EfficientNetB1 = "efficientnet_b1"
    EfficientNetB2 = "efficientnet_b2"
    EfficientNetB3 = "efficientnet_b3"
    EfficientNetB4 = "efficientnet_b4"
    EfficientNetB5 = "efficientnet_b5"
    EfficientNetB6 = "efficientnet_b6"
    EfficientNetB7 = "efficientnet_b7"
    
    # MobileNet
    MobileNetV2 = "mobilenet_v2"
    MobileNetV3Small = "mobilenet_v3_small"
    MobileNetV3Large = "mobilenet_v3_large"
    
    # Inception
    InceptionV3 = "inception_v3"
    GoogLeNet = "googlenet"
    
    # Other
    AlexNet = "alexnet"
    SqueezeNet = "squeezenet"

# ==========================================
# SCHEMAS FOR API (FastAPI Compatible)
# ==========================================

class ClientBase(BaseModel):
    """Base schema with metrics as string (for FastAPI)"""
    client_name: str
    status: Status
    model_type: ModelType
    has_local_data: bool
    dataset_path: Optional[str] = None
    model_path: str
    metrics: str  # ← String to store JSON
    
    @field_validator('metrics', mode='before')
    @classmethod
    def ensure_metrics_is_string(cls, v):
        """
        Ensure metrics is always a simple JSON string (not double-encoded).
        
        Handles:
        - None → "{}"
        - "" → "{}"
        - dict → JSON string
        - "\"{}\\"" (double-encoded) → "{}"
        - valid JSON string → pass through
        """
        # Handle None or empty
        if v is None or v == "":
            return "{}"
        
        # Handle dict - convert to JSON string
        if isinstance(v, dict):
            return json.dumps(v)
        
        # Handle string
        if isinstance(v, str):
            v = v.strip()
            
            # Empty string
            if not v:
                return "{}"
            
            # Check for double-encoding: "\"{}\""
            if v.startswith('"') and v.endswith('"'):
                try:
                    # Try to parse once to remove outer quotes
                    inner = json.loads(v)
                    # If inner is a string, we had double-encoding
                    if isinstance(inner, str):
                        # Validate it's proper JSON
                        json.loads(inner)  # This will raise if invalid
                        return inner
                    else:
                        # Inner was already a dict, re-encode properly
                        return json.dumps(inner)
                except json.JSONDecodeError:
                    # Not valid JSON, just remove quotes
                    return v.strip('"') if v.strip('"') else "{}"
            
            # Regular JSON string - validate it
            try:
                json.loads(v)  # Validate
                return v
            except json.JSONDecodeError:
                # Invalid JSON, return empty object
                return "{}"
        
        # Unknown type, default to empty object
        return "{}"

class ClientCreate(BaseModel):
    """Schema for creating new clients"""
    client_name: str
    model_type: ModelType
    status: Status = Status.Inactive
    has_local_data: bool = False
    dataset_path: Optional[str] = "/content/drive/MyDrive/College/FLEX-Med/datasets/all_idb2_raw"
    model_path: Optional[str] = None
    metrics: str = "{}"  # ← Default to empty JSON object
    
    @field_validator('metrics', mode='before')
    @classmethod
    def ensure_metrics_is_string(cls, v):
        """Same validation as ClientBase"""
        if v is None or v == "":
            return "{}"
        
        if isinstance(v, dict):
            return json.dumps(v)
        
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return "{}"
            
            # Handle double-encoding
            if v.startswith('"') and v.endswith('"'):
                try:
                    inner = json.loads(v)
                    if isinstance(inner, str):
                        json.loads(inner)
                        return inner
                    else:
                        return json.dumps(inner)
                except json.JSONDecodeError:
                    return v.strip('"') if v.strip('"') else "{}"
            
            try:
                json.loads(v)
                return v
            except json.JSONDecodeError:
                return "{}"
        
        return "{}"
    
    def model_post_init(self, __context):
        """Set default model_path if not provided"""
        if self.model_path is None:
            self.model_path = f"/content/drive/MyDrive/College/models/{self.client_name}.pt"

class Client(ClientBase):
    """Complete client schema with ID"""
    id: int
    created_at: str  # ISO format datetime string
    
    class Config:
        from_attributes = True  # Updated from orm_mode in Pydantic v2

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_empty_metrics() -> str:
    """
    Create empty metrics as a simple JSON string.
    Use this when enrolling new clients.
    
    Returns:
        Simple JSON string: "{}"
    """
    return "{}"

def parse_metrics(metrics_str: str) -> dict:
    """
    Safely parse metrics JSON string to dict.
    
    Args:
        metrics_str: JSON string (may be empty, malformed, or double-encoded)
    
    Returns:
        Dict containing metrics, or empty dict if parsing fails
    """
    if not metrics_str or metrics_str.strip() == "":
        return {}
    
    try:
        # Try direct parse
        result = json.loads(metrics_str)
        
        # If result is a string (double-encoded), parse again
        if isinstance(result, str):
            result = json.loads(result)
        
        return result if isinstance(result, dict) else {}
        
    except json.JSONDecodeError:
        return {}

def update_metrics(metrics_str: str, **updates) -> str:
    """
    Update specific fields in metrics.
    
    Args:
        metrics_str: Current metrics as JSON string
        **updates: Key-value pairs to update
    
    Returns:
        Updated metrics as JSON string
    
    Example:
        new_metrics = update_metrics(
            client.metrics,
            pre_fl={'accuracy': 0.65},
            post_fl={'accuracy': 0.82}
        )
    """
    metrics = parse_metrics(metrics_str)
    metrics.update(updates)
    return json.dumps(metrics)

# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    
    # Example 1: Create client with default metrics
    print("=" * 70)
    print("Example 1: Create Client with Default Empty Metrics")
    print("=" * 70)
    
    client1 = ClientCreate(
        client_name="Hospital_A",
        model_type=ModelType.ResNet18,
        status=Status.Active,
        has_local_data=True,
        dataset_path="/path/to/data"
    )
    
    print(f"Client Name: {client1.client_name}")
    print(f"Metrics: {client1.metrics}")
    print(f"Type: {type(client1.metrics)}")
    print(f"Is valid JSON: {bool(parse_metrics(client1.metrics) is not None)}")
    
    # Example 2: Create client with metrics as dict (auto-converts)
    print("\n" + "=" * 70)
    print("Example 2: Create Client with Dict (Auto-converts to String)")
    print("=" * 70)
    
    client2 = ClientCreate(
        client_name="Hospital_B",
        model_type=ModelType.MobileNetV2,
        status=Status.Active,
        has_local_data=True,
        metrics={  # ← Sending as dict
            "pre_fl": {"accuracy": 0.65},
            "post_fl": {"accuracy": 0.82}
        }
    )
    
    print(f"Client Name: {client2.client_name}")
    print(f"Metrics: {client2.metrics}")
    print(f"Type: {type(client2.metrics)}")
    
    # Example 3: Handle double-encoded string (common bug)
    print("\n" + "=" * 70)
    print("Example 3: Handle Double-Encoded String")
    print("=" * 70)
    
    # Simulate double-encoded metrics from database
    double_encoded = '"{}"'  # This is what you were getting
    
    client3 = ClientCreate(
        client_name="Hospital_C",
        model_type=ModelType.DenseNet121,
        status=Status.Active,
        has_local_data=False,
        metrics=double_encoded  # ← Double-encoded input
    )
    
    print(f"Input (double-encoded): {double_encoded}")
    print(f"Output (cleaned): {client3.metrics}")
    print(f"Type: {type(client3.metrics)}")
    
    # Example 4: Parse and update metrics
    print("\n" + "=" * 70)
    print("Example 4: Parse and Update Metrics")
    print("=" * 70)
    
    # Start with empty
    current_metrics = create_empty_metrics()
    print(f"Initial: {current_metrics}")
    
    # Add pre-FL metrics
    current_metrics = update_metrics(
        current_metrics,
        pre_fl={'accuracy': 0.65, 'f1': 0.62}
    )
    print(f"After pre-FL: {current_metrics}")
    
    # Add post-FL metrics
    current_metrics = update_metrics(
        current_metrics,
        post_fl={'accuracy': 0.82, 'f1': 0.80}
    )
    print(f"After post-FL: {current_metrics}")
    
    # Parse to read
    parsed = parse_metrics(current_metrics)
    print(f"Parsed: {parsed}")
    print(f"Pre-FL Accuracy: {parsed.get('pre_fl', {}).get('accuracy', 'N/A')}")
    print(f"Post-FL Accuracy: {parsed.get('post_fl', {}).get('accuracy', 'N/A')}")
    
    # Example 5: Complete client for database
    print("\n" + "=" * 70)
    print("Example 5: Complete Client Object")
    print("=" * 70)
    
    complete_client = Client(
        id=1,
        client_name="Hospital_A",
        status=Status.Active,
        model_type=ModelType.ResNet18,
        has_local_data=True,
        dataset_path="/path/to/data",
        model_path="/path/to/model.pt",
        metrics=create_empty_metrics(),
        created_at=datetime.now().isoformat()
    )
    
    print(f"Complete Client JSON:")
    print(complete_client.model_dump_json(indent=2))