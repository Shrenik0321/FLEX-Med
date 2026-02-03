from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_validator, ConfigDict


class ClientSimulationStatus(str, Enum):
    """Status of a client within a specific FL simulation"""
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class ClientSimulationMetricsBase(BaseModel):
    """Base schema for client simulation metrics"""
    simulation_id: int
    client_id: int
    metrics: dict = {}  # JSONB stored as dict
    status: ClientSimulationStatus = ClientSimulationStatus.PENDING
    error_message: Optional[str] = None

    @field_validator('metrics', mode='before')
    @classmethod
    def ensure_metrics_is_dict(cls, v):
        """Ensure metrics is a dict (not string-encoded JSON)"""
        if v is None or v == "":
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            # In case Supabase returns string (shouldn't happen with JSONB)
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return {}


class ClientSimulationMetricsCreate(BaseModel):
    """Schema for creating new client simulation metrics record"""
    simulation_id: int
    client_id: int
    status: ClientSimulationStatus = ClientSimulationStatus.PENDING


class ClientSimulationMetricsUpdate(BaseModel):
    """Schema for updating metrics"""
    metrics: Optional[dict] = None
    status: Optional[ClientSimulationStatus] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @field_validator('metrics', mode='before')
    @classmethod
    def ensure_metrics_is_dict(cls, v):
        """Ensure metrics is a dict if provided"""
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return None


class ClientSimulationMetrics(ClientSimulationMetricsBase):
    """Complete schema with all fields (for responses)"""
    id: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra='ignore')

    @field_validator('created_at', 'updated_at', 'started_at', 'completed_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string or datetime object"""
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Handle ISO format strings from Supabase
            try:
                # Remove 'Z' and replace with +00:00 for UTC
                v_clean = v.replace('Z', '+00:00')
                return datetime.fromisoformat(v_clean)
            except Exception:
                return None
        return v


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_client_metrics(metrics: dict) -> dict:
    """
    Parse and validate client metrics structure.

    Args:
        metrics: Metrics dict from database

    Returns:
        Validated metrics dict with expected structure
    """
    if not metrics or not isinstance(metrics, dict):
        return {}

    # Ensure expected structure
    result = {
        "global": metrics.get("global", {}),
        "rounds": metrics.get("rounds", [])
    }

    return result


def get_latest_round_metrics(metrics: dict) -> Optional[dict]:
    """
    Get metrics from the latest training round.

    Args:
        metrics: Full metrics dict

    Returns:
        Latest round metrics or None
    """
    rounds = metrics.get("rounds", [])
    if not rounds:
        return None

    # Rounds should be sorted by round number
    latest = max(rounds, key=lambda r: r.get("round", 0))
    return latest.get("validation")


def get_data_heterogeneity(metrics: dict) -> Optional[dict]:
    """
    Get data heterogeneity statistics for a client's partition.

    This includes information about the Dirichlet partitioning that created
    the non-IID data distribution for this client.

    Args:
        metrics: Full metrics dict

    Returns:
        Dict with heterogeneity metrics or None if not available:
        {
            "total_samples": int,
            "train_samples": int,
            "val_samples": int,
            "class_distribution": {
                "leukemia": int,
                "healthy": int,
                "leukemia_pct": float,
                "healthy_pct": float
            },
            "imbalance_ratio": float,
            "partition_id": int
        }
    """
    return metrics.get("data_heterogeneity")


def get_improvement_summary(metrics: dict) -> dict:
    """
    Get improvement summary from theoretical baseline to post_fl.

    Theoretical baselines for binary classification:
    - accuracy, precision, recall, f1_score: 0.5 (random chance)
    - loss: 0.693 (ln(2) for random binary classifier)
    - specificity, roc_auc: 0.5
    - healthy_accuracy, leukemia_accuracy: 0.5
    - class_gap: 1.0 (max gap)

    Args:
        metrics: Full metrics dict

    Returns:
        Dict with improvement metrics
    """
    improvement = metrics.get("global", {}).get("improvement", {})

    return {
        "accuracy": improvement.get("accuracy", 0.0),
        "loss": improvement.get("loss", 0.0),
        "f1_score": improvement.get("f1_score", 0.0),
        "precision": improvement.get("precision", 0.0),
        "recall": improvement.get("recall", 0.0),
        "specificity": improvement.get("specificity", 0.0),
        "roc_auc": improvement.get("roc_auc", 0.0),
        "class_gap": improvement.get("class_gap", 0.0),
        "healthy_accuracy": improvement.get("healthy_accuracy", 0.0),
        "leukemia_accuracy": improvement.get("leukemia_accuracy", 0.0),
    }


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("Client Simulation Metrics Schema Examples")
    print("=" * 70)

    # Example 1: Create new metrics record
    print("\nExample 1: Create Metrics Record")
    print("-" * 70)

    create_data = ClientSimulationMetricsCreate(
        simulation_id=1,
        client_id=26,
        status=ClientSimulationStatus.PENDING
    )
    print(f"Simulation ID: {create_data.simulation_id}")
    print(f"Client ID: {create_data.client_id}")
    print(f"Status: {create_data.status}")

    # Example 2: Complete metrics object
    print("\nExample 2: Complete Metrics Object")
    print("-" * 70)

    metrics_data = {
        "global": {
            "post_fl": {
                "accuracy": 0.88,
                "loss": 0.255,
                "f1_score": 0.873
            },
            "improvement": {
                "accuracy": 0.38,  # From 0.5 baseline
                "loss": -0.438,   # From ln(2) baseline
                "f1_score": 0.373
            }
        },
        "rounds": [
            {
                "round": 1,
                "validation": {
                    "accuracy": 0.65,
                    "loss": 0.5
                }
            }
        ]
    }

    complete = ClientSimulationMetrics(
        id=1,
        simulation_id=1,
        client_id=26,
        metrics=metrics_data,
        status=ClientSimulationStatus.COMPLETED,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    print(f"Metrics ID: {complete.id}")
    print(f"Status: {complete.status}")

    # Example 3: Helper functions
    print("\nExample 3: Helper Functions")
    print("-" * 70)

    improvement = get_improvement_summary(complete.metrics)
    print(f"Improvement Summary:")
    print(f"  Accuracy: {improvement['accuracy']:.2%}")
    print(f"  F1 Score: {improvement['f1_score']:.2%}")

    latest_round = get_latest_round_metrics(complete.metrics)
    if latest_round:
        print(f"\nLatest Round Metrics:")
        print(f"  Accuracy: {latest_round.get('accuracy', 0):.2%}")
        print(f"  Loss: {latest_round.get('loss', 0):.4f}")
