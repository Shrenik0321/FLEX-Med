from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, field_validator
import json

class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class FLSimulationBase(BaseModel):
    """Base schema for FL Simulation"""
    configs: dict  # JSONB - FL hyperparameters
    aggregate_metrics: dict = {}  # JSONB - Aggregated metrics from all clients
    status: SimulationStatus = SimulationStatus.PENDING
    error_message: Optional[str] = None

    @field_validator('configs', mode='before')
    @classmethod
    def ensure_configs_is_dict(cls, v):
        """Ensure configs is a dict (not None or empty string)"""
        if v is None or v == "":
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return {}

    @field_validator('aggregate_metrics', mode='before')
    @classmethod
    def ensure_aggregate_metrics_is_dict(cls, v):
        """Ensure aggregate_metrics is a dict (not string-encoded JSON)"""
        if v is None or v == "":
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            # In case Supabase returns string (shouldn't happen with JSONB)
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return {}

class FLSimulationCreate(BaseModel):
    """Schema for creating new simulation"""
    configs: dict

    # Note: client_ids can be computed from client_simulation_metrics table
    # but we keep it here for convenience/caching

class FLSimulationUpdate(BaseModel):
    """Schema for updating simulation"""
    status: Optional[SimulationStatus] = None
    aggregate_metrics: Optional[dict] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[int] = None

class FLSimulation(FLSimulationBase):
    """Complete simulation schema with all fields"""
    id: int
    created_at: str  # ISO format
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration: Optional[int] = None

    class Config:
        from_attributes = True

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_simulation_metrics(metrics_str: str) -> dict:
    """
    Parse simulation metrics JSON string.

    Args:
        metrics_str: JSON string containing simulation metrics

    Returns:
        Dict containing metrics, or empty dict if parsing fails
    """
    if not metrics_str or metrics_str.strip() == "":
        return {}
    try:
        return json.loads(metrics_str)
    except json.JSONDecodeError:
        return {}

def compute_aggregate_metrics(clients_data: List[dict]) -> dict:
    """
    Compute aggregate metrics from multiple clients' metrics.

    This function aggregates individual client metrics (pre_fl, post_fl, rounds)
    into simulation-level averages and statistics.

    Args:
        clients_data: List of client records with 'metrics' field

    Returns:
        Aggregated metrics structure with:
        - aggregate: {pre_fl, post_fl, improvement} with averages and std
        - rounds: Array of per-round aggregate metrics
        - best_round: Best performing round
        - total_rounds_completed, total_clients
    """
    # Parse all client metrics
    all_metrics = []
    for client in clients_data:
        metrics_str = client.get('metrics', '{}')
        if isinstance(metrics_str, str):
            try:
                metrics = json.loads(metrics_str)
                all_metrics.append(metrics)
            except:
                continue

    if not all_metrics:
        return {}

    # Calculate aggregates for pre_fl and post_fl
    def calc_avg(metric_name: str, stage: str) -> float:
        """Calculate average of a metric across all clients"""
        values = []
        for m in all_metrics:
            if stage in m.get('global', {}):
                val = m['global'][stage].get(metric_name)
                if val is not None:
                    values.append(val)
        return sum(values) / len(values) if values else 0.0

    def calc_std(metric_name: str, stage: str) -> float:
        """Calculate standard deviation of a metric across all clients"""
        values = []
        for m in all_metrics:
            if stage in m.get('global', {}):
                val = m['global'][stage].get(metric_name)
                if val is not None:
                    values.append(val)
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    # Aggregate round-by-round metrics
    rounds_aggregate = []
    num_rounds = max(len(m.get('rounds', [])) for m in all_metrics) if all_metrics else 0

    for round_num in range(1, num_rounds + 1):
        round_metrics = {
            'round': round_num,
            'avg_accuracy': 0.0,
            'avg_loss': 0.0,
            'avg_f1': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'num_clients_trained': 0,
            'timestamp': None
        }

        values_acc = []
        values_loss = []
        values_f1 = []
        values_precision = []
        values_recall = []

        for m in all_metrics:
            rounds = m.get('rounds', [])
            for r in rounds:
                if r.get('round') == round_num:
                    if 'validation' in r:
                        val = r['validation']
                        values_acc.append(val.get('accuracy', 0))
                        values_loss.append(val.get('loss', 0))
                        values_f1.append(val.get('f1_score', 0))
                        values_precision.append(val.get('precision', 0))
                        values_recall.append(val.get('recall', 0))
                        if not round_metrics['timestamp']:
                            round_metrics['timestamp'] = val.get('evaluated_at')

        if values_acc:
            round_metrics['avg_accuracy'] = sum(values_acc) / len(values_acc)
            round_metrics['avg_loss'] = sum(values_loss) / len(values_loss)
            round_metrics['avg_f1'] = sum(values_f1) / len(values_f1)
            round_metrics['avg_precision'] = sum(values_precision) / len(values_precision)
            round_metrics['avg_recall'] = sum(values_recall) / len(values_recall)
            round_metrics['num_clients_trained'] = len(values_acc)
            rounds_aggregate.append(round_metrics)

    # Find best round
    best_round = max(rounds_aggregate, key=lambda r: r['avg_accuracy']) if rounds_aggregate else {}

    return {
        "aggregate": {
            "pre_fl": {
                "avg_accuracy": calc_avg('accuracy', 'pre_fl'),
                "avg_loss": calc_avg('loss', 'pre_fl'),
                "avg_precision": calc_avg('precision', 'pre_fl'),
                "avg_recall": calc_avg('recall', 'pre_fl'),
                "avg_f1": calc_avg('f1_score', 'pre_fl'),
                "std_accuracy": calc_std('accuracy', 'pre_fl'),
                "num_clients": len(all_metrics)
            },
            "post_fl": {
                "avg_accuracy": calc_avg('accuracy', 'post_fl'),
                "avg_loss": calc_avg('loss', 'post_fl'),
                "avg_precision": calc_avg('precision', 'post_fl'),
                "avg_recall": calc_avg('recall', 'post_fl'),
                "avg_f1": calc_avg('f1_score', 'post_fl'),
                "std_accuracy": calc_std('accuracy', 'post_fl'),
                "num_clients": len(all_metrics)
            },
            "improvement": {
                "avg_accuracy": calc_avg('accuracy', 'post_fl') - calc_avg('accuracy', 'pre_fl'),
                "avg_loss": calc_avg('loss', 'post_fl') - calc_avg('loss', 'pre_fl'),
                "avg_precision": calc_avg('precision', 'post_fl') - calc_avg('precision', 'pre_fl'),
                "avg_recall": calc_avg('recall', 'post_fl') - calc_avg('recall', 'pre_fl'),
                "avg_f1": calc_avg('f1_score', 'post_fl') - calc_avg('f1_score', 'pre_fl')
            }
        },
        "rounds": rounds_aggregate,
        "best_round": {
            "round": best_round.get('round'),
            "avg_accuracy": best_round.get('avg_accuracy')
        } if best_round else {},
        "total_rounds_completed": len(rounds_aggregate),
        "total_clients": len(all_metrics)
    }


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("FL Simulation Schema Examples")
    print("=" * 70)

    # Example 1: Create simulation
    print("\nExample 1: Create Simulation")
    print("-" * 70)

    sim_create = FLSimulationCreate(
        client_ids=[1, 2, 3],
        configs={
            "num_server_rounds": 10,
            "lr": 0.0001,
            "batch_size": 32
        }
    )
    print(f"Client IDs: {sim_create.client_ids}")
    print(f"Configs: {sim_create.configs}")

    # Example 2: Complete simulation object
    print("\nExample 2: Complete Simulation")
    print("-" * 70)

    sim = FLSimulation(
        id=1,
        client_ids=[1, 2],
        configs={"num_server_rounds": 5, "lr": 0.0001},
        metrics='{"aggregate": {"post_fl": {"avg_accuracy": 0.85}}}',
        status=SimulationStatus.COMPLETED,
        created_at=datetime.now().isoformat(),
        started_at=datetime.now().isoformat(),
        completed_at=datetime.now().isoformat(),
        duration=3600
    )
    print(f"Simulation ID: {sim.id}")
    print(f"Status: {sim.status}")
    print(f"Duration: {sim.duration}s")

    # Example 3: Parse metrics
    print("\nExample 3: Parse Metrics")
    print("-" * 70)

    metrics = parse_simulation_metrics(sim.metrics)
    print(f"Parsed metrics: {json.dumps(metrics, indent=2)}")
