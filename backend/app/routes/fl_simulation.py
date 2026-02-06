from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import requests
import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
from supabase import Client as SupabaseClient
from app.config import get_supabase_client, get_settings, Settings
from app.schemas.fl_simulations import (
    FLSimulationCreate, FLSimulation, FLSimulationUpdate, SimulationStatus,
    compute_aggregate_metrics
)
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Add federated_learning to Python path for importing FL logic
FL_PATH = Path(__file__).parent.parent.parent / "federated_learning"
if str(FL_PATH) not in sys.path:
    sys.path.insert(0, str(FL_PATH))

# Import FL simulation logic from flex_med package
try:
    from flex_med.task import load_client_config
    # from flex_med.server import run_federated_learning # Removed: module does not exist and function is unused
    FL_AVAILABLE = True
    logger.info("FL simulation package imported successfully")
except ImportError as e:
    FL_AVAILABLE = False
    logger.warning(f"FL simulation package not available: {e}")

router = APIRouter()

class StartFLRequest(BaseModel):
    """Request body for starting FL simulation."""
    client_ids: Optional[List[int]] = None  # If None, all clients are used
    dataset: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@router.post("/start_fl")
async def start_fl(
    request: Optional[StartFLRequest] = None,
    supabase: SupabaseClient = Depends(get_supabase_client),
    settings: Settings = Depends(get_settings)
):
    """
    Start a new federated learning simulation.

    This endpoint:
    1. Fetches clients from the database (optionally filtered by client_ids)
    2. Creates a simulation record to track the FL run
    3. Forwards the request to the FL orchestrator (Colab/ngrok)
    4. Updates simulation status based on orchestrator response

    Request Body:
        - client_ids: Optional list of client IDs to participate (if None, all clients are used)
        - dataset: Optional dataset selection
        - config: Optional FL configuration overrides

    Returns:
        - message: Success/error message
        - simulation_id: ID of the created simulation record
        - orchestrator_response: Response from FL orchestrator
    """
    # Parse request body
    selected_client_ids = request.client_ids if request else None
    dataset = request.dataset if request else None
    user_config = request.config if request else None

    # Fetch all clients from database
    response = supabase.from_("clients").select("*").execute()
    all_clients = response.data if response.data else []

    if not all_clients:
        raise HTTPException(status_code=400, detail="No clients found in database")

    # Filter clients based on selected_client_ids
    if selected_client_ids:
        clients_data = [c for c in all_clients if c['id'] in selected_client_ids]

        if not clients_data:
            raise HTTPException(
                status_code=400,
                detail=f"No clients found with IDs: {selected_client_ids}"
            )

        # Validate that all selected IDs exist
        found_ids = {c['id'] for c in clients_data}
        missing_ids = set(selected_client_ids) - found_ids
        if missing_ids:
            logger.warning(f"Some client IDs not found: {missing_ids}")
    else:
        # No filter, use all clients
        clients_data = all_clients

    logger.info(f"Starting FL with {len(clients_data)} clients (selected from {len(all_clients)} total)")

    # Extract client IDs for simulation record
    client_ids = [c['id'] for c in clients_data]

    # Define FL configuration (merge user config with defaults)
    default_configs = {
        "num_server_rounds": 2,
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "local_epochs": 2,
        "lr": 0.0001,
        "lr_decay": 0.98,
        "distill_lr": 0.001,
        "distill_epochs": 2,
        "temperature": 3.0,
        "batch_size": 32
    }

    # Merge user-provided config with defaults
    configs = {**default_configs}
    if user_config:
        # Map frontend config keys to backend config keys
        config_mapping = {
            "numRounds": "num_server_rounds",
            "localEpochs": "local_epochs",
            "learningRate": "lr",
            "batchSize": "batch_size",
            "distillEpochs": "distill_epochs",
            "distillLearningRate": "distill_lr",
            "temperature": "temperature"
        }

        for frontend_key, backend_key in config_mapping.items():
            if frontend_key in user_config:
                configs[backend_key] = user_config[frontend_key]

        logger.info(f"Applied user config overrides: {user_config}")
        
    # The notebook's /start_fl endpoint updates its local pyproject.toml
    # based on the number of clients in the request payload

    # Create simulation record (without client_ids - new schema)
    sim_data = {
        "configs": configs,
        "status": SimulationStatus.PENDING.value,
    }

    try:
        sim_response = supabase.from_("fl_simulations").insert(sim_data).execute()
        if not sim_response.data:
            raise HTTPException(status_code=500, detail="Failed to create simulation record")

        simulation_id = sim_response.data[0]['id']
        logger.info(f"Created simulation record {simulation_id} for {len(client_ids)} clients")

    except Exception as e:
        logger.error(f"Error creating simulation record: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Create client_simulation_metrics records for participating clients (new schema)
    try:
        for client in clients_data:
            client_metrics_data = {
                "simulation_id": simulation_id,
                "client_id": client["id"],
                "status": "pending",
                "metrics": {}  # Empty JSONB, will be populated during FL
            }
            supabase.from_("client_simulation_metrics").insert(client_metrics_data).execute()

        logger.info(f"Created {len(clients_data)} client_simulation_metrics records for simulation {simulation_id}")

    except Exception as e:
        logger.error(f"Error creating client_simulation_metrics records: {e}")
        # Not fatal - continue with simulation (metrics will be missing though)
        logger.warning("Simulation will continue but metrics may not be saved properly")

    # Clean up metrics field to prevent double-encoding
    for client in clients_data:
        metrics_value = client.get('metrics', '{}')

        # Handle string metrics
        if isinstance(metrics_value, str):
            # Remove double-encoding if present: "\"{}\""
            if metrics_value.startswith('"') and metrics_value.endswith('"'):
                try:
                    metrics_value = json.loads(metrics_value)
                except json.JSONDecodeError:
                    metrics_value = metrics_value.strip('"')

            # Ensure valid JSON
            if not metrics_value or metrics_value.strip() == '':
                metrics_value = '{}'

        # Handle dict metrics
        elif isinstance(metrics_value, dict):
            metrics_value = json.dumps(metrics_value)

        # Default to empty JSON
        else:
            metrics_value = '{}'

        client['metrics'] = metrics_value

    try:
        # Update simulation status to RUNNING
        supabase.from_("fl_simulations").update({
            "status": SimulationStatus.RUNNING.value,
            "started_at": datetime.now().isoformat()
        }).eq("id", simulation_id).execute()

        logger.info(f"Starting FL simulation {simulation_id} via orchestrator at {settings.federated_training_orchestrator_url}")

        # Prepare request payload with simulation_id and client data
        orchestrator_payload = {
            "simulation_id": simulation_id,
            "clients": clients_data,
            "supabase_url": settings.supabase_url,
            "supabase_key": settings.supabase_key
        }

        # Forward to FL orchestrator
        response = requests.post(
            f"{settings.federated_training_orchestrator_url}/start_fl",
            json=orchestrator_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        orchestrator_response = response.json()
        logger.info(f"FL simulation {simulation_id} started successfully")

        return {
            "message": "FL started successfully",
            "simulation_id": simulation_id,
            "orchestrator_response": orchestrator_response
        }

    except requests.exceptions.RequestException as e:
        # Mark simulation as failed
        error_message = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_message = f"Orchestrator API Error: {e.response.text}"

        supabase.from_("fl_simulations").update({
            "status": SimulationStatus.FAILED.value,
            "error_message": error_message
        }).eq("id", simulation_id).execute()

        logger.error(f"FL simulation {simulation_id} failed: {error_message}")

        if hasattr(e, 'response') and e.response is not None:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=error_message
            )
        raise HTTPException(status_code=500, detail=f"Failed to connect to orchestrator: {error_message}")

    except Exception as e:
        # Catch any other unexpected errors
        error_message = f"Unexpected error: {str(e)}"

        supabase.from_("fl_simulations").update({
            "status": SimulationStatus.FAILED.value,
            "error_message": error_message
        }).eq("id", simulation_id).execute()

        logger.error(f"FL simulation {simulation_id} failed: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

# ==========================================
# FL SIMULATION MANAGEMENT ENDPOINTS
# ==========================================

@router.post("/fl_simulations", response_model=FLSimulation)
async def create_simulation(
    simulation_data: FLSimulationCreate,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Create a new FL simulation record.

    This endpoint is typically called before starting an FL run to track
    the simulation configuration and status.

    Note: This is a standalone CRUD endpoint. For running FL, use /start_fl or /start_fl_simulation.
    """
    data = {
        "configs": simulation_data.configs,
        "status": SimulationStatus.PENDING.value,
    }

    try:
        response = supabase.from_("fl_simulations").insert(data).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create simulation")
        return response.data[0]
    except Exception as e:
        logger.error(f"Error creating simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/fl_simulations", response_model=List[FLSimulation])
async def list_simulations(
    status: Optional[str] = None,
    limit: int = 50,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    List all FL simulations, ordered by creation date (newest first).

    Args:
        status: Optional filter by status (pending, running, completed, failed)
        limit: Maximum number of results to return (default: 50)
    """
    try:
        query = supabase.from_("fl_simulations").select("*")

        # Apply status filter if provided
        if status:
            query = query.eq("status", status)

        # Order by created_at descending and limit
        query = query.order("created_at", desc=True).limit(limit)

        response = query.execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error listing simulations: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/fl_simulations/{simulation_id}", response_model=FLSimulation)
async def get_simulation(
    simulation_id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get a specific FL simulation by ID.

    Returns the full simulation record including configs, metrics, and timing info.
    """
    try:
        response = supabase.from_("fl_simulations").select("*").eq("id", simulation_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.patch("/fl_simulations/{simulation_id}")
async def update_simulation(
    simulation_id: int,
    update_data: FLSimulationUpdate,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Update an FL simulation record.

    Allows updating status, metrics, error messages, and timing information.
    Only provided fields will be updated (null fields are ignored).
    """
    # Convert to dict, exclude None values
    data = update_data.model_dump(exclude_none=True)

    if not data:
        raise HTTPException(status_code=400, detail="No update data provided")

    # Convert datetime objects to ISO strings if present
    for field in ['started_at', 'completed_at']:
        if field in data and isinstance(data[field], datetime):
            data[field] = data[field].isoformat()

    try:
        response = supabase.from_("fl_simulations").update(data).eq("id", simulation_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/fl_simulations/{simulation_id}/complete")
async def complete_simulation(
    simulation_id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Mark a simulation as completed and compute final aggregate metrics.

    This endpoint:
    1. Fetches the simulation record
    2. Fetches all participating client metrics
    3. Computes aggregate metrics (averages, std, round-by-round)
    4. Updates simulation with final metrics, completion time, and duration

    Should be called after FL orchestrator completes training.
    """
    try:
        # Fetch simulation record
        sim_response = supabase.from_("fl_simulations").select("*").eq("id", simulation_id).execute()
        if not sim_response.data:
            raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")

        simulation = sim_response.data[0]

        # Fetch all client metrics from client_simulation_metrics table (new schema)
        metrics_response = supabase.from_("client_simulation_metrics").select("*").eq("simulation_id", simulation_id).execute()
        metrics_data = metrics_response.data if metrics_response.data else []

        if not metrics_data:
            raise HTTPException(
                status_code=400,
                detail=f"No client metrics found for simulation {simulation_id}"
            )

        # Convert metrics from client_simulation_metrics format to clients format
        # (compute_aggregate_metrics expects 'metrics' field as string)
        clients_data = []
        for metric_record in metrics_data:
            clients_data.append({
                'id': metric_record['client_id'],
                'metrics': json.dumps(metric_record['metrics']) if isinstance(metric_record['metrics'], dict) else metric_record['metrics']
            })

        # Compute aggregate metrics
        aggregate_metrics = compute_aggregate_metrics(clients_data)
        metrics_str = json.dumps(aggregate_metrics)

        # Calculate duration
        started_at = simulation.get('started_at')
        completed_at = datetime.now()
        duration = None

        if started_at:
            # Parse ISO string to datetime
            if isinstance(started_at, str):
                started_dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            else:
                started_dt = started_at
            duration = int((completed_at - started_dt).total_seconds())

        # Update simulation
        update_data = {
            "status": SimulationStatus.COMPLETED.value,
            "aggregate_metrics": aggregate_metrics,  # Store as JSONB dict (not string)
            "completed_at": completed_at.isoformat(),
            "duration": duration
        }

        response = supabase.from_("fl_simulations").update(update_data).eq("id", simulation_id).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to update simulation")

        logger.info(f"Simulation {simulation_id} completed successfully. Duration: {duration}s")
        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error completing simulation: {str(e)}")