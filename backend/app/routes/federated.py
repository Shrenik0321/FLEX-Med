from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import requests
import json
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime
from supabase import Client as SupabaseClient
from app.config import get_supabase_client, get_settings, Settings
from app.schemas.fl_simulations import (
    FLSimulationCreate, FLSimulation, FLSimulationUpdate, SimulationStatus,
    compute_aggregate_metrics
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Load orchestrator URL from settings
# Ngrok URL for local training orchestrator (update when Colab session changes)
FEDERATED_TRAINING_ORCHESTRATOR_URL = "https://02d564fec0cd.ngrok-free.app"

# Global FL pipeline status tracker
fl_status = {
    "is_running": False,
    "current_stage": None,
    "progress": 0,
    "total_rounds": 0,
    "current_round": 0,
    "clients": 0,
    "started_at": None,
    "completed_at": None,
    "error": None,
    "logs": []
}

def update_fl_status(stage: str = None, progress: int = None, current_round: int = None, log: str = None, error: str = None, completed: bool = False):
    """Update global FL status for monitoring"""
    global fl_status

    if stage:
        fl_status["current_stage"] = stage
    if progress is not None:
        fl_status["progress"] = progress
    if current_round is not None:
        fl_status["current_round"] = current_round
    if log:
        timestamp = datetime.now().strftime("%H:%M:%S")
        fl_status["logs"].append(f"[{timestamp}] {log}")
        # Keep only last 50 logs
        if len(fl_status["logs"]) > 50:
            fl_status["logs"] = fl_status["logs"][-50:]
    if error:
        fl_status["error"] = error
    if completed:
        fl_status["is_running"] = False
        fl_status["completed_at"] = datetime.now().isoformat()
        fl_status["progress"] = 100

@router.post("/start_fl")
async def start_fl(supabase: SupabaseClient = Depends(get_supabase_client)):
    """
    Start a new federated learning simulation.

    This endpoint:
    1. Fetches all clients from the database
    2. Creates a simulation record to track the FL run
    3. Forwards the request to the FL orchestrator (Colab/ngrok)
    4. Updates simulation status based on orchestrator response

    Returns:
        - message: Success/error message
        - simulation_id: ID of the created simulation record
        - orchestrator_response: Response from FL orchestrator
    """
    # Fetch all clients from database
    response = supabase.from_("clients").select("*").execute()
    clients_data = response.data if response.data else []

    if not clients_data:
        raise HTTPException(status_code=400, detail="No clients found in database")

    # Extract client IDs for simulation record
    client_ids = [c['id'] for c in clients_data]

    # Define FL configuration (from pyproject.toml defaults)
    configs = {
        "num_server_rounds": 10,
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "local_epochs": 8,
        "lr": 0.0001,
        "lr_decay": 0.98,
        "distill_lr": 0.001,
        "distill_epochs": 2,
        "temperature": 3.0,
        "batch_size": 32
    }

    # Create simulation record
    sim_data = {
        "client_ids": client_ids,
        "configs": configs,
        "status": SimulationStatus.PENDING,
        "metrics": "{}"
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
            "status": SimulationStatus.RUNNING,
            "started_at": datetime.now().isoformat()
        }).eq("id", simulation_id).execute()

        logger.info(f"Starting FL simulation {simulation_id} via orchestrator")

        # Forward to FL orchestrator
        response = requests.post(
            f"{FEDERATED_TRAINING_ORCHESTRATOR_URL}/start_fl",
            json=clients_data,
            headers={"Content-Type": "application/json"}
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
            "status": SimulationStatus.FAILED,
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
            "status": SimulationStatus.FAILED,
            "error_message": error_message
        }).eq("id", simulation_id).execute()

        logger.error(f"FL simulation {simulation_id} failed: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/resume_fl")
async def resume_fl(supabase: SupabaseClient = Depends(get_supabase_client)):
    """
    Resume FL simulation from last checkpoint.

    This endpoint:
    1. Fetches clients from database
    2. Validates pre_fl metrics exist (required for resume)
    3. Forwards to Colab orchestrator's /resume_fl endpoint
    4. Skips pre-FL evaluation (already done)
    5. Runs FL from last checkpoint
    6. Post-FL evaluation runs after completion
    """
    # Fetch all clients from database
    response = supabase.from_("clients").select("*").execute()
    clients_data = response.data if response.data else []

    if not clients_data:
        raise HTTPException(status_code=400, detail="No clients found in database")

    # Validate that pre_fl metrics exist for all clients
    missing_pre_fl = []
    for client in clients_data:
        metrics_value = client.get('metrics', '{}')
        if isinstance(metrics_value, str):
            try:
                metrics = json.loads(metrics_value) if metrics_value else {}
            except json.JSONDecodeError:
                metrics = {}
        else:
            metrics = metrics_value or {}

        if 'pre_fl' not in metrics:
            missing_pre_fl.append(client.get('client_name', f"ID:{client.get('id')}"))

    if missing_pre_fl:
        raise HTTPException(
            status_code=400,
            detail=f"Pre-FL metrics missing for clients: {', '.join(missing_pre_fl)}. Use /start_fl for fresh training."
        )

    # Clean up metrics field (same as start_fl)
    for client in clients_data:
        metrics_value = client.get('metrics', '{}')
        if isinstance(metrics_value, str):
            if metrics_value.startswith('"') and metrics_value.endswith('"'):
                try:
                    metrics_value = json.loads(metrics_value)
                except json.JSONDecodeError:
                    metrics_value = metrics_value.strip('"')
            if not metrics_value or metrics_value.strip() == '':
                metrics_value = '{}'
        elif isinstance(metrics_value, dict):
            metrics_value = json.dumps(metrics_value)
        else:
            metrics_value = '{}'
        client['metrics'] = metrics_value

    try:
        # Forward to Colab orchestrator's /resume_fl endpoint
        response = requests.post(
            f"{FEDERATED_TRAINING_ORCHESTRATOR_URL}/resume_fl",
            json=clients_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Orchestrator API Error: {e.response.text}"
            )
        raise HTTPException(status_code=500, detail=f"Failed to connect to orchestrator: {str(e)}")

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
    """
    data = {
        "client_ids": simulation_data.client_ids,
        "configs": simulation_data.configs,
        "status": SimulationStatus.PENDING,
        "metrics": "{}"
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
        client_ids = simulation['client_ids']

        # Fetch all client metrics
        clients_response = supabase.from_("clients").select("*").in_("id", client_ids).execute()
        clients_data = clients_response.data if clients_response.data else []

        if not clients_data:
            raise HTTPException(
                status_code=400,
                detail=f"No clients found for simulation {simulation_id}"
            )

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
            "status": SimulationStatus.COMPLETED,
            "metrics": metrics_str,
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