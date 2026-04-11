from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import requests
import json
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

logger = logging.getLogger(__name__)

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

    # Load configuration from system_config table (single source of truth)
    try:
        config_response = supabase.from_("system_config").select("*").limit(1).execute()
        if config_response.data and len(config_response.data) > 0:
            system_config = config_response.data[0]["config"]
            logger.info(f"Loaded system_config from database")
        else:
            logger.warning("No system_config found in database, using defaults")
            system_config = {}
    except Exception as e:
        logger.warning(f"Failed to load system_config from database: {e}, using defaults")
        system_config = {}

    # Build FL configs — global params from top-level system_config
    configs = {
        "num_server_rounds": system_config.get("num_rounds", 10),
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "local_epochs": system_config.get("local_epochs", 1),
        "lr": system_config.get("learning_rate", 0.0005),
        "lr_decay": system_config.get("lr_decay", 0.90),
        "weight_decay": system_config.get("weight_decay", 0.02),
        "distill_lr": system_config.get("distill_lr", 0.0005),
        "distill_epochs": system_config.get("distill_epochs", 1),
        "temperature": system_config.get("temperature", 4.0),
        "batch_size": system_config.get("batch_size", 32),
        # Centralized baseline: trains each client independently (no FL) before Round 1
        # Set to true in system_config when you want a baseline comparison stored in DB
        "run_centralized_baseline": bool(system_config.get("run_centralized_baseline", False)),
    }

    # Build training strategy config
    training_config = {
        "dirichlet_alpha": system_config.get("dirichlet_alpha", 1.5),
        "dirichlet_seed": system_config.get("dirichlet_seed", 42),
        "dirichlet_min_partition_size": system_config.get("dirichlet_min_partition_size", 400),
        "minority_boost": system_config.get("minority_boost", 0.78),
        "distill_weight_base": system_config.get("distill_weight_base", 0.80),
        "distill_decay_rate": system_config.get("distill_decay_rate", 0.18),
        "train_loss_weight": system_config.get("train_loss_weight", 0.72),
        "distill_loss_weight": system_config.get("distill_loss_weight", 0.28),
        "weight_decay": system_config.get("weight_decay", 0.02),
    }

    # Build dataset paths config
    dataset_paths = {
        "public_anchor": system_config.get("public_anchor_dataset_path", "/content/datasets/cnmc/cnmc_public_anchor"),
        "public_test": system_config.get("public_test_dataset_path", "/content/datasets/cnmc/cnmc_public_test"),
        "local_train": system_config.get("local_train_dataset_path", "/content/datasets/cnmc/cnmc_local_train"),
    }

    # Resolve orchestrator URL from system_config or fallback to settings
    orchestrator_url = system_config.get("ngrok_url", settings.federated_training_orchestrator_url)

    logger.info(f"FL config: {configs}")
    logger.info(f"Training config: {training_config}")
    logger.info(f"Orchestrator URL: {orchestrator_url}")

    # Create simulation record with full config snapshot
    sim_data = {
        "configs": {
            **configs,
            "dirichlet_alpha": training_config["dirichlet_alpha"],
            "training_config": training_config,
        },
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

        logger.info(f"Starting FL simulation {simulation_id} via orchestrator at {orchestrator_url}")

        # Prepare request payload with simulation_id, client data, and full config
        orchestrator_payload = {
            "simulation_id": simulation_id,
            "clients": clients_data,
            "supabase_url": settings.supabase_url,
            "supabase_key": settings.supabase_key,
            "fl_config": configs,
            "training_config": training_config,
            "dataset_paths": dataset_paths,
        }

        # Forward to FL orchestrator
        response = requests.post(
            f"{orchestrator_url}/start_fl",
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