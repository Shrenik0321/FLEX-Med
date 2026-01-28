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

# Import toml for pyproject.toml updates
try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False
    logger.warning("toml package not available, pyproject.toml updates will be skipped")


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

@router.post("/start_fl")
async def start_fl(
    supabase: SupabaseClient = Depends(get_supabase_client),
    settings: Settings = Depends(get_settings)
):
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

@router.post("/resume_fl")
async def resume_fl(
    supabase: SupabaseClient = Depends(get_supabase_client),
    settings: Settings = Depends(get_settings)
):
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
        # Forward to FL orchestrator's /resume_fl endpoint
        logger.info(f"Resuming FL via orchestrator at {settings.federated_training_orchestrator_url}")
        response = requests.post(
            f"{settings.federated_training_orchestrator_url}/resume_fl",
            json=clients_data,
            headers={"Content-Type": "application/json"},
            timeout=30
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


# ==========================================
# Helper Functions
# ==========================================

def update_pyproject_toml(configs: dict, num_clients: int, fl_path: Path) -> bool:
    """
    Update pyproject.toml with FL simulation parameters.

    Args:
        configs: FL hyperparameters dict
        num_clients: Number of participating clients
        fl_path: Path to federated_learning directory

    Returns:
        True if update succeeded, False otherwise
    """
    if not TOML_AVAILABLE:
        logger.warning("toml package not available, skipping pyproject.toml update")
        return False

    pyproject_path = fl_path / "pyproject.toml"

    if not pyproject_path.exists():
        logger.warning(f"pyproject.toml not found at {pyproject_path}")
        return False

    try:
        # Read current pyproject.toml
        with open(pyproject_path, 'r') as f:
            pyproject_data = toml.load(f)

        # Ensure nested structure exists
        if 'tool' not in pyproject_data:
            pyproject_data['tool'] = {}
        if 'flwr' not in pyproject_data['tool']:
            pyproject_data['tool']['flwr'] = {}
        if 'app' not in pyproject_data['tool']['flwr']:
            pyproject_data['tool']['flwr']['app'] = {}
        if 'config' not in pyproject_data['tool']['flwr']['app']:
            pyproject_data['tool']['flwr']['app']['config'] = {}

        # Update configuration
        app_config = pyproject_data['tool']['flwr']['app']['config']
        app_config['num-server-rounds'] = configs['num_server_rounds']
        app_config['local-epochs'] = configs['local_epochs']
        app_config['lr'] = configs['lr']
        app_config['lr-decay'] = configs.get('lr_decay', 0.98)
        app_config['distill-lr'] = configs.get('distill_lr', 0.001)
        app_config['distill-epochs'] = configs.get('distill_epochs', 2)
        app_config['temperature'] = configs.get('temperature', 3.0)
        app_config['batch-size'] = configs.get('batch_size', 32)

        # Reduce parallelism to prevent OOM errors
        # fraction_train controls how many clients run simultaneously during training
        # For 2 clients: 0.5 means 1 client at a time (50% of 2 = 1)
        # For 3 clients: 0.67 means 2 clients at a time (67% of 3 = 2)
        app_config['fraction-train'] = configs.get('fraction_train', 0.5)
        app_config['fraction-evaluate'] = configs.get('fraction_evaluate', 0.5)

        # Update simulation config
        if 'federations' not in pyproject_data['tool']['flwr']['app']:
            pyproject_data['tool']['flwr']['app']['federations'] = {}
        if 'default' not in pyproject_data['tool']['flwr']['app']['federations']:
            pyproject_data['tool']['flwr']['app']['federations']['default'] = {}

        pyproject_data['tool']['flwr']['app']['federations']['default']['options'] = {
            'num-supernodes': num_clients
        }

        # Write back
        with open(pyproject_path, 'w') as f:
            toml.dump(pyproject_data, f)

        logger.info(f"Updated pyproject.toml with {num_clients} clients and FL config")
        return True

    except Exception as e:
        logger.error(f"Failed to update pyproject.toml: {e}")
        return False


@router.post("/start_fl_simulation")
async def run_fl_simulation(
    background_tasks: BackgroundTasks,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """
    Run federated learning simulation locally (integrated approach).

    This endpoint:
    1. Fetches all clients from database
    2. Creates simulation record to track the FL run
    3. Creates client_simulation_metrics records for participating clients
    4. Prepares client configs and passes via environment variable
    5. Updates pyproject.toml with FL configuration
    6. Runs FL simulation locally using Flower CLI as background task
    7. Updates simulation status upon completion

    Returns:
        - message: Success message
        - simulation_id: ID of the created simulation record
        - mode: "integrated" to indicate local execution
        - num_clients: Number of participating clients
    """

    if not FL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="FL simulation package not available. Please check federated_learning path."
        )

    # --------------------------------------------------------------
    # Fetch clients
    # --------------------------------------------------------------

    response = supabase.from_("clients").select("*").execute()
    clients_data = response.data or []

    if not clients_data:
        raise HTTPException(status_code=400, detail="No clients found in database")

    client_ids = [c["id"] for c in clients_data]

    # --------------------------------------------------------------
    # FL Configuration
    # --------------------------------------------------------------

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
        "batch_size": 32,
    }

    sim_data = {
        "configs": configs,
        "status": SimulationStatus.PENDING.value,
    }

    # --------------------------------------------------------------
    # Create simulation record
    # --------------------------------------------------------------

    try:
        sim_response = supabase.from_("fl_simulations").insert(sim_data).execute()
        if not sim_response.data:
            raise RuntimeError("Simulation insert returned empty response")

        simulation_id = sim_response.data[0]["id"]
        logger.info(f"Created simulation record {simulation_id}")

    except Exception as e:
        logger.exception("Error creating simulation record")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # --------------------------------------------------------------
    # Create client_simulation_metrics records for participating clients
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Prepare client config for FL system (pass via environment variable)
    # --------------------------------------------------------------

    # Wrap client data and include simulation metadata
    simulation_config = {
        "simulation_id": simulation_id,
        "clients": clients_data
    }

    # Serialize to JSON string for passing via environment variable
    client_config_json = json.dumps(simulation_config)
    logger.info(f"Prepared configuration for {len(clients_data)} clients")

    # --------------------------------------------------------------
    # Update pyproject.toml with FL configuration
    # --------------------------------------------------------------

    update_pyproject_toml(configs, len(clients_data), FL_PATH)

    # --------------------------------------------------------------
    # Background Task (MUST be sync)
    # --------------------------------------------------------------

    def run_fl_background():
        try:
            supabase.from_("fl_simulations").update({
                "status": SimulationStatus.RUNNING.value,
                "started_at": datetime.utcnow().isoformat(),
            }).eq("id", simulation_id).execute()

            logger.info(f"Starting FL simulation {simulation_id}")

            # Redirect logs to simulation_run.log
            log_file_path = FL_PATH / "flex_med" / "utils" / "simulation_run.log"
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if FL_PATH contains spaces (which breaks Ray/Flower subprocess spawning)
            exec_path = FL_PATH
            env_override = os.environ.copy()
            run_cmd = ["flwr", "run", "."]

            # Add Supabase credentials, simulation_id, and client configs for FL system
            from app.config import get_settings
            settings = get_settings()
            env_override["SUPABASE_URL"] = settings.supabase_url
            env_override["SUPABASE_KEY"] = settings.supabase_key
            env_override["FLEX_MED_SIMULATION_ID"] = str(simulation_id)
            env_override["FLEX_MED_CLIENT_CONFIGS"] = client_config_json

            # Configure Ray to allow higher memory usage before killing workers
            # Default is 0.95 (95%), increase to 0.98 (98%) to prevent OOM kills
            env_override["RAY_memory_usage_threshold"] = "0.98"

            # Alternatively, disable Ray's memory monitor (use with caution)
            # env_override["RAY_memory_monitor_refresh_ms"] = "0"

            logger.info(f"Passing simulation_id={simulation_id} and client configs to FL system")

            if " " in str(FL_PATH):
                logger.info("Spaces detected in FL_PATH, using space-free symlink workaround")
                # Create a space-free symlink in /tmp for the WHOLE backend root
                # This ensures relative paths to datasets/ etc work
                backend_root = FL_PATH.parent
                symlink_backend = Path("/tmp/flex_med_backend")
                
                if symlink_backend.exists():
                    if symlink_backend.is_symlink():
                        symlink_backend.unlink()
                    elif symlink_backend.is_dir():
                        import shutil
                        shutil.rmtree(symlink_backend)
                
                try:
                    symlink_backend.symlink_to(backend_root)
                    exec_path = symlink_backend / "federated_learning"
                    logger.info(f"Created symlink: {symlink_backend} -> {backend_root}")
                    
                    # Update environment variables to use the symlink path
                    # This ensures Ray workers and config.py use the space-free path
                    env_override["BASE_PATH"] = str(symlink_backend)
                    env_override["FLEX_MED_PROJECT_DIR"] = str(exec_path)
                    
                    # CRITICAL: Use the symlinked python and update PATH/PYTHONPATH 
                    # to avoid Ray worker startup failures due to spaces in original path
                    sym_venv_bin = symlink_backend / ".venv" / "bin"
                    env_override["PATH"] = f"{sym_venv_bin}:{env_override.get('PATH', '')}"
                    env_override["PYTHONPATH"] = f"{symlink_backend}:{exec_path}:{env_override.get('PYTHONPATH', '')}"
                    
                    # Use the symlinked python executable directly to bypass shebang issues
                    python_exec = sym_venv_bin / "python3"
                    run_cmd = [str(python_exec), "-m", "flwr.cli.app", "run", "."]
                    
                    logger.info(f"Running command: {' '.join(run_cmd)}")
                except Exception as sym_err:
                    logger.error(f"Failed to create symlink workaround: {sym_err}")
                    # Fallback to original path if symlink fails
                    exec_path = FL_PATH
                    run_cmd = ["flwr", "run", "."]


            # Open log file for writing
            with open(log_file_path, "w") as log_file:
                # Write header
                log_file.write(f"[API] Starting FL Simulation {simulation_id} at {datetime.now()}\n")
                log_file.write(f"[API] Execution Path: {exec_path}\n")
                log_file.write(f"[API] Command: {' '.join(run_cmd)}\n")
                log_file.write("=" * 80 + "\n")
                log_file.flush()
                
                # Run subprocess with output redirected to log file
                result = subprocess.run(
                    run_cmd,
                    cwd=str(exec_path),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout (log file)
                    env=env_override,
                    text=True,
                    timeout=3600,
                )
                
                # Write footer
                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write(f"[API] FL Simulation {simulation_id} finished at {datetime.now()}\n")
                log_file.write(f"[API] Exit code: {result.returncode}\n")

            if result.returncode == 0:
                # Calculate duration
                sim_response = supabase.from_("fl_simulations").select("started_at").eq("id", simulation_id).execute()
                duration = None
                if sim_response.data and sim_response.data[0].get("started_at"):
                    from datetime import datetime as dt
                    started_at = dt.fromisoformat(sim_response.data[0]["started_at"].replace("Z", "+00:00"))
                    completed_at = dt.utcnow()
                    duration = int((completed_at - started_at).total_seconds())

                # Update simulation status
                # Note: FL system (task.py) already updated aggregate_metrics and status to 'completed'
                # We just add duration here
                update_data = {}
                if duration is not None:
                    update_data["duration"] = duration

                if update_data:
                    supabase.from_("fl_simulations").update(update_data).eq("id", simulation_id).execute()

                logger.info(f"✅ FL simulation {simulation_id} completed (duration: {duration}s)")

            else:
                error_msg = (result.stderr or "Unknown error")[-500:]
                supabase.from_("fl_simulations").update({
                    "status": SimulationStatus.FAILED.value,
                    "error_message": error_msg,
                }).eq("id", simulation_id).execute()

                logger.error(f"FL simulation {simulation_id} failed")

        except subprocess.TimeoutExpired:
            error_msg = "FL simulation timeout (exceeded 1 hour)"
            supabase.from_("fl_simulations").update({
                "status": SimulationStatus.FAILED.value,
                "error_message": error_msg,
            }).eq("id", simulation_id).execute()

            logger.error(error_msg)

        except Exception as e:
            supabase.from_("fl_simulations").update({
                "status": SimulationStatus.FAILED.value,
                "error_message": str(e),
            }).eq("id", simulation_id).execute()

            logger.exception("Unexpected FL simulation error")

    # --------------------------------------------------------------
    # Register background task
    # --------------------------------------------------------------

    background_tasks.add_task(run_fl_background)

    return {
        "message": "FL simulation started (integrated mode)",
        "simulation_id": simulation_id,
        "mode": "integrated",
        "num_clients": len(client_ids),
    }

@router.get("/fl_simulation_logs")
async def get_fl_simulation_logs(
    lines: int = 100,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """
    Get the latest logs from the FL simulation.
    
    Args:
        lines: Number of lines to return from the end of the log file (default: 100)
    
    Returns:
        - logs: List of log lines
        - total_lines: Total number of lines in the log file
        - log_file_path: Path to the log file
    """
    log_file_path = FL_PATH / "flex_med" / "utils" / "simulation_run.log"
    
    if not log_file_path.exists():
        return {
            "logs": [],
            "total_lines": 0,
            "log_file_path": str(log_file_path),
            "message": "No log file found - simulation may not have started yet"
        }
    
    try:
        with open(log_file_path, "r") as f:
            all_lines = f.readlines()
        
        # Get the last N lines
        log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": [line.strip() for line in log_lines],
            "total_lines": len(all_lines),
            "log_file_path": str(log_file_path),
            "showing_lines": len(log_lines)
        }
    except Exception as e:
        logger.exception("Error reading simulation logs")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")

