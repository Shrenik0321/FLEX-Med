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
from app.services.fl_evaluation_service import evaluate_all_clients, calculate_improvement
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Load orchestrator URL from settings
# Ngrok URL for local training orchestrator (update when Colab session changes)
FEDERATED_TRAINING_ORCHESTRATOR_URL = os.getenv(
    "FEDERATED_TRAINING_ORCHESTRATOR_URL",
    "https://6bf8b79c0b23.ngrok-free.app"
)

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
    # Fetch all clients from database
    response = supabase.from_("clients").select("*").execute()
    clients_data = response.data if response.data else []

    if not clients_data:
        raise HTTPException(status_code=400, detail="No clients found in database")

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
        response = requests.post(
            f"{FEDERATED_TRAINING_ORCHESTRATOR_URL}/start_fl",
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


@router.get("/simulation/status")
async def get_simulation_status():
    """
    Get current status of FL simulation from orchestrator.
    """
    try:
        response = requests.get(
            f"{FEDERATED_TRAINING_ORCHESTRATOR_URL}/simulation/status",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[FastAPI] Failed to get simulation status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get simulation status: {str(e)}"
        )


@router.get("/metrics/comparison")
async def get_metrics_comparison():
    """
    Get detailed comparison of pre-FL vs post-FL metrics from orchestrator.
    """
    try:
        response = requests.get(
            f"{FEDERATED_TRAINING_ORCHESTRATOR_URL}/metrics/comparison",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[FastAPI] Failed to get metrics comparison: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics comparison: {str(e)}"
        )


# <========================================== CONSOLIDATED FL ENDPOINT ==========================================>


@router.get("/fl_status")
async def get_fl_status():
    """
    Get current status of FL pipeline execution

    Returns:
        - is_running: Whether FL is currently running
        - current_stage: Current pipeline stage (pre_eval, simulation, post_eval, etc.)
        - progress: Overall progress percentage (0-100)
        - current_round: Current FL round (during simulation)
        - total_rounds: Total number of rounds
        - clients: Number of clients
        - started_at: When pipeline started
        - completed_at: When pipeline completed (if finished)
        - error: Error message (if failed)
        - logs: Recent log entries (last 50)
    """
    return fl_status


@router.post("/start_fl_complete")
async def start_fl_complete(
    background_tasks: BackgroundTasks,
    num_rounds: int = 5,
    lr: float = 0.001,
    local_epochs: int = 1,
    public_test_path: Optional[str] = None,
    supabase: SupabaseClient = Depends(get_supabase_client),
    settings: Settings = Depends(get_settings)
):
    """
    🚀 COMPREHENSIVE FL ENDPOINT - Consolidates entire FL pipeline:

    1. Fetch clients from Supabase
    2. Pre-FL Evaluation (baseline metrics on public test set)
    3. Run FL Simulation (via Flower)
    4. Post-FL Evaluation (final metrics on public test set)
    5. Update Supabase with all metrics

    Args:
        num_rounds: Number of federated learning rounds (default: 5)
        lr: Learning rate for client training (default: 0.001)
        local_epochs: Number of local epochs per client (default: 1)
        public_test_path: Path to public test dataset (defaults to env var)

    Returns:
        Immediate response with job ID (actual execution happens in background)
    """

    # Check if FL is already running
    global fl_status
    if fl_status["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="FL pipeline is already running. Check /api/fl_status for progress."
        )

    # Step 1: Fetch all clients from database
    logger.info("\n" + "="*70)
    logger.info("FEDERATED LEARNING - COMPLETE PIPELINE")
    logger.info("="*70)

    response = supabase.from_("clients").select("*").execute()
    clients_data = response.data if response.data else []

    if not clients_data:
        raise HTTPException(status_code=400, detail="No clients found in database")

    logger.info(f"✓ Fetched {len(clients_data)} clients from database")

    # Use environment variable for public test path if not provided
    if public_test_path is None:
        public_test_path = os.getenv("PUBLIC_TEST_PATH", "/content/drive/MyDrive/College/FLEX-Med/datasets/public_test")

    # Initialize FL status
    fl_status.update({
        "is_running": True,
        "current_stage": "initializing",
        "progress": 0,
        "total_rounds": num_rounds,
        "current_round": 0,
        "clients": len(clients_data),
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "logs": [f"FL pipeline started with {len(clients_data)} clients, {num_rounds} rounds"]
    })

    # Schedule background task for FL execution
    background_tasks.add_task(
        run_complete_fl_pipeline,
        clients_data=clients_data,
        num_rounds=num_rounds,
        lr=lr,
        local_epochs=local_epochs,
        public_test_path=public_test_path,
        supabase=supabase,
        settings=settings
    )

    return {
        "status": "started",
        "message": "Federated learning pipeline started",
        "clients": len(clients_data),
        "num_rounds": num_rounds,
        "learning_rate": lr,
        "local_epochs": local_epochs,
        "status_endpoint": "/api/fl_status",
        "note": "Use GET /api/fl_status to monitor progress in real-time"
    }


async def run_complete_fl_pipeline(
    clients_data: List[Dict],
    num_rounds: int,
    lr: float,
    local_epochs: int,
    public_test_path: str,
    supabase: SupabaseClient,
    settings: Settings
):
    """
    Background task that runs the complete FL pipeline

    Pipeline Steps:
    1. Pre-FL Evaluation
    2. Write temporary config file for FL
    3. Run FL simulation
    4. Post-FL Evaluation
    5. Update database with metrics
    """

    try:
        # ===== STEP 1: PRE-FL EVALUATION =====
        update_fl_status(stage="pre_fl_evaluation", progress=10, log="Starting pre-FL evaluation")
        logger.info("\n[PIPELINE] Step 1: Pre-FL Evaluation")
        logger.info("="*70)

        pre_fl_results = evaluate_all_clients(
            clients=clients_data,
            public_test_path=public_test_path,
            stage="pre_fl"
        )

        # Update database with pre-FL metrics
        for result in pre_fl_results:
            client_id = result['client_id']

            # Fetch existing metrics
            existing = supabase.from_("clients").select("metrics").eq("id", client_id).execute()
            existing_metrics = {}

            if existing.data and existing.data[0]['metrics']:
                metrics_str = existing.data[0]['metrics']
                if isinstance(metrics_str, str):
                    try:
                        existing_metrics = json.loads(metrics_str)
                    except json.JSONDecodeError:
                        existing_metrics = {}
                elif isinstance(metrics_str, dict):
                    existing_metrics = metrics_str

            # Add pre_fl metrics
            existing_metrics['pre_fl'] = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'specificity': result['specificity'],
                'roc_auc': result['roc_auc'],
                'healthy_accuracy': result['healthy_accuracy'],
                'leukemia_accuracy': result['leukemia_accuracy'],
                'confusion_matrix': result['confusion_matrix'],
                'num_samples': result['num_test_samples'],
                'evaluated_at': result['evaluated_at']
            }

            # Update database
            supabase.from_("clients").update({
                "metrics": json.dumps(existing_metrics)
            }).eq("id", client_id).execute()

            logger.info(f"✓ Updated pre-FL metrics for client {client_id}")

        update_fl_status(progress=25, log="Pre-FL evaluation complete, metrics saved to database")
        logger.info("✅ Pre-FL evaluation complete and saved to database\n")

        # ===== STEP 2: WRITE TEMPORARY CONFIG FOR FL =====
        update_fl_status(stage="preparing_config", progress=30, log="Preparing FL configuration")
        logger.info("[PIPELINE] Step 2: Preparing FL Configuration")
        logger.info("="*70)

        # Create temporary config file for FL simulation
        temp_config_path = Path(tempfile.gettempdir()) / "fl_config.json"

        with open(temp_config_path, 'w') as f:
            json.dump(clients_data, f, indent=2)

        update_fl_status(progress=35, log=f"Configuration prepared: {temp_config_path}")
        logger.info(f"✓ Wrote temporary config to: {temp_config_path}\n")

        # ===== STEP 3: RUN FL SIMULATION =====
        update_fl_status(stage="fl_simulation", progress=40, log=f"Starting FL simulation ({num_rounds} rounds)")
        logger.info("[PIPELINE] Step 3: Running Federated Learning Simulation")
        logger.info("="*70)

        # Determine FL directory path
        fl_dir = Path(__file__).resolve().parents[3] / "federated_learning"

        if not fl_dir.exists():
            raise FileNotFoundError(f"FL directory not found: {fl_dir}")

        logger.info(f"FL Directory: {fl_dir}")
        logger.info(f"Running {num_rounds} rounds with lr={lr}")

        # Set environment variable for FL code to pick up config
        env = os.environ.copy()
        env['FLEX_MED_CONFIG_FILE'] = str(temp_config_path)

        # Run FL simulation via subprocess
        # Dynamically set num-supernodes based on actual number of clients
        num_clients = len(clients_data)
        process = subprocess.run(
            ["flwr", "run", ".",
             f"--run-config", f"num-server-rounds={num_rounds}",
             f"--run-config", f"lr={lr}",
             f"--run-config", f"local-epochs={local_epochs}",
             f"--run-config", f"num-supernodes={num_clients}"],
            cwd=str(fl_dir),
            env=env,
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            logger.error(f"FL simulation failed with return code {process.returncode}")
            logger.error(f"STDERR: {process.stderr}")
            update_fl_status(error=f"FL simulation failed: {process.stderr}", completed=True)
            raise RuntimeError(f"FL simulation failed: {process.stderr}")

        update_fl_status(progress=70, log="FL simulation completed successfully")
        logger.info("✅ FL simulation completed successfully\n")
        logger.info(f"STDOUT:\n{process.stdout}")

        # ===== STEP 4: POST-FL EVALUATION =====
        update_fl_status(stage="post_fl_evaluation", progress=75, log="Starting post-FL evaluation")
        logger.info("[PIPELINE] Step 4: Post-FL Evaluation")
        logger.info("="*70)

        post_fl_results = evaluate_all_clients(
            clients=clients_data,
            public_test_path=public_test_path,
            stage="post_fl"
        )

        # Update database with post-FL metrics and calculate improvement
        for result in post_fl_results:
            client_id = result['client_id']

            # Fetch existing metrics (now includes pre_fl)
            existing = supabase.from_("clients").select("metrics").eq("id", client_id).execute()
            existing_metrics = {}

            if existing.data and existing.data[0]['metrics']:
                metrics_str = existing.data[0]['metrics']
                if isinstance(metrics_str, str):
                    try:
                        existing_metrics = json.loads(metrics_str)
                    except json.JSONDecodeError:
                        existing_metrics = {}
                elif isinstance(metrics_str, dict):
                    existing_metrics = metrics_str

            # Add post_fl metrics
            existing_metrics['post_fl'] = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'specificity': result['specificity'],
                'roc_auc': result['roc_auc'],
                'healthy_accuracy': result['healthy_accuracy'],
                'leukemia_accuracy': result['leukemia_accuracy'],
                'confusion_matrix': result['confusion_matrix'],
                'num_samples': result['num_test_samples'],
                'evaluated_at': result['evaluated_at']
            }

            # Calculate improvement if pre_fl exists
            if 'pre_fl' in existing_metrics:
                existing_metrics['improvement'] = calculate_improvement(
                    existing_metrics['pre_fl'],
                    existing_metrics['post_fl']
                )

            # Update database
            supabase.from_("clients").update({
                "metrics": json.dumps(existing_metrics)
            }).eq("id", client_id).execute()

            logger.info(f"✓ Updated post-FL metrics for client {client_id}")

            # Log improvement if available
            if 'improvement' in existing_metrics:
                imp = existing_metrics['improvement']
                logger.info(f"  Improvement: Accuracy {imp['accuracy']:+.2%}, F1 {imp['f1_score']:+.3f}")

        update_fl_status(progress=95, log="Post-FL evaluation complete, metrics saved to database")
        logger.info("✅ Post-FL evaluation complete and saved to database\n")

        # ===== STEP 5: CLEANUP =====
        update_fl_status(stage="cleanup", progress=98, log="Cleaning up temporary files")
        # Remove temporary config file
        temp_config_path.unlink(missing_ok=True)

        # ===== FINAL SUMMARY =====
        logger.info("\n" + "="*70)
        logger.info("✅ FEDERATED LEARNING PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Clients: {len(clients_data)}")
        logger.info(f"Rounds: {num_rounds}")
        logger.info(f"All metrics saved to Supabase database")
        logger.info("="*70 + "\n")

        # Mark as complete
        update_fl_status(
            stage="completed",
            progress=100,
            log=f"FL pipeline completed successfully! {len(clients_data)} clients trained for {num_rounds} rounds",
            completed=True
        )

    except Exception as e:
        logger.error(f"\n❌ FL Pipeline failed: {e}")
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(traceback.format_exc())
        update_fl_status(error=str(e), log=f"Pipeline failed: {str(e)}", completed=True)
        raise
