from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import requests
import json
import subprocess
import tempfile
import os
from pathlib import Path
from supabase import Client as SupabaseClient
from app.config import get_supabase_client, get_settings, Settings
from app.services.fl_evaluation_service import evaluate_all_clients, calculate_improvement
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

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
            "https://79233b82dab3.ngrok-free.app/start_fl",
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
            "https://541103493561.ngrok-free.app/simulation/status",
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
            "https://541103493561.ngrok-free.app/metrics/comparison",
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
        "note": "Check logs for progress. Metrics will be updated in database when complete."
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

        logger.info("✅ Pre-FL evaluation complete and saved to database\n")

        # ===== STEP 2: WRITE TEMPORARY CONFIG FOR FL =====
        logger.info("[PIPELINE] Step 2: Preparing FL Configuration")
        logger.info("="*70)

        # Create temporary config file for FL simulation
        temp_config_path = Path(tempfile.gettempdir()) / "fl_config.json"

        with open(temp_config_path, 'w') as f:
            json.dump(clients_data, f, indent=2)

        logger.info(f"✓ Wrote temporary config to: {temp_config_path}\n")

        # ===== STEP 3: RUN FL SIMULATION =====
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
        process = subprocess.run(
            ["flwr", "run", ".",
             f"--run-config", f"num-server-rounds={num_rounds}",
             f"--run-config", f"lr={lr}",
             f"--run-config", f"local-epochs={local_epochs}"],
            cwd=str(fl_dir),
            env=env,
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            logger.error(f"FL simulation failed with return code {process.returncode}")
            logger.error(f"STDERR: {process.stderr}")
            raise RuntimeError(f"FL simulation failed: {process.stderr}")

        logger.info("✅ FL simulation completed successfully\n")
        logger.info(f"STDOUT:\n{process.stdout}")

        # ===== STEP 4: POST-FL EVALUATION =====
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

        logger.info("✅ Post-FL evaluation complete and saved to database\n")

        # ===== STEP 5: CLEANUP =====
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

    except Exception as e:
        logger.error(f"\n❌ FL Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
