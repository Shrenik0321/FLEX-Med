import requests
from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from supabase import Client as SupabaseClient
from typing import List
from pydantic import BaseModel
import os
import logging

from app.schemas.client import Client, ClientCreate
from app.config import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter()

# Ngrok URL for local training orchestrator (update when Colab session changes)
LOCAL_TRAIN_ORCHESTRATOR_URL = os.getenv(
    "LOCAL_TRAIN_ORCHESTRATOR_URL",
    "https://intraspinal-agape-deidra.ngrok-free.dev"
)

def fix_client_data(client: dict) -> dict:
    # 1. Fix Status: Map invalid status to Inactive
    if client.get("status") not in ["Active", "Inactive"]:
        client["status"] = "Inactive"
    
    # 2. Fix has_local_data: Default to False if None
    if client.get("has_local_data") is None:
        client["has_local_data"] = False
        
    # 3. Fix model_path: Construct default if missing
    if not client.get("model_path"):
        name = client.get("client_name", "unknown")
        client["model_path"] = f"/content/drive/MyDrive/College/models/{name}.pt"
        
    # 4. Fix metrics: Default structure if None
    if client.get("metrics") is None:
        client["metrics"] = {
            "pre_fl_acc": 0.0,
            "post_fl_acc": 0.0,
            "last_updated": None
        }
    return client

@router.get("/clients", response_model=List[Client])
def get_clients(supabase: SupabaseClient = Depends(get_supabase_client)):
    response = supabase.from_("clients").select("*").execute()
    if response.data is None:
        raise HTTPException(status_code=404, detail="Clients not found")
    
    # Fix incompatible data from DB to match new Schema
    cleaned_data = [fix_client_data(c) for c in response.data]
    return cleaned_data

@router.get("/clients/{client_id}", response_model=Client)
def get_client(client_id: int, supabase: SupabaseClient = Depends(get_supabase_client)):
    response = supabase.from_("clients").select("*").eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Fix incompatible data
    return fix_client_data(response.data[0])

@router.post("/clients", response_model=Client)
def create_client(client: ClientCreate, supabase: SupabaseClient = Depends(get_supabase_client)):
    client_data = jsonable_encoder(client)
    # Set Metrics
    client_data["metrics"] = {}
    
    # Ensure consistency: if no local data, dataset_path must be null
    if not client.has_local_data:
        client_data["dataset_path"] = None

    if not client_data.get("model_path"):
        # Construct default model path: /content/drive/MyDrive/College/models/{client_name}.pt
        client_data["model_path"] = f"/content/drive/MyDrive/College/models/{client.client_name}.pt"
    
    response = supabase.from_("clients").insert(client_data).execute()
    if len(response.data) == 0:
        raise HTTPException(status_code=400, detail="Client could not be created")

    # Trigger model creation
    try:
        requests.post(
            f"{LOCAL_TRAIN_ORCHESTRATOR_URL}/enroll_client",
            json={"client_name": client.client_name, "model_type": client.model_type}
        )
    except Exception as e:
        print(f"Failed to trigger model creation: {e}")

    return response.data[0]

@router.post("/clients/{client_id}/start_local_train")
async def start_local_train(
    client_id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Start local training for a specific client.

    Fetches client details from Supabase and forwards request to the
    Colab orchestrator (ngrok endpoint).

    Args:
        client_id: ID of the client to train
        epochs: Number of training epochs (default: 5)
        batch_size: Batch size for training (default: 16)
        learning_rate: Learning rate (default: 0.001)

    Returns:
        Training status and configuration
    """
    # Configs
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001

    # Fetch client from database
    response = supabase.from_("clients").select("*").eq("id", client_id).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail=f"Client with ID {client_id} not found")

    client = response.data[0]

    # Validate client has required data
    if not client.get('dataset_path'):
        raise HTTPException(
            status_code=400,
            detail=f"Client '{client.get('client_name')}' does not have a dataset path configured"
        )

    if not client.get('model_path'):
        raise HTTPException(
            status_code=400,
            detail=f"Client '{client.get('client_name')}' does not have a model path configured"
        )

    # Prepare request for orchestrator
    train_payload = {
        "client_name": client['client_name'],
        "dataset_path": client['dataset_path'],
        "model_path": client['model_path'],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    logger.info(f"Starting local training for client: {client['client_name']}")
    logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    try:
        response = requests.post(
            f"{LOCAL_TRAIN_ORCHESTRATOR_URL}/start_local_train",
            json=train_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        result = response.json()

        # Update client status to indicate training in progress
        supabase.from_("clients").update({
            "status": "Training"
        }).eq("id", client_id).execute()

        return {
            "status": "started",
            "client_id": client_id,
            "client_name": client['client_name'],
            "message": result.get("message", "Training started"),
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to start local training: {e}")
        if hasattr(e, 'response') and e.response is not None:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Orchestrator Error: {e.response.text}"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to training orchestrator: {str(e)}"
        )

@router.get("/clients/{client_name}/train_status")
async def get_train_status(client_name: str):
    """
    Get training status and logs for a specific client.

    Polls the Colab orchestrator for real-time training logs.

    Args:
        client_name: Name of the client to get status for

    Returns:
        Training status ("pending", "running", "completed", "failed") and logs
    """
    try:
        response = requests.get(
            f"{LOCAL_TRAIN_ORCHESTRATOR_URL}/train_status/{client_name}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {
                "status": "not_found",
                "logs": [],
                "message": f"No training logs found for client '{client_name}'"
            }
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error fetching training status: {e.response.text}"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to training orchestrator: {str(e)}"
        )

@router.post("/clients/{client_name}/train_complete")
async def mark_training_complete(
    client_name: str,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Mark training as complete and update client status.

    Called by frontend when training finishes to reset client status.

    Args:
        client_name: Name of the client that completed training

    Returns:
        Updated client status
    """
    # Find client by name
    response = supabase.from_("clients").select("*").eq("client_name", client_name).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail=f"Client '{client_name}' not found")

    client = response.data[0]

    # Update client status back to Active
    supabase.from_("clients").update({
        "status": "Active"
    }).eq("id", client['id']).execute()

    logger.info(f"Training completed for client: {client_name}")

    return {
        "status": "completed",
        "client_id": client['id'],
        "client_name": client_name,
        "message": f"Training marked as complete for {client_name}"
    }

@router.get("/local_training/logs")
async def get_local_training_logs(lines: int = 100):
    """
    Fetch recent lines from the central local_training.log file.

    This is the unified log that captures all local training activities,
    similar to simulation_run.log for federated learning.

    Args:
        lines: Number of recent lines to return (default: 100)

    Returns:
        JSON object with log lines and metadata
    """
    try:
        response = requests.get(
            f"{LOCAL_TRAIN_ORCHESTRATOR_URL}/logs",
            params={"lines": lines},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch training logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch training logs: {str(e)}"
        )

@router.get("/local_training/logs/tail")
async def tail_local_training_logs(lines: int = 50):
    """
    Get last N lines from central training log as plain text.
    Perfect for real-time monitoring in the UI.

    Args:
        lines: Number of recent lines to return (default: 50)

    Returns:
        Plain text with recent log entries
    """
    try:
        response = requests.get(
            f"{LOCAL_TRAIN_ORCHESTRATOR_URL}/logs/tail",
            params={"lines": lines},
            timeout=10
        )
        response.raise_for_status()
        return response.text

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to tail training logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to tail training logs: {str(e)}"
        )

@router.put("/clients/{client_id}", response_model=Client)
def update_client(client_id: int, client: ClientCreate, supabase: SupabaseClient = Depends(get_supabase_client)):
    response = supabase.from_("clients").update(jsonable_encoder(client)).eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    return response.data[0]

@router.delete("/clients/{client_id}")
def delete_client(client_id: int, supabase: SupabaseClient = Depends(get_supabase_client)):
    response = supabase.from_("clients").delete().eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    return {"ok": True}
