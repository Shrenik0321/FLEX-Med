from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from supabase import Client as SupabaseClient
from typing import List
import logging

from app.schemas.client import Client, ClientCreate
from app.config import get_supabase_client, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


def fix_client_data(client: dict) -> dict:
    """Fix legacy or incomplete client data to match current schema."""
    settings = get_settings()

    # 1. Fix Status: Map invalid status to Inactive
    if client.get("status") not in ["Active", "Inactive"]:
        client["status"] = "Inactive"

    # 2. Fix model_path: Construct default if missing using centralized config
    if not client.get("model_path"):
        name = client.get("client_name", "unknown")
        client["model_path"] = str(settings.default_model_dir / f"{name}.pt")

    # Note: Metrics are now stored in client_simulation_metrics table, not here
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
    """
    Create a new client.

    Note: Metrics are stored separately in client_simulation_metrics table.
    This endpoint only creates the client configuration.
    """
    settings = get_settings()
    client_data = jsonable_encoder(client)

    # Set default model path using centralized config
    if not client_data.get("model_path"):
        client_data["model_path"] = str(settings.default_model_dir / f"{client.client_name}.pt")

    response = supabase.from_("clients").insert(client_data).execute()
    if len(response.data) == 0:
        raise HTTPException(status_code=400, detail="Client could not be created")

    logger.info(f"Created client: {client.client_name} (ID: {response.data[0]['id']})")

    return response.data[0]

@router.put("/clients/{client_id}", response_model=Client)
def update_client(client_id: int, client: ClientCreate, supabase: SupabaseClient = Depends(get_supabase_client)):
    response = supabase.from_("clients").update(jsonable_encoder(client)).eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    return response.data[0]

@router.delete("/clients/{client_id}")
def delete_client(client_id: int, supabase: SupabaseClient = Depends(get_supabase_client)):
    """
    Delete a client.

    WARNING: This will cascade delete all associated client_simulation_metrics records.
    """
    response = supabase.from_("clients").delete().eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    return {"ok": True}


@router.get("/clients/{client_id}/metrics-history")
def get_client_metrics_history(
    client_id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get the full metrics history for a specific client across all simulations.

    Returns a list of all simulations this client participated in, with their metrics.
    Useful for tracking client performance over time.
    """
    try:
        # First verify client exists
        client_response = supabase.from_("clients").select("*").eq("id", client_id).execute()
        if not client_response.data:
            raise HTTPException(status_code=404, detail="Client not found")

        client = client_response.data[0]

        # Fetch all metrics for this client across simulations
        metrics_response = supabase.from_("client_simulation_metrics") \
            .select("*, fl_simulations(id, created_at, configs, status)") \
            .eq("client_id", client_id) \
            .order("created_at", desc=True) \
            .execute()

        history = []
        for record in metrics_response.data or []:
            simulation = record.get("fl_simulations", {})
            metrics = record.get("metrics", {})

            # Extract key metrics for quick reference
            pre_fl = metrics.get("global", {}).get("pre_fl", {})
            post_fl = metrics.get("global", {}).get("post_fl", {})
            improvement = metrics.get("global", {}).get("improvement", {})

            history.append({
                "simulation_id": simulation.get("id"),
                "simulation_created_at": simulation.get("created_at"),
                "simulation_status": simulation.get("status"),
                "client_status": record.get("status"),
                "num_rounds": len(metrics.get("rounds", [])),
                "pre_fl_accuracy": pre_fl.get("accuracy"),
                "post_fl_accuracy": post_fl.get("accuracy"),
                "improvement_accuracy": improvement.get("accuracy"),
                "pre_fl_loss": pre_fl.get("loss"),
                "post_fl_loss": post_fl.get("loss"),
                "full_metrics": metrics  # Include full metrics for detailed view
            })

        return {
            "client_id": client_id,
            "client_name": client.get("client_name"),
            "model_type": client.get("model_type"),
            "total_simulations": len(history),
            "history": history
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics history for client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
