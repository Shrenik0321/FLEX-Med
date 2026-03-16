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

