import requests
from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from supabase import Client as SupabaseClient
from typing import List

from app.schemas.client import Client, ClientCreate
from app.config import get_supabase_client

router = APIRouter()

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
            "https://intraspinal-agape-deidra.ngrok-free.dev/enroll_client",
            json={"client_name": client.client_name, "model_type": client.model_type}
        )
    except Exception as e:
        print(f"Failed to trigger model creation: {e}")

    return response.data[0]

@router.post("/clients/{client_id}/start_local_train", response_model=Client)
def start_local_train(client_id: int, supabase: SupabaseClient = Depends(get_supabase_client)):
    # 1. Fetch Client
    response = supabase.from_("clients").select("*").eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # 2. Fix data
    client = fix_client_data(response.data[0])

    # 3. Trigger local training
    try:
        requests.post(
            "https://381672b1ebec.ngrok-free.app/start_local_train",
            json={
                "client_name": client.get("client_name"),
                "dataset_path": client.get("dataset_path"),
                "model_path": client.get("model_path"),
                "epochs": 5
            }
        )
    except Exception as e:
        print(f"Failed to trigger local training: {e}")

    return client

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
