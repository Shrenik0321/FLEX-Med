from fastapi import APIRouter, Depends, HTTPException
from supabase import Client
from typing import List

from app.schemas.client import Client, ClientCreate
from app.config import get_supabase_client

router = APIRouter()

@router.get("/clients", response_model=List[Client])
def get_clients(supabase: Client = Depends(get_supabase_client)):
    response = supabase.from_("clients").select("*").execute()
    if response.data is None:
        raise HTTPException(status_code=404, detail="Clients not found")
    return response.data

@router.get("/clients/{client_id}", response_model=Client)
def get_client(client_id: int, supabase: Client = Depends(get_supabase_client)):
    response = supabase.from_("clients").select("*").eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    return response.data[0]

@router.post("/clients", response_model=Client)
def create_client(client: ClientCreate, supabase: Client = Depends(get_supabase_client)):
    response = supabase.from_("clients").insert(client.dict()).execute()
    if len(response.data) == 0:
        raise HTTPException(status_code=400, detail="Client could not be created")
    return response.data[0]

@router.put("/clients/{client_id}", response_model=Client)
def update_client(client_id: int, client: ClientCreate, supabase: Client = Depends(get_supabase_client)):
    response = supabase.from_("clients").update(client.dict()).eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    return response.data[0]

@router.delete("/clients/{client_id}")
def delete_client(client_id: int, supabase: Client = Depends(get_supabase_client)):
    response = supabase.from_("clients").delete().eq("id", client_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Client not found")
    return {"ok": True}
