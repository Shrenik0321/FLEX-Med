from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import requests
import requests
import json
from supabase import Client as SupabaseClient
from app.config import get_supabase_client

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
            "https://6f816048fc47.ngrok-free.app/start_fl",
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
