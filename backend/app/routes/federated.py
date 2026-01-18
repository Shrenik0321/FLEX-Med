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
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Load orchestrator URL from settings
# Ngrok URL for local training orchestrator (update when Colab session changes)
FEDERATED_TRAINING_ORCHESTRATOR_URL = "https://2357626794df.ngrok-free.app"

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