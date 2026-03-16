from fastapi import APIRouter, Depends, HTTPException
from supabase import Client as SupabaseClient
from app.config import get_supabase_client
from app.schemas.client_simulation_metrics import (
    ClientSimulationMetrics,
    ClientSimulationMetricsCreate,
    ClientSimulationMetricsUpdate,
)
from typing import List
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/client-simulation-metrics", tags=["client-simulation-metrics"])


@router.post("/", response_model=ClientSimulationMetrics)
async def create_client_simulation_metrics(
    data: ClientSimulationMetricsCreate,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    try:
        response = supabase.from_("client_simulation_metrics").insert(data.dict()).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create metrics record")

        logger.info(f"Created metrics record for client {data.client_id} in simulation {data.simulation_id}")
        return response.data[0]

    except Exception as e:
        logger.error(f"Error creating client simulation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/{simulation_id}", response_model=List[ClientSimulationMetrics])
async def get_metrics_by_simulation(
    simulation_id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    try:
        response = supabase.from_("client_simulation_metrics") \
            .select("*, clients(client_name, model_type)") \
            .eq("simulation_id", simulation_id) \
            .execute()

        return response.data or []

    except Exception as e:
        logger.error(f"Error fetching metrics for simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{id}", response_model=ClientSimulationMetrics)
async def update_client_simulation_metrics(
    id: int,
    data: ClientSimulationMetricsUpdate,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    try:
        # Only include non-None fields in update
        update_dict = {k: v for k, v in data.dict().items() if v is not None}

        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        response = supabase.from_("client_simulation_metrics") \
            .update(update_dict) \
            .eq("id", id) \
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Metrics record not found")

        logger.info(f"Updated metrics record {id}")
        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating metrics record {id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

