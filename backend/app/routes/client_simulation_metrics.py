from fastapi import APIRouter, Depends, HTTPException
from supabase import Client as SupabaseClient
from app.config import get_supabase_client
from app.schemas.client_simulation_metrics import (
    ClientSimulationMetrics,
    ClientSimulationMetricsCreate,
    ClientSimulationMetricsUpdate,
    ClientSimulationStatus,
    get_improvement_summary,
    get_latest_round_metrics
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
    """
    Create a new metrics record for a client in a simulation.

    This endpoint is typically called when starting a new FL simulation
    to initialize metrics records for all participating clients.
    """
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
    """
    Get all client metrics for a specific simulation.

    Returns metrics for all clients that participated in the given simulation,
    including client details (name, model_type) via join.
    """
    try:
        response = supabase.from_("client_simulation_metrics") \
            .select("*, clients(client_name, model_type)") \
            .eq("simulation_id", simulation_id) \
            .execute()

        return response.data or []

    except Exception as e:
        logger.error(f"Error fetching metrics for simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/client/{client_id}", response_model=List[ClientSimulationMetrics])
async def get_metrics_by_client(
    client_id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get all simulation metrics for a specific client (history).

    Returns the full history of all simulations this client has participated in,
    ordered by most recent first. Useful for tracking client performance over time.
    """
    try:
        response = supabase.from_("client_simulation_metrics") \
            .select("*") \
            .eq("client_id", client_id) \
            .order("created_at", desc=True) \
            .execute()

        return response.data or []

    except Exception as e:
        logger.error(f"Error fetching metrics for client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{id}", response_model=ClientSimulationMetrics)
async def get_client_simulation_metrics(
    id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get a specific client-simulation metrics record by ID.
    """
    try:
        response = supabase.from_("client_simulation_metrics") \
            .select("*") \
            .eq("id", id) \
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Metrics record not found")

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics record {id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{id}", response_model=ClientSimulationMetrics)
async def update_client_simulation_metrics(
    id: int,
    data: ClientSimulationMetricsUpdate,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Update metrics for a specific client-simulation pair.

    This endpoint is typically called by the FL system to update metrics
    during training (pre_fl, rounds, post_fl).
    """
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


@router.get("/simulation/{simulation_id}/summary")
async def get_simulation_summary(
    simulation_id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get a summary of metrics for all clients in a simulation.

    Returns:
        - List of clients with their improvement metrics
        - Latest round metrics for each client
        - Overall simulation statistics
    """
    try:
        # Fetch all client metrics for this simulation
        response = supabase.from_("client_simulation_metrics") \
            .select("*, clients(client_name, model_type)") \
            .eq("simulation_id", simulation_id) \
            .execute()

        if not response.data:
            return {
                "simulation_id": simulation_id,
                "clients": [],
                "total_clients": 0
            }

        # Build summary for each client
        client_summaries = []
        for record in response.data:
            metrics = record.get("metrics", {})

            summary = {
                "client_id": record.get("client_id"),
                "client_name": record.get("clients", {}).get("client_name"),
                "model_type": record.get("clients", {}).get("model_type"),
                "status": record.get("status"),
                "improvement": get_improvement_summary(metrics),
                "latest_round": get_latest_round_metrics(metrics),
                "total_rounds": len(metrics.get("rounds", []))
            }

            client_summaries.append(summary)

        return {
            "simulation_id": simulation_id,
            "clients": client_summaries,
            "total_clients": len(client_summaries),
            "completed_clients": sum(1 for c in client_summaries if c["status"] == "completed"),
            "failed_clients": sum(1 for c in client_summaries if c["status"] == "failed")
        }

    except Exception as e:
        logger.error(f"Error fetching simulation summary {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{id}")
async def delete_client_simulation_metrics(
    id: int,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Delete a specific client-simulation metrics record.

    WARNING: This will permanently delete the metrics data.
    Consider using status updates instead for most use cases.
    """
    try:
        response = supabase.from_("client_simulation_metrics") \
            .delete() \
            .eq("id", id) \
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Metrics record not found")

        logger.warning(f"Deleted metrics record {id}")
        return {"message": "Metrics record deleted successfully", "id": id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting metrics record {id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
