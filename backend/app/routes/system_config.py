"""API routes for system configuration management."""
from fastapi import APIRouter, HTTPException, Depends
from supabase import Client as SupabaseClient
from app.config import get_supabase_client
from app.schemas.system_config import SystemConfig, SystemConfigUpdate, SystemConfigData
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system_config", response_model=SystemConfig)
async def get_system_config(
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get the current system configuration.
    
    Returns the singleton system_config record from the database.
    """
    try:
        response = supabase.from_("system_config").select("*").limit(1).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=404,
                detail="System configuration not found. Please run database migrations."
            )
        
        config_record = response.data[0]
        
        # Parse the JSONB config field into SystemConfigData
        return SystemConfig(
            id=config_record["id"],
            config=SystemConfigData(**config_record["config"]),
            created_at=config_record["created_at"],
            updated_at=config_record["updated_at"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching system config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.put("/system_config", response_model=SystemConfig)
async def update_system_config(
    config_update: SystemConfigUpdate,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Update the system configuration.
    
    Replaces the entire configuration with the provided data.
    Uses singleton pattern - always updates the first (and only) record.
    """
    try:
        # Fetch the existing config to get its ID
        response = supabase.from_("system_config").select("id").limit(1).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=404,
                detail="System configuration not found. Please run database migrations."
            )
        
        config_id = response.data[0]["id"]
        
        # Update the configuration
        update_data = {
            "config": config_update.config.model_dump(),
            "updated_at": datetime.now().isoformat()
        }
        
        response = supabase.from_("system_config").update(update_data).eq("id", config_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to update system configuration")
        
        updated_record = response.data[0]
        
        logger.info(f"System configuration updated successfully")
        
        return SystemConfig(
            id=updated_record["id"],
            config=SystemConfigData(**updated_record["config"]),
            created_at=updated_record["created_at"],
            updated_at=updated_record["updated_at"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.patch("/system_config")
async def partial_update_system_config(
    config_updates: dict,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Partially update system configuration.
    
    Only updates the fields provided in the request body.
    Other fields remain unchanged.
    """
    try:
        # Fetch current config
        response = supabase.from_("system_config").select("*").limit(1).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=404,
                detail="System configuration not found. Please run database migrations."
            )
        
        current_record = response.data[0]
        config_id = current_record["id"]
        current_config = current_record["config"]
        
        # Merge updates into current config
        updated_config = {**current_config, **config_updates}
        
        # Validate the merged config
        try:
            SystemConfigData(**updated_config)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
        
        # Update the database
        update_data = {
            "config": updated_config,
            "updated_at": datetime.now().isoformat()
        }
        
        response = supabase.from_("system_config").update(update_data).eq("id", config_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to update system configuration")
        
        updated_record = response.data[0]
        
        logger.info(f"System configuration partially updated: {list(config_updates.keys())}")
        
        return SystemConfig(
            id=updated_record["id"],
            config=SystemConfigData(**updated_record["config"]),
            created_at=updated_record["created_at"],
            updated_at=updated_record["updated_at"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error partially updating system config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
