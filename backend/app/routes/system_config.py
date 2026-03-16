"""API routes for system configuration management."""
from fastapi import APIRouter, HTTPException, Depends
from supabase import Client as SupabaseClient
from app.config import get_supabase_client
from app.schemas.system_config import (
    SystemConfig, SystemConfigUpdate, SystemConfigData,
)
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system_config", response_model=SystemConfig)
async def get_system_config(
    supabase: SupabaseClient = Depends(get_supabase_client)
):
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

