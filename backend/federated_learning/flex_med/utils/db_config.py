"""flex-med: Database Configuration Helper."""

from supabase import Client
import toml
import os
import sys
from pathlib import Path

# Try to import from app.config
try:
    from app.config import get_settings, get_supabase_client
except ImportError:
    current_file = Path(__file__).resolve()
    backend_root = current_file.parent.parent.parent.parent
    sys.path.append(str(backend_root))
    try:
        from app.config import get_settings, get_supabase_client
    except ImportError:
        from backend.app.config import get_settings, get_supabase_client


# <--- CREDENTIALS --->
# Retrieved from unified settings
_settings = get_settings()

# <--- PROJECT CONFIG PATH --->
PYPROJECT_PATH = str(_settings.pyproject_path)

def get_client_configuration(partition_id: int):
    """
    Connects to Supabase and fetches config for the specific partition_id.
    
    Maps Flower partition_id (0, 1, 2) to Database ID (1, 2, 3).
    """
    try:
        # 1. Initialize
        supabase = get_supabase_client()
        
        # 2. Map Partition ID (0-indexed) to DB ID (1-indexed)
        # If your DB IDs are 0, 1, 2, remove the '+ 1'
        db_id = partition_id + 1 
        
        print(f"[DB] Fetching config for Partition {partition_id} (DB ID {db_id})...")
        
        # 3. Query
        response = supabase.table('clients').select('*').eq('id', db_id).execute()
        
        # 4. Validate and return
        if response.data and len(response.data) > 0:
            data = response.data[0]
            
            # Validate required fields
            required_fields = ['client_name', 'model_type', 'dataset_url']
            missing_fields = [f for f in required_fields if f not in data]
            
            if missing_fields:
                print(f"[DB] Warning: Missing fields in DB: {missing_fields}")
            
            # Validate model_type
            valid_models = ['resnet', 'resnet18', 'mobilenet', 'mobilenet_v2', 'densenet', 'densenet121']
            model_type = data.get('model_type', '').lower()
            if model_type not in valid_models:
                print(f"[DB] Warning: Invalid model_type '{model_type}'. Valid options: {valid_models}")
            
            print(f"[DB] Successfully loaded config for {data.get('client_name', 'Unknown')}")
            return data
        else:
            print(f"[DB] Error: No config found for Partition {partition_id} (DB ID {db_id})")
            return None

    except Exception as e:
        print(f"[DB] Connection Error: {e}")
        return None


def sync_client_count_to_config():
    """
    Queries database for client count and automatically updates pyproject.toml.
    
    This ensures the number of supernodes in your federated learning simulation
    matches the number of clients in your database.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("\n" + "="*70)
        print("🔄 Syncing Client Count to Configuration")
        print("="*70)
        
        # Step 1: Query database for client count
        supabase = get_supabase_client()
        response = supabase.table('clients').select('id', count='exact').execute()
        
        # Handle different response formats
        num_clients = response.count if hasattr(response, 'count') else len(response.data)
        
        print(f"📊 Database clients found: {num_clients}")
        
        if num_clients == 0:
            print("⚠️  Warning: No clients in database!")
            print("   Add clients before running simulation.")
            return False
        
        # Step 2: Read pyproject.toml
        if not os.path.exists(PYPROJECT_PATH):
            print(f"❌ Error: pyproject.toml not found at {PYPROJECT_PATH}")
            return False
        
        with open(PYPROJECT_PATH, 'r') as f:
            config = toml.load(f)
        
        # Step 3: Get current value
        current_value = config.get('tool', {}) \
                             .get('flwr', {}) \
                             .get('federations', {}) \
                             .get('local-simulation', {}) \
                             .get('options', {}) \
                             .get('num-supernodes')
        
        # Step 4: Update only the num-supernodes value (preserves all other structure)
        config['tool']['flwr']['federations']['local-simulation']['options']['num-supernodes'] = num_clients
        
        # Step 5: Write back to file
        with open(PYPROJECT_PATH, 'w') as f:
            toml.dump(config, f)
        
        # Step 6: Report results
        if current_value != num_clients:
            print(f"✅ Updated num-supernodes: {current_value} → {num_clients}")
        else:
            print(f"✅ Already synced (num-supernodes = {num_clients})")
        
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Error syncing configuration: {e}")
        print("="*70 + "\n")
        return False


# ============================================================================
# AUTO-SYNC ON MODULE IMPORT
# ============================================================================
# Automatically sync client count when this module is imported.
# This ensures pyproject.toml is always up-to-date with the database.
# ============================================================================

try:
    # Only auto-sync if pyproject.toml exists (skip during package installation)
    if os.path.exists(PYPROJECT_PATH):
        sync_client_count_to_config()
except Exception as e:
    print(f"[DB] Warning: Auto-sync failed on import: {e}")
    print("[DB] You can manually call sync_client_count_to_config() if needed")
