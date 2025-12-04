import os
from supabase import create_client, Client

# <--- CONFIGURATION --->
# Replace these with your actual Supabase Project URL and Anon Key
# You can find these in Supabase Dashboard -> Project Settings -> API
SUPABASE_URL = "https://xyzcompany.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI..."

def init_supabase() -> Client:
    """Initialize the Supabase client."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[DB] Connection initialized successfully.")
        return client
    except Exception as e:
        print(f"[DB] Error connecting to Supabase: {e}")
        return None

def get_client_config(supabase: Client, partition_id: int):
    """
    Fetch configuration for a specific client partition_id.
    """
    try:
        print(f"[DB] Fetching config for Partition ID: {partition_id}...")
        
        # This matches the SQL: SELECT * FROM client_configs WHERE partition_id = X
        response = supabase.table("client_configs") \
            .select("*") \
            .eq("partition_id", partition_id) \
            .execute()

        # Check if data was returned
        if response.data:
            data = response.data[0] # Get the first (and only) row
            print(f"[DB] Success! Received data for {data.get('client_name', 'Unknown Client')}")
            return data
        else:
            print(f"[DB] Warning: No record found for Partition ID {partition_id}")
            return None

    except Exception as e:
        print(f"[DB] Query Failed: {e}")
        return None

# <--- MAIN TEST EXECUTION --->
if __name__ == "__main__":
    # 1. Initialize
    db_client = init_supabase()

    # 2. Test fetching specific clients (Simulating your partition logic)
    if db_client:
        
        # Test Case 1: Client 0 (Should be ResNet / ALL-IDB)
        config_0 = get_client_config(db_client, 0)
        print(f"   -> Data: {config_0}\n")

        # Test Case 2: Client 1 (Should be MobileNet / CNMC)
        config_1 = get_client_config(db_client, 1)
        print(f"   -> Data: {config_1}\n")

        # Test Case 3: Client 2 (Should be DenseNet / Free Rider)
        config_2 = get_client_config(db_client, 2)
        print(f"   -> Data: {config_2}\n")