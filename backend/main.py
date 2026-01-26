import uvicorn
import os
from pathlib import Path
import federated_learning.flex_med.utils.config as config

if __name__ == "__main__":
    # Set environment variables for FL simulation paths
    # backend root
    backend_root = Path(__file__).parent
    # project root (FLEX-Med) - still useful if needed, but FL is now inside backend
    project_root = backend_root.parent
    
    os.environ["BASE_PATH"] = str(backend_root) # FL expects BASE_PATH to be parent of dataset/checkpoints?
    # In unified config: base_path is backend root.
    # But FL config used BASE_PATH for datasets/checkpoints.
    # User likely has datasets in FLEX-Med/datasets?
    # Let's set BASE_PATH to backend_root or project_root depending on where datasets are.
    # In config.py I set dataset_base_path = BASE_PATH or parent of backend.
    
    # We'll set FLEX_MED_PROJECT_DIR to the internal FL dir
    os.environ["FLEX_MED_PROJECT_DIR"] = str(backend_root / "federated_learning")
    
    print(config.PUBLIC_ANCHOR_DATASET_PATH)    
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )