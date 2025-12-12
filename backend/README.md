# FLEX-Med FastAPI backend

## Setup
1) Create and activate a virtual environment.
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Set environment variables (optional):
```
export FLEX_MED_MODEL_PATH=/absolute/path/to/model_client_0.pt
export FLEX_MED_PUBLIC_DATA_PATH=/absolute/path/to/public_anchor
export FLEX_MED_DEVICE=cuda  # or cpu
```

## Run locally
```
python -m backend.main
```

The API will be available at http://localhost:8000 and the docs at http://localhost:8000/docs.

## Endpoints
- `GET /health` — health check.
- `POST /predict/upload` — send an image file (form-data key: `file`) and receive a prediction.
- `GET /predict/random` — sample a random image from `FLEX_MED_PUBLIC_DATA_PATH` and predict (requires the env var above).

