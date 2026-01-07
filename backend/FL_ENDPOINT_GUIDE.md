# 🚀 Federated Learning Endpoint Guide

## Overview

The `/api/start_fl_complete` endpoint consolidates the **entire federated learning pipeline** into a single backend call, eliminating the need for Colab orchestration and manual data.json management.

## 📋 What It Does

### Complete FL Pipeline:
1. **Fetch Clients** - Retrieves all client configurations from Supabase
2. **Pre-FL Evaluation** - Evaluates all models on public test dataset (baseline metrics)
3. **FL Simulation** - Runs Flower federated learning simulation locally
4. **Post-FL Evaluation** - Evaluates trained models on public test dataset
5. **Database Update** - Saves all metrics (pre-FL, post-FL, improvement) to Supabase

### No More:
- ❌ Manual data.json file management
- ❌ Ngrok URL dependencies
- ❌ Colab orchestration notebooks
- ❌ Separate pre/post evaluation scripts

## 🔧 Setup

### 1. Environment Variables

Create/update `backend/.env`:

```bash
# Required
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Optional (defaults to Colab path if not set)
PUBLIC_TEST_PATH=/path/to/public/test/dataset

# For production deployment
FLEX_MED_CONFIG_FILE=/tmp/fl_config.json  # Auto-managed, no manual intervention needed
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt

# Ensure FL dependencies are installed
cd ../federated_learning
pip install -e .
```

### 3. Verify Paths

Ensure the following directory structure exists:

```
FLEX-Med/
├── backend/
│   └── app/
│       ├── routes/federated.py  # New endpoint
│       └── services/
│           └── fl_evaluation_service.py  # Evaluation logic
└── federated_learning/
    ├── flex_med/
    │   ├── task_new.py         # Updated to support runtime config
    │   ├── client_app_new.py
    │   └── server_app_new.py
    └── pyproject.toml
```

## 📡 API Usage

### Endpoint

```http
POST /api/start_fl_complete
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rounds` | int | 5 | Number of federated learning rounds |
| `lr` | float | 0.001 | Learning rate for client training |
| `local_epochs` | int | 1 | Number of local epochs per client |
| `public_test_path` | str | env var | Path to public test dataset |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/start_fl_complete" \
  -H "Content-Type: application/json" \
  -d '{
    "num_rounds": 10,
    "lr": 0.0005,
    "local_epochs": 2
  }'
```

### Response (Immediate)

```json
{
  "status": "started",
  "message": "Federated learning pipeline started",
  "clients": 3,
  "num_rounds": 10,
  "learning_rate": 0.0005,
  "local_epochs": 2,
  "note": "Check logs for progress. Metrics will be updated in database when complete."
}
```

**Note**: The endpoint returns immediately. The FL pipeline runs in the background. Monitor server logs for progress.

## 📊 Metrics Structure

After completion, each client in the database will have updated metrics:

```json
{
  "id": 0,
  "client_name": "Hospital_A",
  "model_type": "resnet18",
  "metrics": {
    "pre_fl": {
      "accuracy": 0.7234,
      "precision": 0.712,
      "recall": 0.698,
      "f1_score": 0.705,
      "specificity": 0.748,
      "roc_auc": 0.723,
      "healthy_accuracy": 0.752,
      "leukemia_accuracy": 0.695,
      "confusion_matrix": [[150, 50], [45, 155]],
      "num_samples": 400,
      "evaluated_at": "2026-01-07T10:30:00"
    },
    "post_fl": {
      "accuracy": 0.9125,
      "precision": 0.901,
      "recall": 0.918,
      "f1_score": 0.909,
      "specificity": 0.907,
      "roc_auc": 0.913,
      "healthy_accuracy": 0.906,
      "leukemia_accuracy": 0.919,
      "confusion_matrix": [[181, 19], [16, 184]],
      "num_samples": 400,
      "evaluated_at": "2026-01-07T10:45:00"
    },
    "improvement": {
      "accuracy": 0.1891,
      "precision": 0.189,
      "recall": 0.220,
      "f1_score": 0.204,
      "roc_auc": 0.190,
      "specificity": 0.159,
      "healthy_accuracy": 0.154,
      "leukemia_accuracy": 0.224
    }
  }
}
```

## 🔍 Monitoring Progress

### Server Logs

```bash
cd backend
uvicorn app.main:app --reload --log-level info
```

### Log Output Example:

```
======================================================================
FEDERATED LEARNING - COMPLETE PIPELINE
======================================================================
✓ Fetched 3 clients from database

[PIPELINE] Step 1: Pre-FL Evaluation
======================================================================
Device: cuda
Public Test Path: /path/to/public_test
Evaluating Hospital_A (resnet18)
  ✓ Loaded model from: /models/client_0.pt
  ✓ Loaded 400 test samples
  ✓ Accuracy: 72.34%, F1: 0.705
✓ Updated pre-FL metrics for client 0
...

[PIPELINE] Step 2: Preparing FL Configuration
======================================================================
✓ Wrote temporary config to: /tmp/fl_config.json

[PIPELINE] Step 3: Running Federated Learning Simulation
======================================================================
FL Directory: /path/to/federated_learning
Running 10 rounds with lr=0.001
[SERVER] Starting Federated Learning (10 rounds, lr=0.001)
[ROUND 1] Client 0 - Training Phase
...
✅ FL simulation completed successfully

[PIPELINE] Step 4: Post-FL Evaluation
======================================================================
...
✓ Updated post-FL metrics for client 0
  Improvement: Accuracy +18.91%, F1 +0.204

======================================================================
✅ FEDERATED LEARNING PIPELINE COMPLETE
======================================================================
Clients: 3
Rounds: 10
All metrics saved to Supabase database
======================================================================
```

## 🧪 Testing

### 1. Test with Minimal Configuration

```bash
curl -X POST "http://localhost:8000/api/start_fl_complete" \
  -H "Content-Type: application/json" \
  -d '{
    "num_rounds": 1,
    "lr": 0.001,
    "local_epochs": 1
  }'
```

### 2. Verify Database Updates

```sql
SELECT id, client_name, metrics->>'pre_fl' as pre_fl, metrics->>'post_fl' as post_fl
FROM clients;
```

### 3. Check Model Files

Ensure models are saved to paths specified in client configurations:

```bash
ls -lh /models/
# Should show updated timestamps after FL completes
```

## 🚨 Troubleshooting

### Issue: "No clients found in database"
**Solution**: Ensure clients are properly registered in Supabase `clients` table.

### Issue: "Public test data not found"
**Solution**:
1. Check `PUBLIC_TEST_PATH` environment variable
2. Ensure dataset exists at specified path
3. Dataset should be in ImageFolder format:
   ```
   public_test/
   ├── all/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── hem/
       ├── image3.jpg
       └── image4.jpg
   ```

### Issue: "FL simulation failed"
**Solution**:
1. Check `flwr` is installed: `pip install flwr`
2. Verify `pyproject.toml` exists in federated_learning/
3. Check FL simulation logs in server output

### Issue: Background task not running
**Solution**:
1. Ensure FastAPI background tasks are enabled
2. Check server logs for exceptions
3. Verify all dependencies are installed

## 📈 Production Deployment

### Recommended Setup:

1. **Server Requirements**:
   - 16GB RAM (minimum)
   - 4 vCPUs
   - GPU (optional but recommended)
   - 50GB storage

2. **Environment Variables**:
   ```bash
   # Production .env
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your_service_role_key
   PUBLIC_TEST_PATH=/var/data/flex-med/public_test
   ```

3. **Process Management** (systemd or Docker):
   ```bash
   # systemd service
   sudo systemctl start flex-med-backend
   sudo systemctl enable flex-med-backend
   ```

4. **Nginx Reverse Proxy**:
   ```nginx
   location /api {
       proxy_pass http://localhost:8000;
       proxy_read_timeout 3600;  # Long timeout for FL
   }
   ```

## 🔄 Migration from Colab

### Old Workflow (Colab + Ngrok):
1. Start Colab notebook
2. Expose ngrok URL
3. Update URLs in `federated.py`
4. Call `/start_fl` endpoint
5. Ngrok calls Colab orchestrator
6. Manual data.json management

### New Workflow (Integrated):
1. Call `/api/start_fl_complete`
2. Done! ✅

### Migration Steps:
1. ✅ Deploy backend to server with GPU
2. ✅ Set environment variables
3. ✅ Use `/start_fl_complete` endpoint
4. ❌ Remove Colab dependency
5. ❌ Remove ngrok URLs
6. ❌ Remove data.json manual sync

## 📝 Notes

- **Background Execution**: The endpoint returns immediately but FL runs asynchronously
- **No data.json Required**: Configuration is temporarily written and auto-cleaned
- **Database as Source of Truth**: All metrics stored in Supabase, no intermediate files
- **Production Ready**: Can be deployed directly to any server with Python + PyTorch

## 🔗 Related Files

- `backend/app/routes/federated.py` - Main endpoint implementation
- `backend/app/services/fl_evaluation_service.py` - Evaluation logic
- `federated_learning/flex_med/task_new.py` - FL runtime config support
- `federated_learning/flex_med/client_app_new.py` - Client training logic
- `federated_learning/flex_med/server_app_new.py` - Server aggregation logic
