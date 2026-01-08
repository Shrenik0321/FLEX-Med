# 🔍 FL Pipeline Monitoring Guide

## How to Track FL Execution Progress

When you call `/api/start_fl_complete`, the FL pipeline runs in the background. Here's how to monitor it:

## 📊 Status Endpoint

### GET `/api/fl_status`

Returns real-time status of the FL pipeline execution.

### Example Usage

```bash
# Start FL pipeline
curl -X POST "http://localhost:8000/api/start_fl_complete" \
  -H "Content-Type: application/json" \
  -d '{"num_rounds": 5}'

# Check status
curl http://localhost:8000/api/fl_status
```

### Response Format

```json
{
  "is_running": true,
  "current_stage": "fl_simulation",
  "progress": 45,
  "total_rounds": 5,
  "current_round": 2,
  "clients": 3,
  "started_at": "2026-01-08T10:30:00",
  "completed_at": null,
  "error": null,
  "logs": [
    "[10:30:00] FL pipeline started with 3 clients, 5 rounds",
    "[10:30:15] Pre-FL evaluation complete, metrics saved to database",
    "[10:30:20] Configuration prepared: /tmp/fl_config.json",
    "[10:30:25] Starting FL simulation (5 rounds)",
    "[10:35:10] FL simulation completed successfully"
  ]
}
```

## 📝 Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `is_running` | boolean | Whether FL is currently executing |
| `current_stage` | string | Current pipeline stage (see stages below) |
| `progress` | int | Overall progress percentage (0-100) |
| `total_rounds` | int | Total number of FL rounds |
| `current_round` | int | Current round during simulation (0 when not in simulation) |
| `clients` | int | Number of clients participating |
| `started_at` | string | ISO timestamp when pipeline started |
| `completed_at` | string | ISO timestamp when pipeline completed (null if running) |
| `error` | string | Error message if pipeline failed (null if no error) |
| `logs` | array | Recent log entries (last 50 messages) |

## 🔄 Pipeline Stages

The `current_stage` field progresses through these values:

1. **`initializing`** (0-10%) - Setting up pipeline
2. **`pre_fl_evaluation`** (10-25%) - Evaluating models before training
3. **`preparing_config`** (25-35%) - Creating configuration files
4. **`fl_simulation`** (35-70%) - Running federated learning rounds
5. **`post_fl_evaluation`** (70-95%) - Evaluating trained models
6. **`cleanup`** (95-98%) - Removing temporary files
7. **`completed`** (100%) - Pipeline finished successfully

## 🖥️ Monitoring Methods

### Method 1: Manual Polling (Simple)

```bash
# Check status every 10 seconds
watch -n 10 "curl -s http://localhost:8000/api/fl_status | jq '.current_stage, .progress, .logs[-3:]'"
```

### Method 2: Python Script (Automated)

```python
import requests
import time

def monitor_fl_pipeline(api_url="http://localhost:8000"):
    """Monitor FL pipeline until completion"""

    print("Monitoring FL pipeline...")
    print("-" * 60)

    while True:
        try:
            response = requests.get(f"{api_url}/api/fl_status")
            status = response.json()

            # Print current status
            print(f"\r[{status['progress']:3d}%] {status['current_stage']:20s}", end='')

            # Check if completed
            if not status['is_running']:
                print(f"\n\n{'='*60}")
                if status['error']:
                    print(f"❌ FAILED: {status['error']}")
                else:
                    print(f"✅ COMPLETED")
                print(f"{'='*60}")

                # Print recent logs
                print("\nRecent logs:")
                for log in status['logs'][-10:]:
                    print(f"  {log}")
                break

            time.sleep(5)  # Poll every 5 seconds

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

# Usage
monitor_fl_pipeline()
```

### Method 3: Browser (Real-time Dashboard)

Create a simple HTML file:

```html
<!DOCTYPE html>
<html>
<head>
    <title>FL Pipeline Monitor</title>
    <style>
        body { font-family: monospace; padding: 20px; background: #1e1e1e; color: #fff; }
        .status { margin: 20px 0; }
        .progress-bar { width: 100%; height: 30px; background: #333; border-radius: 5px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00ff88, #00aa44); transition: width 0.3s; }
        .logs { background: #2d2d2d; padding: 15px; border-radius: 5px; max-height: 400px; overflow-y: auto; }
        .log-entry { margin: 5px 0; }
        .error { color: #ff6b6b; }
        .success { color: #51cf66; }
    </style>
</head>
<body>
    <h1>🔍 FL Pipeline Monitor</h1>
    <div class="status">
        <p>Status: <span id="stage">-</span></p>
        <p>Progress: <span id="progress">0</span>%</p>
        <div class="progress-bar">
            <div class="progress-fill" id="progress-bar" style="width: 0%"></div>
        </div>
    </div>

    <h3>Recent Logs:</h3>
    <div class="logs" id="logs"></div>

    <script>
        const API_URL = 'http://localhost:8000';

        async function updateStatus() {
            try {
                const response = await fetch(`${API_URL}/api/fl_status`);
                const status = await response.json();

                // Update progress
                document.getElementById('stage').textContent = status.current_stage || 'idle';
                document.getElementById('progress').textContent = status.progress;
                document.getElementById('progress-bar').style.width = `${status.progress}%`;

                // Update logs
                const logsDiv = document.getElementById('logs');
                logsDiv.innerHTML = status.logs
                    .slice(-20)
                    .map(log => `<div class="log-entry">${log}</div>`)
                    .join('');

                // Auto-scroll to bottom
                logsDiv.scrollTop = logsDiv.scrollHeight;

                // Stop polling if completed
                if (!status.is_running) {
                    if (status.error) {
                        logsDiv.innerHTML += `<div class="log-entry error">❌ Error: ${status.error}</div>`;
                    } else {
                        logsDiv.innerHTML += '<div class="log-entry success">✅ Pipeline completed!</div>';
                    }
                    clearInterval(interval);
                }
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }

        // Poll every 3 seconds
        const interval = setInterval(updateStatus, 3000);
        updateStatus(); // Initial call
    </script>
</body>
</html>
```

Save as `fl_monitor.html` and open in browser!

## 📱 Quick Status Check Commands

### Check if FL is running
```bash
curl -s http://localhost:8000/api/fl_status | jq '.is_running'
```

### Get current progress
```bash
curl -s http://localhost:8000/api/fl_status | jq '.progress'
```

### Get current stage
```bash
curl -s http://localhost:8000/api/fl_status | jq '.current_stage'
```

### Get latest logs
```bash
curl -s http://localhost:8000/api/fl_status | jq '.logs[-5:]'
```

### Check for errors
```bash
curl -s http://localhost:8000/api/fl_status | jq '.error'
```

## 🚨 Error Handling

### If FL pipeline fails:

```bash
# Check error message
curl -s http://localhost:8000/api/fl_status | jq '.error, .logs[-10:]'
```

Common errors and solutions:

| Error | Solution |
|-------|----------|
| "FL pipeline is already running" | Wait for current job to finish or restart backend |
| "No clients found in database" | Ensure clients are registered in Supabase |
| "Public test data not found" | Check `PUBLIC_TEST_PATH` environment variable |
| "FL simulation failed" | Check `flwr` installation and FL code logs |

## 📊 Example: Complete Monitoring Session

```bash
# Terminal 1: Start FL pipeline
curl -X POST "http://localhost:8000/api/start_fl_complete" \
  -d '{"num_rounds": 5}'

# Response:
{
  "status": "started",
  "status_endpoint": "/api/fl_status",
  "note": "Use GET /api/fl_status to monitor progress in real-time"
}

# Terminal 2: Monitor progress
watch -n 5 "curl -s http://localhost:8000/api/fl_status | jq '{stage: .current_stage, progress: .progress, latest_log: .logs[-1]}'"

# Output (updates every 5 seconds):
{
  "stage": "pre_fl_evaluation",
  "progress": 15,
  "latest_log": "[10:30:12] Evaluating Hospital_A (resnet18)"
}

# ... wait for completion ...

{
  "stage": "completed",
  "progress": 100,
  "latest_log": "[10:45:00] FL pipeline completed successfully! 3 clients trained for 5 rounds"
}
```

## 🔧 Server Logs

For detailed debugging, also check server logs:

```bash
# Backend logs
cd backend
uvicorn app.main:app --reload --log-level info

# Watch logs in real-time
tail -f /path/to/backend.log
```

## 💡 Tips

1. **Start with 1 round** for testing: `{"num_rounds": 1}`
2. **Use watch command** for live updates without refreshing
3. **Monitor both** `/api/fl_status` endpoint AND server logs
4. **Check database** after completion to verify metrics were saved
5. **Don't start another FL job** while one is running (409 error protection)

## 🎯 Summary

**To monitor FL pipeline execution:**

1. ✅ Call `/api/start_fl_complete` to start
2. ✅ Poll `/api/fl_status` to check progress
3. ✅ Watch `logs` array for detailed updates
4. ✅ Check `progress` for percentage complete
5. ✅ Look for `error` field if something fails
6. ✅ When `is_running: false`, pipeline is done

**Simple one-liner:**
```bash
curl -X POST http://localhost:8000/api/start_fl_complete -d '{"num_rounds":5}' && \
watch -n 5 'curl -s http://localhost:8000/api/fl_status | jq .'
```
