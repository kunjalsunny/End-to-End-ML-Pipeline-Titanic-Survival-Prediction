## End to End data science repository ##

## MLflow Tracking (DagsHub)

This project uses **MLflow with DagsHub** as a remote tracking server.

### 1. Create a DagsHub Access Token
- Go to DagsHub → Settings → Tokens
- Generate a new token(If not created while signup)

### 2. Export environment variables

#### Linux / macOS
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/kunjalsunny/Data-Science-End-to-End.mlflow
export MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<your_access_token>
