# End-to-End ML Pipeline: Titanic Survival Prediction

End-to-end ML project scaffold with:
- modular Python package structure (`src/`)
- experiment tracking with **MLflow** (configured for **DagsHub** remote tracking)
- dataset/model artifact tracking with **DVC**
- a simple app entrypoint (`application.py`)
- containerization via `Dockerfile`
- deployment-friendly config (`.ebextensions/`)

---

```text
kunjalsunny-data-science-end-to-end/
├─ README.md
├─ application.py
├─ requirements.txt
├─ setup.py
├─ Dockerfile
├─ template.py
├─ templates/
│  ├─ index.html
│  └─ home.html
├─ src/
│  └─ datascience/
│     ├─ exception.py
│     ├─ logger.py
│     ├─ utils.py
│     ├─ components/
│     │  ├─ data_ingestion.py
│     │  ├─ data_transformation.py
│     │  ├─ model_trainer.py
│     │  └─ model_monitoring.py
│     └─ pipelines/
│        ├─ training_pipeline.py
│        └─ prediction_pipeline.py
├─ artifacts/               # generated outputs (tracked via DVC if needed)
│  ├─ preprocessor.pkl
│  └─ *.dvc
├─ .github/workflows/
│  ├─ ci.yml
│  └─ cd.yml
├─ .dvc/
│  └─ config
├─ .ebextensions/
│  └─ python.config
└─ (auto-generated)
   ├─ mlruns/               # MLflow local store (do not commit)
   └─ catboost_info/        # CatBoost logs (do not commit)


```
## Quickstart

### 1. Clone
```bash
git clone https://github.com/kunjalsunny/Data-Science-End-to-End.git
cd Data-Science-End-to-End
```

### Windows
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

MLflow Tracking (DagsHub)

Set these environment variables to log runs to DagsHub MLflow.

```bash
$env:MLFLOW_TRACKING_URI="https://dagshub.com/kunjalsunny/Data-Science-End-to-End.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
$env:MLFLOW_TRACKING_PASSWORD="<your_dagshub_access_token>"

```

# Data-Science-End-to-End (Titanic)

End-to-end ML project scaffold with:
- modular Python package structure (src/)
- experiment tracking with MLflow (DagsHub remote tracking)
- dataset/model artifact tracking with DVC
- app entrypoint: application.py
- containerization via Dockerfile
- deployment config via .ebextensions/

## Quickstart

### Clone
```bash
git clone https://github.com/kunjalsunny/Data-Science-End-to-End.git
cd Data-Science-End-to-End


Setup
Windows (PowerShell)

python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .


macOS / Linux
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .



MLflow Tracking (DagsHub)
Windows (PowerShell)
$env:MLFLOW_TRACKING_URI="https://dagshub.com/kunjalsunny/Data-Science-End-to-End.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
$env:MLFLOW_TRACKING_PASSWORD="<your_dagshub_access_token>"


macOS / Linux
export MLFLOW_TRACKING_URI="https://dagshub.com/kunjalsunny/Data-Science-End-to-End.mlflow"
export MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
export MLFLOW_TRACKING_PASSWORD="<your_dagshub_access_token>"


Run
Run app
python application.py
Run Flask

Windows (PowerShell)
$env:FLASK_APP="application.py"
flask run

macOS / Linux
export FLASK_APP=application.py
flask run


Open:
http://127.0.0.1:5000

Run notebooks
jupyter notebook

DVC
Init
dvc init
Track artifacts (example)
dvc add artifacts/
git add artifacts.dvc .gitignore
git commit -m "Track artifacts with DVC"
Add remote + push (example: S3)
dvc remote add -d storage s3://<your-bucket>/<path>
dvc remote modify storage region <your-region>
dvc push
Pull on new machine
dvc pull

Docker
Build
docker build -t ds-end-to-end .
Run
docker run -p 5000:5000 ds-end-to-end

Open:
http://127.0.0.1:5000