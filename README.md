# Data-Science-End-to-End (Titanic)

End-to-end ML project scaffold with:
- modular Python package structure (`src/`)
- experiment tracking with **MLflow** (configured for **DagsHub** remote tracking)
- dataset/model artifact tracking with **DVC**
- a simple app entrypoint (`application.py`)
- containerization via `Dockerfile`
- deployment-friendly config (`.ebextensions/`)

---

## Project structure
Directory structure:
└── kunjalsunny-data-science-end-to-end/
    ├── README.md
    ├── application.py
    ├── Dockerfile
    ├── requirements.txt
    ├── setup.py
    ├── template.py
    ├── .dockerignore
    ├── .dvcignore
    ├── artifacts/
    │   ├── preprocessor.pkl
    │   ├── raw_data.csv.dvc
    │   └── test_data.csv
    ├── catboost_info/
    │   ├── catboost_training.json
    │   ├── learn_error.tsv
    │   ├── time_left.tsv
    │   └── learn/
    │       └── events.out.tfevents
    ├── mlruns/
    │   └── 0/
    │       ├── meta.yaml
    │       ├── 376168c49249443fbf265de1d3d8d058/
    │       │   ├── meta.yaml
    │       │   ├── metrics/
    │       │   │   └── accuracy
    │       │   ├── outputs/
    │       │   │   └── m-23b56877e0154f58a9a1ab0898ea902b/
    │       │   │       └── meta.yaml
    │       │   ├── params/
    │       │   │   ├── learning_rate
    │       │   │   ├── n_estimators
    │       │   │   └── subsample
    │       │   └── tags/
    │       │       ├── mlflow.runName
    │       │       ├── mlflow.source.git.commit
    │       │       ├── mlflow.source.name
    │       │       └── mlflow.source.type
    │       ├── db55ab5856db48609b817ae5cc79d8c9/
    │       │   ├── meta.yaml
    │       │   ├── metrics/
    │       │   │   └── accuracy
    │       │   ├── outputs/
    │       │   │   └── m-31953c5766a54e039010b84af67b8827/
    │       │   │       └── meta.yaml
    │       │   ├── params/
    │       │   │   └── n_estimators
    │       │   └── tags/
    │       │       ├── mlflow.runName
    │       │       ├── mlflow.source.git.commit
    │       │       ├── mlflow.source.name
    │       │       └── mlflow.source.type
    │       ├── f09a530a024944509f365b3221b7628f/
    │       │   ├── meta.yaml
    │       │   ├── metrics/
    │       │   │   └── accuracy
    │       │   ├── outputs/
    │       │   │   └── m-14f0864496ec4c45a65ca6b0943d70e8/
    │       │   │       └── meta.yaml
    │       │   ├── params/
    │       │   │   ├── learning_rate
    │       │   │   ├── n_estimators
    │       │   │   └── subsample
    │       │   └── tags/
    │       │       ├── mlflow.runName
    │       │       ├── mlflow.source.git.commit
    │       │       ├── mlflow.source.name
    │       │       └── mlflow.source.type
    │       └── models/
    │           ├── m-14f0864496ec4c45a65ca6b0943d70e8/
    │           │   ├── meta.yaml
    │           │   ├── artifacts/
    │           │   │   ├── conda.yaml
    │           │   │   ├── MLmodel
    │           │   │   ├── python_env.yaml
    │           │   │   └── requirements.txt
    │           │   ├── metrics/
    │           │   │   └── accuracy
    │           │   ├── params/
    │           │   │   ├── learning_rate
    │           │   │   ├── n_estimators
    │           │   │   └── subsample
    │           │   └── tags/
    │           │       ├── mlflow.source.git.commit
    │           │       ├── mlflow.source.name
    │           │       └── mlflow.source.type
    │           ├── m-23b56877e0154f58a9a1ab0898ea902b/
    │           │   ├── meta.yaml
    │           │   ├── artifacts/
    │           │   │   ├── conda.yaml
    │           │   │   ├── MLmodel
    │           │   │   ├── python_env.yaml
    │           │   │   └── requirements.txt
    │           │   ├── metrics/
    │           │   │   └── accuracy
    │           │   ├── params/
    │           │   │   ├── learning_rate
    │           │   │   ├── n_estimators
    │           │   │   └── subsample
    │           │   └── tags/
    │           │       ├── mlflow.source.git.commit
    │           │       ├── mlflow.source.name
    │           │       └── mlflow.source.type
    │           └── m-31953c5766a54e039010b84af67b8827/
    │               ├── meta.yaml
    │               ├── artifacts/
    │               │   ├── conda.yaml
    │               │   ├── MLmodel
    │               │   ├── python_env.yaml
    │               │   └── requirements.txt
    │               ├── metrics/
    │               │   └── accuracy
    │               ├── params/
    │               │   └── n_estimators
    │               └── tags/
    │                   ├── mlflow.source.git.commit
    │                   ├── mlflow.source.name
    │                   └── mlflow.source.type
    ├── src/
    │   ├── __init__.py
    │   └── datascience/
    │       ├── __init__.py
    │       ├── exception.py
    │       ├── logger.py
    │       ├── utils.py
    │       ├── components/
    │       │   ├── __init__.py
    │       │   ├── data_ingestion.py
    │       │   ├── data_transformation.py
    │       │   ├── model_monitoring.py
    │       │   └── model_trainer.py
    │       └── pipelines/
    │           ├── __init__.py
    │           ├── prediction_pipeline.py
    │           └── training_pipeline.py
    ├── templates/
    │   ├── home.html
    │   └── index.html
    ├── .dvc/
    │   └── config
    ├── .ebextensions/
    │   └── python.config
    └── .github/
        └── workflows/
            ├── cd.yml
            └── ci.yml


## Quickstart (Copy–Paste)

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