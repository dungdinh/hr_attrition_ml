# HR Attrition ML Production Project

Production-ready project for predicting employee attrition probability with:
- 5-model benchmark
- cross validation
- hyperparameter tuning
- ROC curve
- SHAP explainability
- feature importance visualization
- auto model selection
- model export to `.pkl`
- FastAPI inference service
- Streamlit dashboard for HR risk scoring
- Docker and GitHub Actions CI/CD

## 1. Project structure

```bash
hr_attrition_ml_prod/
├─ app/
├─ artifacts/
├─ data/
├─ plots/
├─ src/
├─ tests/
├─ .github/workflows/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

## 2. Local setup

Create and activate a virtual environment:

### Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train the model

Run training from the project root:

```bash
python -m src.train --data data/hr_attrition.csv
```

Outputs are saved to:
- `artifacts/best_model.pkl`
- `artifacts/metrics.json`
- `artifacts/model_benchmark.json`
- `artifacts/tuned_results.json`
- `plots/model_comparison.png`
- `plots/roc_curve.png`
- `plots/feature_importance.png`
- `plots/shap_summary.png`

## 4. Run the API

Start FastAPI:

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints:
- `/health`
- `/model-info`
- `/predict`
- `/docs`

## 5. Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

## 6. Docker

Build the image:

```bash
docker build -t hr-attrition-ml .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 hr-attrition-ml
```

With Docker Compose:

```bash
docker compose up --build
```

## 7. GitHub CI/CD

This repository includes:
- `.github/workflows/ci.yml` for validation on push / pull request
- `.github/workflows/cd.yml` for building and publishing a container image to GHCR on version tags

### CI
The CI workflow runs:
- dependency install
- import / syntax checks
- smoke tests

### CD
The CD workflow builds and pushes a Docker image to GitHub Container Registry when you push a tag like:

```bash
git tag v1.0.0
git push origin v1.0.0
```

## 8. Data notes

The sample dataset is expected at:

```bash
data/hr_attrition.csv
```

Leakage / identifier columns are removed automatically when present:
- `Attrition Reason`
- `AttritionDate`
- `IsVoluntary`
- `Employee Name`
- `Manager Name`
- `Join Date`
- employee ID columns

## 9. Feature set

The project compares these five models:
- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- Hist Gradient Boosting

It then selects the best model using ROC-AUC and exports the winning pipeline.
# hr_attrition_ml
