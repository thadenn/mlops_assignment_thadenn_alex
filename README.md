# MLOps Assignment – Streamlit + PyCaret + Hydra + DVC

Production-style demo with two models: **Used Car Price** and **Wheat Seeds**. Deps via Poetry. Reproducible pipeline via DVC. Local MLflow file store. CI smoke test with GitHub Actions.

## Live URLs
- **Web app:** `https://thadenn-mlops-assignment-thadenn-alex-srcapp-mbmctn.streamlit.app/car_price_prediction`
- **GitHub repo:** `https://github.com/thadenn/mlops_assignment_thadenn_alex`

## Deployment guide

### A) Streamlit Community Cloud
1. Fork/import this repo to your GitHub.
2. Streamlit Cloud → New app → pick repo/branch.
3. **Main file:** `src/app.py`
4. **Python:** 3.10.18 (Advanced settings).
5. Deploy. Streamlit reads `pyproject.toml` + `poetry.lock`.

### B) Local from GitHub
```bash
pip install poetry
git clone <REPO_URL>
cd <repo-folder>
poetry install
# optional: reproduce pipeline (writes models/data)
# Windows: set MLFLOW_TRACKING_URI=file:mlruns
# macOS/Linux: export MLFLOW_TRACKING_URI=file:mlruns
poetry run dvc repro
poetry run streamlit run src/app.py
```
MLflow UI (optional):
```bash
poetry run mlflow ui
# open http://127.0.0.1:5000
```

### C) Local from a zip
Unzip → open terminal in folder → follow section **B**.

## User guide

### Launch
- Local: open the URL printed.
- Cloud: open the deployed app URL.

### Pages
- **🚗 Used Car Price Predictor (India)**  
  Uses `models/catboost_used_car_model.pkl`.  
  - Single prediction via form.  
  - Batch: upload CSV with same columns used in training.

- **🌾 Wheat Seeds Classifier**  
  Uses `models/wheat_seeds_pipeline.pkl` and:
  - `src/config/pycaret_setup_config.json`
  - `data/wheatseeds/example_request.json` (optional)
  - `data/wheatseeds/wheat_seeds_batch_examples.csv` (optional)
  - Single prediction via form.  
  - Batch: upload CSV with same columns used in training.

### Required runtime files
```
models/catboost_used_car_model.pkl
models/wheat_seeds_pipeline.pkl
src/config/pycaret_setup_config.json
data/wheatseeds/example_request.json (optional)
data/wheatseeds/wheat_seeds_batch_examples.csv (optional)
```
If you ran `poetry run dvc repro`, these are generated.

## Application folder structure
```
.
├─ src/
│  ├─ app.py
│  ├─ preprocessing_car.py
│  ├─ modelling_car.py
│  ├─ modelling_wheat.py
│  └─ pages/
│     ├─ car_price_prediction.py
│     └─ wheat_seeds.py
├─ src/config/
│  └─ pycaret_setup_config.json
│  └─ config.yaml
│  └─ config_wheat.yaml
├─ data/
│  ├─ raw/
│  │  └─ 02_Used_Car_Prices.xlsx
│  ├─ processed/
│  │  └─ clean_used_car_prices.csv
│  └─ wheatseeds/
│     ├─ 03_Wheat_Seeds.csv
│     ├─ example_request.json
│     └─ wheat_seeds_batch_examples.csv
├─ models/
│  ├─ catboost_used_car_model.pkl
│  └─ wheat_seeds_pipeline.pkl
├─ dvc.yaml
├─ dvc.lock
├─ pyproject.toml
├─ poetry.lock
└─ .github/workflows/ci.yml
```

## DVC & MLflow notes
- **DVC**: `dvc repro` runs stages and regenerates artifacts; raw datasets can be tracked via `*.dvc` files.
- **MLflow**: file store by default (`file:mlruns`). The app does not require MLflow UI to run.

## Branching workflow (implemented)

We use **GitHub Flow**: short-lived feature branches, PRs into `main`, CI must pass before merge.  
This README was updated via a branch and PR.

**Example (for this README update)**
```bash
git checkout -b docs/update-readme
git add README.md
git commit -m "docs: update README with deploy links and branching note"
git push -u origin docs/update-readme
# Open a Pull Request on GitHub → wait for CI → Merge → delete branch
```

## Team members
| Name | Dataset | Email | GitHub |
|------|------|-------|--------|
| Thadenn Thien | 03_Wheat_Seeds.csv | thadennthn@gmail.com | @thadenn |
| Alex Kersten | 02_Used_Car_Prices.xlsx | alex.yishun@gmail.com | @alex-billybob |


## Support
- Python: 3.10.18
- OS: Windows / Linux / macOS
