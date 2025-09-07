
# 🏠 Housing Price Tier Prediction Pipeline

An end-to-end ML pipeline using ZenML, MLflow, and Streamlit to predict housing price tiers and visualize them on a map.

## 🔧 Tech Stack

- ZenML for orchestration
- MLflow for experiment tracking
- Streamlit for dashboard
- Docker for containerization

## 🚀 Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
zenml init
python src/pipeline.py
streamlit run dashboard/app.py
```





house_pricing/
├── dashboard/             # Streamlit dashboard
│   └── Dockerfile         # Dockerfile for dashboard container
├── data/                  # Data storage
│   ├── processed/         # Cleaned datasets
│   └── raw/               # Raw scraped or downloaded data
├── mlruns/                # MLflow experiment tracking
├── models/                # Saved model artifacts
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── Dockerfile         # Dockerfile for pipeline container
│   ├── etl/               # Data transformation scripts
│   ├── ingestion/         # Scraping or API ingestion
│   ├── modeling/          # Model training and evaluation
│   └── visualization/     # Mapping and plotting scripts
