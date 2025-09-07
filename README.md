

## Tech Stack

# 🏠 Housing Price Tier Prediction Pipeline

An end-to-end ML pipeline using  **ZenML** ,  **MLflow** , and **Streamlit** to predict housing price tiers and visualize them on an interactive map.

## 🔧 Tech Stack

* 🧪 **ZenML** – Orchestration and pipeline management
* 📊 **MLflow** – Experiment tracking and model registry
* 📈 **Streamlit** – Interactive dashboard for predictions and visualizations
* 🐳 **Docker** – Containerization for reproducible environments

## 🚀 Quickstart

```bash
house_pricing/
├── dashboard/             # Streamlit dashboard
│   └── Dockerfile         # Dockerfile for dashboard container
├── data/    
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
🌐 GitHub Access
The dataset is versioned and available in the GitHub repository under data/raw/housing.csv. This ensures reproducibility and transparency for users and collaborators.

📜 License
This project is licensed under the MIT License. See LICENSE for details.
🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

# Initialize ZenML
zenml init

# Run the pipeline
python src/pipeline.py

# Launch the dashboard
streamlit run dashboard/app.py
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
zenml init
python src/pipeline.py
streamlit run dashboard/app.py

data/raw/housing.csv

```
