# src/steps/data_loader.py
from zenml import step
import pandas as pd

@step
def load_data() -> pd.DataFrame:
    # Load or simulate data
    return pd.DataFrame()
