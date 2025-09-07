from zenml import step
import pandas as pd
from sklearn.preprocessing import StandardScaler

@step
def preprocess_data(df: pd.DataFrame) -> tuple:
    # Example preprocessing logic
    df = df.dropna()
    df["price_category"] = pd.qcut(df["price"], q=3, labels=["Low", "Medium", "High"])
    X = df.drop(columns=["price", "price_category"])
    y = df["price_category"]
    return X, y
