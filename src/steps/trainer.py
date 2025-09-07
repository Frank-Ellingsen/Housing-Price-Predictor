from zenml import step
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

@step
def train_model(df: pd.DataFrame):
    X = df[["bedrooms", "bathrooms", "sqft_living", "floors", "grade", "condition", "yr_built", "lat", "long"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test
