import pandas as pd

def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    df.columns = df.columns.str.lower().str.strip()

    # Clean and convert price column
    df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)
    # Encode 'condition' column
    condition_map = {
    "poor": 1,
    "fair": 2,
    "average": 3,
    "good": 4,
    "very good": 5,
    "excellent": 6
}
    df["condition"] = df["condition"].str.lower().map(condition_map)

    # Drop rows with missing coordinates or price
    df = df.dropna(subset=["price", "lat", "long"])

    # Create price tiers
    df["price_category"] = pd.qcut(df["price"], q=3, labels=["Low", "Medium", "High"])

    # Select relevant columns
    df = df[[
        "price", "lat", "long", "bedrooms", "bathrooms",
        "sqft_living", "yr_built", "price_category"
    ]]
    df.columns = df.columns.str.strip().str.lower()

    return df


