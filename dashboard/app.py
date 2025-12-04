# 📦 Imports
import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_theme(style="darkgrid")

# ⚙️ Page Config
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="wide")
st.title("🏘️ Housing Price Tier Prediction")

# 📂 Load Default Dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/house.csv")

data = load_data()

# 📥 File Upload
uploaded_file = st.file_uploader("Upload housing data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    if df["price"].dtype == "object":
        df["price"] = df["price"].str.replace('[\$,]', '', regex=True).str.strip().astype(float)

    if "price_category" not in df.columns:
        df["price_category"] = pd.qcut(df["price"], q=3, labels=["Low", "Medium", "High"])

    zip_avg_price = df.groupby("zipcode")["price"].transform("mean")
    df["zipcode_bin"] = pd.qcut(zip_avg_price, q=3, labels=["Low", "Medium", "High"])
    df["zipcode_bin"] = df["zipcode_bin"].map({"Low": 1, "Medium": 2, "High": 3})

    df["condition"] = df["condition"].astype(str).str.lower().map({
        "poor": 1, "fair": 2, "average": 3, "good": 4, "very good": 5, "excellent": 6
    }).fillna(3)

    df["grade"] = df["grade"].astype(str).str.lower().map({
        "low": 1, "average": 2, "good": 3, "excellent": 4
    })

    df = pd.get_dummies(df, columns=["zipcode"], drop_first=True)

    final_features = [col for col in df.columns if col not in ["price", "price_category", "id"]]
    X = df[final_features]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🤖 Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 📈 Full Prediction Results
    st.subheader("📈 Full Prediction Results")
    prediction_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": y_pred,
        "Error": y_test - y_pred
    })
    st.dataframe(prediction_df.style.format({
        "Actual Price": "${:,.0f}",
        "Predicted Price": "${:,.0f}",
        "Error": "${:,.0f}"
    }))
    st.download_button("Download Predictions", prediction_df.to_csv(index=False), "predicted_prices.csv", "text/csv")

    fig, ax = plt.subplots()
    sns.histplot(prediction_df["Error"], bins=30, kde=True, ax=ax, color="purple")
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error ($)")
    st.pyplot(fig)

    # 🧠 Model Details
    st.write("✅ Model trained. Ready for predictions.")
    st.subheader("🧠 Model Details")
    st.write("**Model Used:** Random Forest Regressor")
    st.json(model.get_params())

    # 📊 Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.subheader("📊 Model Evaluation")
    st.metric("Mean Absolute Error (MAE)", f"${mae:,.0f}")
    st.metric("R² Score", f"{r2:.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x=y_test, y=y_pred, ax=ax, scatter_kws={"s": 50}, line_kws={"color": "red"})
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Housing Prices")
    st.pyplot(fig)

    # 🔍 Feature Importance
    importances = model.feature_importances_
    feature_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    top_features = feature_df.sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top_features, palette="rocket", ax=ax)
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)

    # 📘 Data Dictionary
    data_dict = {
        "price": "Sale price of the house",
        "sqft_living": "Interior living space in square feet",
        "sqft_lot": "Lot size in square feet",
        "bedrooms": "Number of bedrooms",
        "bathrooms": "Number of bathrooms",
        "floors": "Number of floors",
        "waterfront": "Waterfront property (1 = yes, 0 = no)",
        "condition": "Overall condition rating",
        "grade": "Construction and design grade",
        "sqft_above": "Living area above ground",
        "sqft_basement": "Basement area in square feet",
        "yr_built": "Year the house was built",
        "yr_renovated": "Year the house was renovated",
        "zipcode": "Postal code",
        "lat": "Latitude",
        "long": "Longitude",
        "sqft_living15": "Living space of nearby 15 houses",
        "sqft_lot15": "Lot size of nearby 15 houses"
    }
    st.subheader("📘 Data Dictionary")
    st.table(pd.DataFrame(data_dict.items(), columns=["Feature", "Description"]))

    # 🗺️ Map Visualization
    st.subheader("🗺️ Price Tier Map")
    map = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start=12)
    for _, row in df.iterrows():
        tier = str(row["price_category"]).lower()
        color = "green" if tier == "low" else "orange" if tier == "medium" else "red"
        folium.CircleMarker(
            location=[row["lat"], row["long"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"${row['price']:,.0f} USD ({tier.capitalize()} Tier)"
        ).add_to(map)
    folium_static(map)

    # 🏡 Predict Housing Price
    st.subheader("🏡 Predict Housing Price")
    default_lat = df["lat"].mean() if "lat" in df.columns else 47.0
    default_long = df["long"].mean() if "long" in df.columns else 8.0

    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1.0, 5.0, 2.0, step=0.25)
    sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 1500)
    floors = st.slider("Floors", 1, 3, 1)
    grade = st.slider("Grade (1–13)", 1, 13, 7)
    condition = st.slider("Condition (1–6)", 1, 6, 3)
    yr_built = st.number_input("Year Built", 1900, 2025, 2000)
    lat = st.number_input("Latitude", value=default_lat)
    long = st.number_input("Longitude", value=default_long)
    zip_range = st.selectbox("Zipcode Tier", ["Low-priced areas", "Medium-priced areas", "High-priced areas"])
    zipcode_bin = {"Low-priced areas": 1, "Medium-priced areas": 2, "High-priced areas": 3}[zip_range]

    if st.button("Predict Price"):
        input_data = pd.DataFrame(columns=final_features)
        input_data.loc[0] = 0
        input_data.at[0, "bedrooms"] = bedrooms
        input_data.at[0, "bathrooms"] = bathrooms
        input_data.at[0, "sqft_living"] = sqft_living
        input_data.at[0, "floors"] = floors
        input_data.at[0, "grade"] = grade
        input_data.at[0, "condition"] = condition
        input_data.at[0, "yr_built"] = yr_built
        input_data.at[0, "lat"] = lat
        input_data.at[0, "long"] = long
        input_data.at[0, "zipcode_bin"] = zipcode_bin

        predicted_price = model.predict(input_data)[0]
        st.success(f"🏷️ Estimated Price: ${predicted_price:,.0f} USD")
        st.caption(f"Model MAE: ±${mae:,.0f}")




 
 
# 📂 Load Default Dataset
import io

try:
    # Try loading from local file first
    df = pd.read_csv("housing_prices.csv")
    st.success("Dataset loaded successfully from local file.")
except FileNotFoundError:
    # Fallback: fetch from GitHub
    url = "https://raw.githubusercontent.com/Frank-Ellingsen/datafrank.github.io/main/datasets/housing_prices.csv"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text))
        st.success("Dataset fetched from GitHub.")
        
        # Offer download button
        st.download_button(
            label="👈 Download sample CSV to begin",
            data=response.content,
            file_name="housing_prices.csv",
            mime="text/csv"
        )
    else:
        st.error("Failed to fetch dataset from GitHub.")
