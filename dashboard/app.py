import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


@st.cache_data
def load_data():
    return pd.read_csv("data/housing.csv")

data = load_data()

st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏘️ Housing Price Tier Prediction")

uploaded_file = st.file_uploader("Upload housing data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Clean price column
    if df["price"].dtype == "object":
        df["price"] = df["price"].str.replace('[\$,]', '', regex=True).str.strip().astype(float)

    # Generate price_category if missing
    if "price_category" not in df.columns:
        df["price_category"] = pd.qcut(df["price"], q=3, labels=["Low", "Medium", "High"])

    # Zipcode binning
    zip_avg_price = df.groupby("zipcode")["price"].transform("mean")
    df["zipcode_bin"] = pd.qcut(zip_avg_price, q=3, labels=["Low", "Medium", "High"])
    zipcode_map = {"Low": 1, "Medium": 2, "High": 3}
    df["zipcode_bin"] = df["zipcode_bin"].map(zipcode_map)

    zip_bin_ranges = df.groupby("zipcode_bin")["zipcode"].apply(lambda x: f"{x.min()}–{x.max()}")
    # st.write("📍 Zipcode Ranges by Tier:")
    # st.write(zip_bin_ranges)

    # Map condition and grade
    condition_map = {"poor": 1, "fair": 2, "average": 3, "good": 4, "very good": 5, "excellent": 6}
    grade_map = {"low": 1, "average": 2, "good": 3, "excellent": 4}
    df["condition"] = df["condition"].astype(str).str.lower().map(condition_map).fillna(3)
    df["grade"] = df["grade"].astype(str).str.lower().map(grade_map)

    df = pd.get_dummies(df, columns=["zipcode"], drop_first=True)

    features = ["bedrooms", "bathrooms", "sqft_living", "floors", "grade", "condition", "yr_built", "lat", "long", "zipcode_bin"]
    final_features = [col for col in df.columns if col not in ["price", "price_category", "id"]]

    X = df[final_features]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("✅ Model trained. Ready for predictions.")      

    st.subheader("🧠 Model Details")
    st.write("**Model Used:** Random Forest Regressor")
    st.write("**Parameters:**")
    st.json(model.get_params())



    # ✅ Shared evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.subheader("📊 Model Evaluation")
    st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:,.0f}")
    st.metric(label="R² Score", value=f"{r2:.2f}")
    
   

   # Plot with regression line
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x=y_test, y=y_pred, ax=ax, scatter_kws={"s": 50}, line_kws={"color": "red"})
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Housing Prices with Regression Line")
    st.pyplot(fig)

    
    # 🧾 Data Dictionary
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

    
    
   # Feature importance
    # Feature importance
    importances = model.feature_importances_
    features = X.columns
    feature_df = pd.DataFrame({"Feature": features, "Importance": importances})

    # Sort by importance (descending) and select top 10
    top_features = feature_df.sort_values(by="Importance", ascending=False).head(10)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top_features, palette="rocket", ax=ax)
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)


     # Display in Streamlit
    st.subheader("📘 Data Dictionary")
    st.table(pd.DataFrame(data_dict.items(), columns=["Feature", "Description"]))



    # Map visualization
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
    
    

    st.subheader("🏡 Predict Housing Price")

    # Defensive defaults for lat/long
    if "lat" in df.columns and pd.api.types.is_numeric_dtype(df["lat"]):
        default_lat = df["lat"].mean()
    else:
        default_lat = 47.0  # fallback for Norway

    if "long" in df.columns and pd.api.types.is_numeric_dtype(df["long"]):
        default_long = df["long"].mean()
    else:
        default_long = 8.0  # fallback for Norway

    # Input fields
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1.0, 5.0, 2.0, step=0.25)
    sqft_living = st.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=1500)
    floors = st.slider("Floors", 1, 3, 1)
    grade_labels = {
    1: "Basic",
    4: "Fair",
    7: "Average",
    10: "Good",
    13: "Luxury"
    }
    grade = st.slider("Grade (1–13)", 1, 13, 7, help="1=Basic, 13=Luxury")
    st.caption(f"Selected Grade: {grade} – {grade_labels.get(grade, 'Custom')}")

    condition_labels = {
    1: "Poor",
    2: "Fair",
    3: "Average",
    4: "Good",
    5: "Very Good",
    6: "Excellent"
    }
    condition = st.slider("Condition (1–6)", 1, 6, 3, help="1=Poor, 6=Excellent")
    st.caption(f"Selected Condition: {condition} – {condition_labels.get(condition, 'Custom')}")


    
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
    lat = st.number_input("Latitude", value=default_lat)
    long = st.number_input("Longitude", value=default_long)
    zip_range = st.selectbox("Zipcode Tier", ["Low-priced areas", "Medium-priced areas", "High-priced areas"])
    zip_map = {
    "Low-priced areas": 1,
    "Medium-priced areas": 2,
    "High-priced areas": 3
    }
    zipcode_bin = zip_map[zip_range]
    st.caption(f"Selected Zipcode Tier: {zip_range}")


    if st.button("Predict Price"):
        input_data = pd.DataFrame(columns=final_features)
        input_data.loc[0] = 0  # initialize with zeros

        # Fill in values from user input
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




 
 
    else:
        st.info("👈 Upload a CSV file to begin.")
