import streamlit as st
import pandas as pd
import numpy as np

st.title("✈️ Advanced Airline Recommendation System")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/reviews.csv")

# --- Add synthetic features for demo ---
budget_map = {
    "Economy": "Cheap",
    "Premium Economy": "Mid",
    "Business": "Luxury",
    "First": "Luxury"
}
df["Budget"] = df["Class"].map(budget_map).fillna("Cheap")

alliance_map = {
    "Qatar Airways": "Oneworld",
    "Lufthansa": "Star Alliance",
    "Emirates": "None",
    "British Airways": "Oneworld",
    "Delta Air Lines": "SkyTeam",
    "United Airlines": "Star Alliance",
}
df["Alliance"] = df["Airline"].map(alliance_map).fillna("None")

df["Flight Duration"] = np.where(
    df["Route"].str.contains("London|Paris|Rome"), "Short-haul",
    np.where(df["Route"].str.contains("New York|Dubai"), "Long-haul", "Medium-haul")
)

df["Amenities"] = df["Reviews"].apply(
    lambda x: ["WiFi", "Entertainment"] if "wifi" in str(x).lower() else ["Entertainment"]
)

# ===============================
# User Inputs
# ===============================
st.subheader("🔍 Enter Your Preferences")

preferences = st.text_area(
    "Describe your preferences (e.g., 'comfortable airline with good food and entertainment')"
)

# Route dropdown instead of text input
available_routes = df["Route"].dropna().unique()
route = st.selectbox("Select Route:", available_routes)

travel_class = st.selectbox("Select Class:", df["Class"].dropna().unique())

traveller_type = st.selectbox("Select Traveller Type:", df["Type of Traveller"].dropna().unique())

budget = st.selectbox("Select Budget Range:", ["Any", "Cheap", "Mid", "Luxury"])

alliance = st.selectbox("Select Airline Alliance:", ["Any", "Star Alliance", "Oneworld", "SkyTeam", "None"])

duration = st.selectbox("Select Flight Duration:", ["Any", "Short-haul", "Medium-haul", "Long-haul"])

amenities = st.multiselect("Select Preferred Amenities:", ["WiFi", "Extra Legroom", "Lounge Access", "Entertainment"])

top_n = st.number_input("Number of Recommendations:", min_value=1, max_value=10, value=5)

# ===============================
# Recommendation Logic
# ===============================
filtered_df = df.copy()

# Apply filters
if budget != "Any":
    filtered_df = filtered_df[filtered_df["Budget"] == budget]

if alliance != "Any":
    filtered_df = filtered_df[filtered_df["Alliance"] == alliance]

if duration != "Any":
    filtered_df = filtered_df[filtered_df["Flight Duration"] == duration]

if amenities:
    filtered_df = filtered_df[filtered_df["Amenities"].apply(lambda x: all(a in x for a in amenities))]

if travel_class:
    filtered_df = filtered_df[filtered_df["Class"] == travel_class]

if traveller_type:
    filtered_df = filtered_df[filtered_df["Type of Traveller"] == traveller_type]

if route:
    filtered_df = filtered_df[filtered_df["Route"] == route]

# Aggregate airline ratings
recommendations = (
    filtered_df.groupby("Airline")["Overall Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(top_n)
)

# ===============================
# Display Results
# ===============================
if st.button("Get Recommendations"):
    if recommendations.empty:
        st.warning("⚠️ No airlines found matching your preferences.")
    else:
        recommended_df = filtered_df[filtered_df["Airline"].isin(recommendations.index)][
            ["Airline", "Route", "Class", "Amenities", "Reviews", "Overall Rating"]
        ].drop_duplicates(subset=["Airline"])

        st.subheader("✈️ Recommended Airlines")
        st.dataframe(recommended_df.reset_index(drop=True))
