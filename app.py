import streamlit as st
import pandas as pd
import numpy as np
import re

st.title("✈️ Advanced Airline Recommendation System")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/reviews.csv")

# ===============================
# Route Cleaning Function
# ===============================
def clean_route(route):
    if pd.isna(route):
        return "Unknown"
    route = str(route).strip()

    # Normalize "to" and fix typos like "nto"
    route = re.sub(r"\s*to\s*", " to ", route, flags=re.IGNORECASE)
    route = re.sub(r"nto", " to ", route, flags=re.IGNORECASE)

    # Remove "via ..." part
    route = re.split(r"\s+via\s+", route, flags=re.IGNORECASE)[0]

    # Add spacing if missing (like 'toCPT' -> 'to CPT')
    route = re.sub(r"to([A-Z]{2,3})", r"to \1", route)

    return route.title().strip()

df["Route_Clean"] = df["Route"].apply(clean_route)

# ===============================
# Extract Departure & Destination
# ===============================
split_routes = df["Route_Clean"].str.split(" to ", n=1, expand=True)
df["Departure"] = split_routes[0].fillna("Unknown").str.strip()
df["Destination"] = split_routes[1].fillna("Unknown").str.strip()

# Warn if malformed routes remain
bad_routes = df[df["Destination"] == "Unknown"]["Route"].unique()
if len(bad_routes) > 0:
    st.warning(f"⚠️ Some routes could not be parsed: {bad_routes[:5]} ...")

# ===============================
# Clean Class Names
# ===============================
df["Class"] = df["Class"].astype(str).str.strip().str.title()
class_map = {
    "Economyy": "Economy",
    "Eco": "Economy",
    "Busines": "Business"
}
df["Class"] = df["Class"].replace(class_map)

# ===============================
# Add Synthetic Features
# ===============================
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
    df["Route_Clean"].str.contains("London|Paris|Rome", na=False), "Short-haul",
    np.where(df["Route_Clean"].str.contains("New York|Dubai", na=False), "Long-haul", "Medium-haul")
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

# --- Departure & Destination ---
departure_country = st.selectbox(
    "Select Departure:",
    sorted(df["Departure"].dropna().unique())
)

available_destinations = (
    df[df["Departure"] == departure_country]["Destination"]
    .dropna()
    .unique()
)
destination_country = st.selectbox(
    "Select Destination:",
    sorted(available_destinations)
)

# --- Other Filters ---
travel_class = st.selectbox("Select Class:", sorted(df["Class"].dropna().unique()))

traveller_type = st.selectbox("Select Traveller Type:", sorted(df["Type of Traveller"].dropna().unique()))

budget = st.selectbox("Select Budget Range:", ["Any", "Cheap", "Mid", "Luxury"])

alliance = st.selectbox("Select Airline Alliance:", ["Any", "Star Alliance", "Oneworld", "SkyTeam", "None"])

duration = st.selectbox("Select Flight Duration:", ["Any", "Short-haul", "Medium-haul", "Long-haul"])

amenities = st.multiselect("Select Preferred Amenities:", ["WiFi", "Extra Legroom", "Lounge Access", "Entertainment"])

top_n = st.number_input("Number of Recommendations:", min_value=1, max_value=10, value=5)

# ===============================
# Recommendation Logic
# ===============================
filtered_df = df.copy()

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

if departure_country:
    filtered_df = filtered_df[filtered_df["Departure"] == departure_country]

if destination_country:
    filtered_df = filtered_df[filtered_df["Destination"] == destination_country]

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
            ["Airline", "Departure", "Destination", "Class", "Amenities", "Reviews", "Overall Rating"]
        ].drop_duplicates(subset=["Airline"])

        st.subheader("✈️ Recommended Airlines")
        st.dataframe(recommended_df.reset_index(drop=True))
