import streamlit as st
import pickle
import pandas as pd

st.title("✈️ Airline Recommendation System")

# Load data & models
df = pd.read_csv("data/reviews.csv")

with open("model/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

with open("model/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

# Traveller type input
traveller_type = st.selectbox("Select Traveller Type:", df["Type of Traveller"].unique())
top_n = st.slider("Number of Recommendations:", 1, 10, 5)

# Generate recommendations (simple avg-based for demo)
recommendations = (
    df.groupby("Airline")["Overall Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(top_n)
)

st.subheader(f"Top {top_n} Recommended Airlines for {traveller_type}")
st.table(recommendations)
