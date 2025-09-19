import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("✈️ Airline Recommendation System")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/reviews.csv")



# === User Inputs ===
preferences = st.text_area("✍️ Describe your preferences (e.g., 'I want a comfortable airline with good food and entertainment')")

route = st.text_input("🛫 Enter Route (e.g., London–New York)")
travel_class = st.selectbox("💺 Select Class:", df["Class"].unique())

top_n = st.text_input("🔢 How many recommendations do you want? (Enter a number)", "5")

# Validate input
try:
    top_n = int(top_n)
except ValueError:
    st.error("⚠️ Please enter a valid number for recommendations")
    st.stop()

# === Simple Recommendation Demo ===
recommendations = (
    df.groupby("Airline")["Overall Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(top_n)
)

st.subheader(f"Top {top_n} Airlines for your preferences:")
st.table(recommendations)
