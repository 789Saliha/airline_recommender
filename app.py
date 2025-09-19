import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("✈️ Airline Recommendation System")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("C:\Users\zeeshan\Downloads\airline_recommender\data")

# Check required columns
required_cols = ["Airline", "Reviews", "Route", "Class"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Dataset must contain column: {col}")
        st.stop()

# ===============================
# Build TF-IDF model on reviews
# ===============================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["Reviews"].fillna(""))

# ===============================
# UI Inputs
# ===============================
# User preference text
user_query = st.text_area(
    "Describe your preferences:",
    placeholder="e.g., I want a comfortable airline with good food and entertainment",
)

# Route filter
routes = ["All"] + sorted(df["Route"].dropna().unique().tolist())
route = st.selectbox("Select Route:", routes)

# Class filter
classes = ["All"] + sorted(df["Class"].dropna().unique().tolist())
cabin_class = st.selectbox("Select Class:", classes)

# Number of recommendations
top_n = st.slider("Number of Recommendations:", 1, 10, 5)

# ===============================
# Recommendation Logic
# ===============================
if st.button("Get Recommendations"):
    if not user_query.strip():
        st.warning("Please enter your preferences.")
    else:
        # Filter dataset
        filtered_df = df.copy()
        if route != "All":
            filtered_df = filtered_df[filtered_df["Route"] == route]
        if cabin_class != "All":
            filtered_df = filtered_df[filtered_df["Class"] == cabin_class]

        if filtered_df.empty:
            st.error("No reviews found for the selected filters.")
        else:
            # Compute similarity
            query_vec = vectorizer.transform([user_query])
            review_vecs = vectorizer.transform(filtered_df["Reviews"].fillna(""))
            sims = cosine_similarity(query_vec, review_vecs).flatten()

            filtered_df = filtered_df.assign(Similarity=sims)

            # Aggregate by airline
            airline_scores = (
                filtered_df.groupby("Airline")["Similarity"]
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )

            # Show results
            st.subheader(f"Top {top_n} Recommended Airlines for Your Preferences")
            st.table(airline_scores)
