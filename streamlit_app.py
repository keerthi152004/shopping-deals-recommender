import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sys
import os

# Ensure the script directory is in Python's path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import CollaborativeRecommender
from train_collaborative import CollaborativeRecommender

# Load trained models
base_path = os.path.dirname(os.path.abspath(__file__))
kmeans_model = joblib.load(os.path.join(base_path, "kmeans_model.pkl"))
collaborative_model = joblib.load(os.path.join(base_path, "collaborative_model.pkl"))

product_mapping = pd.read_csv(os.path.join(base_path, "products.csv")).set_index("item_id")["product_name"].to_dict()

# Streamlit UI
st.title("AI-Based Personalized Shopping Deals Recommender")

# User inputs
st.sidebar.header("User Information")
total_spend = st.sidebar.number_input("Total Spend", min_value=0, value=500)
items_purchased = st.sidebar.number_input("Items Purchased", min_value=0, value=5)
membership_type = st.sidebar.selectbox("Membership Type", ["Basic", "Silver", "Gold"])
age = st.sidebar.number_input("Age", min_value=18, value=25)
days_since_last_purchase = st.sidebar.number_input("Days Since Last Purchase", min_value=0, value=10)

# Convert membership type to numerical
membership_dict = {"Basic": 0, "Silver": 1, "Gold": 2}
membership_type = membership_dict[membership_type]

# Predict Customer Segment
if st.button("Predict Customer Segment"):
    features = np.array([[total_spend, items_purchased, membership_type, age, days_since_last_purchase]])
    segment = kmeans_model.predict(features)[0]
    st.success(f"Customer Segment: {segment}")

# Get Personalized Recommendations
user_id = st.number_input("Enter User ID for Recommendations", min_value=1, value=102)

if st.button("Get Recommendations"):
    recommended_items = collaborative_model.get_recommendations(user_id)
    recommended_products = [product_mapping.get(item_id, f"Item {item_id}") for item_id in recommended_items]
    
    st.write("### Recommended Deals:")
    for product in recommended_products:
        st.write(f"- {product}")
