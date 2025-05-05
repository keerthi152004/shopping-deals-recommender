# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import os
import sys
import json

# Set up paths
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

from recommender import CollaborativeRecommender

# Load model
model_path = os.path.join(base_path, "collaborative_model_updated.pkl")
try:
    recommender = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load collaborative filtering model: {e}")
    st.stop()

# Load product names
product_mapping_path = os.path.join(base_path, "products.csv")
try:
    product_mapping = pd.read_csv(product_mapping_path)
    product_dict = product_mapping.set_index("item_id")["product_name"].to_dict()
except Exception as e:
    st.warning(f"Could not load product names. Showing item IDs.\nError: {e}")
    product_dict = {}

# Load ratings data
ratings_data_path = os.path.join(base_path, "user_item_ratings.csv")
try:
    ratings_df = pd.read_csv(ratings_data_path)
    ratings_df['numeric_item_id'] = ratings_df['item_id'].apply(lambda x: int(x.split('_')[1]))
except Exception as e:
    st.warning(f"Could not load ratings data. Fallback to showing top-rated products.\nError: {e}")
    ratings_df = pd.DataFrame(columns=["user_id", "item_id", "rating"])

# Load evaluation metrics (optional)
metrics_path = os.path.join(base_path, "evaluation_metrics.json")
try:
    with open(metrics_path, "r") as f:
        evaluation_metrics = json.load(f)
        st.sidebar.subheader("Model Evaluation")
        st.sidebar.write(f"Average RMSE: {evaluation_metrics['rmse']:.4f}")
        st.sidebar.write(f"Average MAE: {evaluation_metrics['mae']:.4f}")
except FileNotFoundError:
    st.sidebar.warning("Evaluation metrics not found. Run training script first.")
except Exception as e:
    st.sidebar.error(f"Error loading evaluation metrics: {e}")

# Initialize session state for feedback
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = []

def record_feedback(user_id, item_id, feedback_type):
    """Records user feedback (e.g., 'like' or 'dislike') in the session state."""
    st.session_state['feedback'].append({'user_id': str(user_id), 'item_id': item_id, 'type': feedback_type})
    if feedback_type == 'like':
        st.toast(f"You liked {product_dict.get(int(item_id.split('_')[1]), item_id)}!")
    elif feedback_type == 'dislike':
        st.toast(f"You disliked {product_dict.get(int(item_id.split('_')[1]), item_id)}!")

# Streamlit UI
st.title("Personalized Shopping Deals Recommender")
st.markdown("Enter your **User ID** to get personalized shopping recommendations.")

user_id = st.number_input("Enter your User ID", min_value=1, step=1)
num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

# This part is now outside the 'Get Recommendations' button
if str(user_id) in recommender.trainset._raw2inner_id_users:
    recommendations = []
    if st.session_state.get('user_recommendations_' + str(user_id)) is not None:
        recommendations = st.session_state['user_recommendations_' + str(user_id)]

    if recommendations:
        st.subheader("Recommended Deals:")
        for item_id in recommendations:
            try:
                base_item_id = int(item_id.split('_')[1])
                item_name = product_dict.get(base_item_id, f"Item {item_id}")
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"- {item_name}")
                with col2:
                    if st.button("Like", key=f"like_{item_id}"):
                        record_feedback(user_id, item_id, 'like')
                with col3:
                    if st.button("Dislike", key=f"dislike_{item_id}"):
                        record_feedback(user_id, item_id, 'dislike')
            except:
                st.write(f"- Item ID: {item_id} (Product name not found)")

if st.button("Get Recommendations"):
    try:
        # Check if user exists in the model
        if str(user_id) in recommender.trainset._raw2inner_id_users:
            recommendations = recommender.get_recommendations(str(user_id), top_n=num_recommendations)
            st.session_state['user_recommendations_' + str(user_id)] = recommendations
            if not recommendations:
                st.warning(
                    "No recommendations available for this user. Showing top-rated products instead.")
                top_products = (
                    ratings_df.groupby("numeric_item_id")["rating"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                    .index.tolist()
                )
                st.subheader("Top Products:")
                for item_id in top_products:
                    item_name = product_dict.get(item_id, f"Item {item_id}")
                    st.write(f"- {item_name}")
            else:
                st.subheader("Recommended Deals:") #moved
                for item_id in recommendations: #moved
                    try:
                        base_item_id = int(item_id.split('_')[1])
                        item_name = product_dict.get(base_item_id, f"Item {item_id}")
                        col1, col2, col3 = st.columns([3, 1, 1]) #moved
                        with col1: #moved
                            st.write(f"- {item_name}") #moved
                        with col2: #moved
                            if st.button("Like", key=f"like_{item_id}"): #moved
                                record_feedback(user_id, item_id, 'like') #moved
                        with col3: #moved
                            if st.button("Dislike", key=f"dislike_{item_id}"): #moved
                                record_feedback(user_id, item_id, 'dislike') #moved
                    except: #moved
                        st.write(f"- Item ID: {item_id} (Product name not found)") #moved
        else:
            st.warning(
                f"User ID {user_id} does NOT exist in the model's internal mapping.")
            st.write("Showing top-rated products instead.")
            top_products = (
                ratings_df.groupby("numeric_item_id")["rating"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .index.tolist()
            )
            st.subheader("Top Products:")
            for item_id in top_products:
                item_name = product_dict.get(item_id, f"Item {item_id}")
                st.write(f"- {item_name}")

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

# Display collected feedback
st.subheader("Current Feedback State (for demonstration):")
st.write(st.session_state['feedback'])
