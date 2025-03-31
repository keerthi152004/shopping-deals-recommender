import joblib
import pandas as pd
from flask import Flask, request, jsonify
from train_collaborative import CollaborativeRecommender  # Import the class

# Initialize Flask app
app = Flask(__name__)

# Load trained models using joblib
try:
    kmeans_model = joblib.load("kmeans_model.pkl")  # Ensure this file exists in the main folder
    collab_model = joblib.load("collaborative_model.pkl")  # Load collaborative model

    # Ensure the loaded model is an instance of CollaborativeRecommender
    if not isinstance(collab_model, CollaborativeRecommender):
        raise TypeError("collaborative_model.pkl is not an instance of CollaborativeRecommender!")

except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    exit(1)  # Exit if models are missing
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Home Route to prevent "Not Found" errors
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the AI-Based Shopping Deals Recommender API!"})

# Define API route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate input
        if not data or "features" not in data or "customer_id" not in data:
            return jsonify({"error": "Missing 'features' or 'customer_id' in request"}), 400

        # Convert features into DataFrame
        customer_features = pd.DataFrame([data["features"]])

        # Predict customer segment using K-Means
        segment = kmeans_model.predict(customer_features)[0]

        # Get recommendations from collaborative filtering model
        recommended_deals = collab_model.get_recommendations(data["customer_id"])

        return jsonify({
            "customer_segment": int(segment),
            "recommended_deals": recommended_deals
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
