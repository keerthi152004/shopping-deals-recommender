import joblib
import pandas as pd
import re  # For regex extraction
from surprise import SVD, Dataset, Reader

# Define file paths
import os

# Get the base directory dynamically
base_path = os.path.dirname(os.path.abspath(__file__))

# Use relative paths
file_path = os.path.join(base_path, "dataset", "user_item_ratings.csv")
model_path = os.path.join(base_path, "collaborative_model.pkl")


try:
    # Load dataset
    df = pd.read_csv(file_path)
    print(" Dataset loaded successfully.")

    # Validate necessary columns
    required_columns = {'user_id', 'item_id', 'rating'}
    if not required_columns.issubset(df.columns):
        raise KeyError(f"Dataset must contain {required_columns}, but found {df.columns}")

    # Convert user_id to integer
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
    
    # ✅ Extract numeric part from 'item_id' and convert to integer
    df['item_id'] = df['item_id'].apply(lambda x: int(re.search(r'\d+', str(x)).group()) if pd.notna(x) else None)
    
    # Handle missing values after conversions
    df.dropna(subset=['user_id', 'item_id', 'rating'], inplace=True)
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    
    # Define rating scale dynamically
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

    # Train the SVD model
    print(" Training collaborative filtering model...")
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    print(" Model training complete.")

    # Define recommender class
    class CollaborativeRecommender:
        def __init__(self, model):
            self.model = model

        def get_recommendations(self, user_id, num_recommendations=5):
            all_items = df['item_id'].unique()
            predictions = [(item, self.model.predict(user_id, item).est) for item in all_items]
            predictions.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in predictions[:num_recommendations]]

    # Save the trained model
    recommender = CollaborativeRecommender(model)
    joblib.dump(recommender, model_path)
    print(f"Model saved successfully at {model_path}")

except Exception as e:
    print(f"Error: {e}")
