import joblib
from surprise import SVD, Dataset, Reader
import pandas as pd

# Load your dataset
file_path = "C:/Users/keert/OneDrive/Desktop/Shopping_Deals_Recommeneder/dataset/user_item_ratings.csv"
df = pd.read_csv(file_path)  # Ensure this file contains user-item interactions

# Define reader
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Train the model
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# Define a wrapper class with the `get_recommendations` method
class CollaborativeRecommender:
    def __init__(self, model):
        self.model = model

    def get_recommendations(self, user_id, num_recommendations=5):
        all_items = df['item_id'].unique()
        predictions = [(item, self.model.predict(user_id, item).est) for item in all_items]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in predictions[:num_recommendations]]

# Save the properly formatted model
recommender = CollaborativeRecommender(model)
joblib.dump(recommender, "collaborative_model.pkl")
