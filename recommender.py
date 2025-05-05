# recommender.py
from surprise import SVD, Dataset, Reader
import pandas as pd

class CollaborativeRecommender:
    def __init__(self, model=None):
        """
        Initialize the recommender system with an optional model (e.g., SVD).
        """
        self.model = model if model else SVD()
        self.trainset = None

    def fit(self, data):
        """
        Fit the collaborative filtering model using Surprise's Dataset.
        """
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def get_recommendations(self, user_id, top_n=5):
        """
        Generate top-N product recommendations for a given user ID.
        Returns a list of raw item IDs (product IDs).
        """
        if self.trainset is None:
            print("Error: Trainset is not available.")
            return []

        user_id_str = str(user_id)
        print(f"Raw User ID in get_recommendations: {user_id_str}") # Debugging print

        # Check if user exists in training data
        if user_id_str not in self.trainset._raw2inner_id_users:
            print(f"User {user_id_str} not found in training set.")
            return []

        try:
            print(f"Raw User ID before to_inner_uid: {user_id_str}") # Debugging print
            user_inner_id = self.trainset.to_inner_uid(user_id_str)
        except Exception as e:
            print(f"Error converting user ID to inner ID: {e}")
            return []

        # Check if user has rated anything
        if not self.trainset.ur[user_inner_id]:
            print(f"User {user_id_str} has no ratings in training set.")
            return []

        # Get all items and rated items for the user
        all_items = set(self.trainset.all_items())
        rated_items = set(j for (j, _) in self.trainset.ur[user_inner_id])
        unrated_items = all_items - rated_items

        predictions = []
        for iid in unrated_items:
            try:
                raw_iid = self.trainset.to_raw_iid(iid)
                pred = self.model.predict(user_id_str, raw_iid)
                predictions.append((raw_iid, pred.est))
            except Exception as e:
                print(f"Prediction error for item {iid}: {e}")

        if not predictions:
            print("No predictions could be made.")
            return []

        # Sort by predicted rating and return top N item IDs
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = [item_id for item_id, _ in predictions[:top_n]]

        return top_items