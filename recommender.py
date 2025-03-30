import pandas as pd

class CollaborativeRecommender:
    def __init__(self, model):
        self.model = model

    def get_recommendations(self, user_id, num_recommendations=5):
        # Ensure 'df' is defined before calling this function
        file_path = "C:/Users/keert/OneDrive/Desktop/Shopping_Deals_Recommeneder/dataset/user_item_ratings.csv"
        df = pd.read_csv(file_path)

        all_items = df['item_id'].unique()
        predictions = [(item, self.model.predict(user_id, item).est) for item in all_items]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in predictions[:num_recommendations]]
