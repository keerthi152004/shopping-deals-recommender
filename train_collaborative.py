# train_collaborative.py
import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from recommender import CollaborativeRecommender
import json  # For saving evaluation metrics

if __name__ == "__main__":
    # Load and prepare data
    ratings_df = pd.read_csv("user_item_ratings.csv")
    # Convert user_id and item_id to strings
    ratings_df["user_id"] = ratings_df["user_id"].astype(str)
    ratings_df["item_id"] = ratings_df["item_id"].astype(str)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)

    # Train model
    model = SVD()
    recommender = CollaborativeRecommender(model)
    recommender.fit(data)

    # Debugging - Trainset after fitting (in training script):
    print("\nDebugging - Trainset after fitting (in training script):")
    print(recommender.trainset)
    if recommender.trainset:
        print("Debugging - Trainset's raw to inner user ID mapping (in training script):")
        print(recommender.trainset._raw2inner_id_users)

    # Evaluate model performance using cross-validation
    print("\nEvaluating model performance using cross-validation...")
    cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

    avg_rmse = cv_results["test_rmse"].mean()
    avg_mae = cv_results["test_mae"].mean()

    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    # Save evaluation metrics
    metrics = {"rmse": avg_rmse, "mae": avg_mae}
    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("\nEvaluation metrics saved to evaluation_metrics.json")

    # Save model
    joblib.dump(recommender, "collaborative_model_updated.pkl", compress=3)