# 🛍️ AI-Based Personalized Shopping Deals Recommender

## 🚀 Project Overview
This project is an AI-powered recommender system that provides **personalized shopping deals** based on customer behavior, purchase history, and customer segmentation. It leverages machine learning techniques like **K-Means clustering** and **Collaborative Filtering (SVD)** to offer relevant product suggestions to users.

---

## 📚 Dataset Details

The dataset includes:
- 🧾 User transaction data (Total Spend, Items Purchased, Membership Type, etc.)
- 📦 Product details (Item ID, Product Name, Category, etc.)
- 🌟 Ratings and Reviews provided by users

---

## 📌 Data Preprocessing
- Removed unnecessary columns  
- Handled missing values  
- Encoded categorical variables  
- Normalized numerical features for improved model performance  

---

## 🔍 Methodology

### ✅ PHASE 1: Data Understanding & Cleaning
- Loaded and explored the dataset
- Removed duplicates and handled missing values
- Conducted Exploratory Data Analysis (EDA) to reveal shopping behavior and trends

### ✅ PHASE 2: Building the Recommender System
- Applied **K-Means clustering** to group similar users
- Built a **Collaborative Filtering** model using **SVD**
- Trained and saved the model with `joblib`

### ✅ PHASE 3: Developing the Web Application
- Designed an interactive **Streamlit UI**
- Implemented backend logic to fetch segment-based recommendations
- Integrated with a **Flask API** for serving recommendation results

### ✅ PHASE 4: Deployment & Documentation
- Deployed the Streamlit app to **Streamlit Cloud**
- Wrote this comprehensive documentation (`README.md`)

---

## 📊 Model Evaluation

Performance metrics for the recommender system:

- **Average RMSE**: `0.2903`  
- **Average MAE**: `0.2234`

These metrics indicate that the model makes relatively accurate recommendations.

---

## ❤️ User Feedback Mechanism

A **like/dislike** system has been introduced to allow users to provide feedback on recommended products. This lays the foundation for future improvements, such as **real-time learning** from user interactions.

**Current Feedback State Example:**
```json
[
  {
    "user_id": "102",
    "item_id": "Item_275_12",
    "type": "like"
  },
  {
    "user_id": "102",
    "item_id": "Item_352_7",
    "type": "dislike"
  }
]
## 🖥 How to Run Locally

### 🔧 Prerequisites
- Python 3.x

### 📦 Install Dependencies
```bash
pip install -r requirements.txt

## ▶️ Run the App
git clone https://github.com/keerthi152004/shopping-deals-recommender.git
cd shopping-deals-recommender
streamlit run streamlit_app.py
The app will open in your browser at: http://localhost:8501
🚀 Live Deployment
🔗 Click Here to Access the Deployed App

🛠 Technologies Used
Python: Pandas, NumPy, Scikit-learn, Surprise, Joblib

Machine Learning: K-Means Clustering, SVD Collaborative Filtering

Streamlit: Frontend UI

Flask: API layer for recommendations

Streamlit Cloud: App Deployment Platform

🤝 Contributing
Want to improve this project?
Fork the repo, make your changes, and submit a pull request! Contributions are welcome.

💎 Contact
📧 Email: keerthiannamareddy@gmail.com

👤 Author: Keerthi A.

