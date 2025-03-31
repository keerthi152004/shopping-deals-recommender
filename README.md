# AI-Based Personalized Shopping Deals Recommender

## 🚀 Project Overview
This project is an **AI-based shopping deals recommender system** that suggests personalized deals to customers based on their past purchases, spending behavior, and customer segmentation. The system leverages **machine learning** techniques such as **K-Means clustering** for customer segmentation and **Collaborative Filtering (SVD)** for recommendation generation.

## 📚 Dataset Details
The dataset used in this project contains:
- **User transaction data** (Total Spend, Items Purchased, Membership Type, etc.)
- **Product details** (Item ID, Product Name, Category, etc.)
- **Ratings and Reviews** provided by users

### 📌 Data Preprocessing
- Removed unnecessary columns
- Handled missing values
- Converted categorical variables into numerical form
- Normalized numerical features for better model performance

## 🔍 Methodology

### **PHASE 1: Data Understanding & Cleaning**
✔ Loaded and inspected the dataset  
✔ Cleaned missing values and removed duplicates  
✔ Conducted **Exploratory Data Analysis (EDA)** to identify shopping patterns  

### **PHASE 2: Building the Recommender System**
✔ Used **K-Means clustering** to segment customers  
✔ Built a **Collaborative Filtering model (SVD)** to generate personalized recommendations  
✔ Trained the model and saved it using **joblib**  

### **PHASE 3: Developing the Web Application**
✔ Created a **Streamlit-based frontend** for user interaction  
✔ Built backend logic to fetch customer segment & recommendations  
✔ Integrated **Flask API** to serve recommendations  

### **PHASE 4: Deployment & Documentation**
✔ Deployed the **Streamlit app on Streamlit Cloud**  
✔ Created this **README file for documentation**  

## 🖥 How to Run Locally
### **Prerequisites**
Make sure you have the following installed:
- Python 3.x
- Required Python libraries (install using the command below)

```bash
pip install -r requirements.txt
```

### **Run the Application**
1. Clone the repository:
```bash
git clone https://github.com/keerthi152004/shopping-deals-recommender.git
cd shopping-deals-recommender
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501/`.

## 🚀 Live Deployment
You can access the deployed app here:  
🔗 **[Live Streamlit App](https://shopping-deals-recommender-cw2murdu6lcenvrwzlm9dr.streamlit.app/)**  

## 🛠 Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, Surprise, Joblib)
- **Machine Learning** (K-Means Clustering, SVD Collaborative Filtering)
- **Streamlit** (For building the UI)
- **Flask** (For backend API)
- **Streamlit Cloud** (For deployment)

## 🤝 Contributing
Want to improve this project? Feel free to fork the repo, make changes, and submit a pull request!

## 💎 Contact
For any questions or feedback, reach out at **creativeexplorer15@gmail.com**

---
**Keerthi A.**

