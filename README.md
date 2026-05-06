📡 Customer Churn Predictor — Flask App
Customer Churn Prediction System
📌 Project Overview
This project is a Machine Learning-based Customer Churn Prediction System. It predicts whether a customer is likely to leave a service based on various features.
The model helps businesses improve customer retention by identifying at-risk customers.
---
🚀 Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib / Seaborn
FastAPI
Jupyter Notebook
---
📊 Features
Data preprocessing and cleaning
Exploratory Data Analysis (EDA)
Machine Learning model training
Model evaluation
Churn prediction
(Optional) API deployment using FastAPI
---
📂 Project Structure
```
churn_app/
├── app.py                  ← Flask application (run this)
├── requirements.txt        ← Python dependencies
├── models/
│   ├── best_model.pkl      ← Trained Random Forest model
│   ├── encoder.pkl         ← Label encoders
│   └── scaler.pkl          ← StandardScaler
└── templates/
    └── index.html          ← Web UI
```
Setup & Run
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Run the app
```bash
python app.py
```
3. Open in browser
Visit: http://127.0.0.1:5000
---

📈 Output
The model predicts:
Customer likely to churn
Customer likely to stay
---
🎯 Future Improvements
Improve model accuracy
Add frontend UI
Deploy on cloud (AWS / Azure)
Real-time prediction system
---
👨‍💻 Author
Sanket Tippe
Fill in the customer details and click Predict Churn to get the result.
