# 📡 Customer Churn Predictor — Flask App

## Project Structure

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

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
Visit: **http://127.0.0.1:5000**

---

Fill in the customer details and click **Predict Churn** to get the result.
