from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# ── Load models ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "models", "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "models", "encoder.pkl"), "rb") as f:
    encoders = pickle.load(f)

with open(os.path.join(BASE, "models", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# ── Column order MUST match training ─────────────────────────────────────────
FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_OPTIONS = {
    "gender":           ["Female", "Male"],
    "Partner":          ["Yes", "No"],
    "Dependents":       ["No", "Yes"],
    "PhoneService":     ["No", "Yes"],
    "MultipleLines":    ["No phone service", "No", "Yes"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["No", "Yes", "No internet service"],
    "OnlineBackup":     ["Yes", "No", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport":      ["No", "Yes", "No internet service"],
    "StreamingTV":      ["No", "Yes", "No internet service"],
    "StreamingMovies":  ["No", "Yes", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod":    ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)", "Credit card (automatic)"],
}


@app.route("/")
def index():
    return render_template("index.html", options=CATEGORICAL_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # ── Build raw feature dict ────────────────────────────────────────
        data = {}
        for col in FEATURE_COLS:
            if col == "SeniorCitizen":
                data[col] = int(form.get(col, 0))
            elif col in NUMERICAL_COLS:
                data[col] = float(form.get(col, 0))
            else:
                data[col] = form.get(col, "")

        # ── Encode categoricals ───────────────────────────────────────────
        for col, enc in encoders.items():
            if col in data:
                data[col] = enc.transform([data[col]])[0]

        # ── Build DataFrame in correct column order ───────────────────────
        row = pd.DataFrame([[data[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

        # ── Scale numerical columns ───────────────────────────────────────
        row[NUMERICAL_COLS] = scaler.transform(row[NUMERICAL_COLS])

        # ── Predict ───────────────────────────────────────────────────────
        pred  = model.predict(row)[0]
        prob  = model.predict_proba(row)[0][1]

        result = {
            "churn":       "Yes" if pred == 1 else "No",
            "probability": round(float(prob) * 100, 2),
            "risk":        "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low"),
        }
        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("\n🚀  Customer Churn Predictor is running!")
    print("   Open  http://127.0.0.1:5000  in your browser\n")
    app.run(debug=True)
