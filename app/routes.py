from flask import render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from app import app

# Load model and scaler
try:
    model = joblib.load("data/xgb_model.pkl")
    scaler = joblib.load("data/scaler.pkl")
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# Default values for V1-V28 (set to 0 as PCA components)
DEFAULT_VALUES = {f"V{i}": 0.0 for i in range(1, 29)}


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        if request.is_json:
            data = request.get_json()
            amount = float(data.get("amount", 0))
            time = float(data.get("time", 0))
        else:
            amount = float(request.form.get("amount", 0))
            time = float(request.form.get("time", 0))

        # Create feature array with default V1-V28 values
        features = [time] + [DEFAULT_VALUES[f"V{i}"] for i in range(1, 29)] + [amount]
        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # Scale features
        df = pd.DataFrame([features], columns=feature_names)
        df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]

        # Prepare response
        result = {
            "prediction": int(prediction),
            "probability": float(probability[1]),
            "confidence": float(probability[1] * 100),
            "status": "Fraudulent" if prediction == 1 else "Legitimate",
            "message": (
                "⚠️ This transaction appears to be FRAUDULENT! Please investigate."
                if prediction == 1
                else "✅ This transaction appears to be LEGITIMATE."
            ),
            "amount": amount,
            "time": time,
        }

        if request.is_json:
            return jsonify(result)
        else:
            return render_template("result.html", result=result)

    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        else:
            return render_template("error.html", error=error_msg)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for predictions"""
    try:
        data = request.get_json()

        # Validate input
        if "amount" not in data or "time" not in data:
            return (
                jsonify({"error": "Missing fields: amount and time are required"}),
                400,
            )

        amount = float(data["amount"])
        time = float(data["time"])

        # Create feature array with default V1-V28 values
        features = [time] + [DEFAULT_VALUES[f"V{i}"] for i in range(1, 29)] + [amount]
        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # Scale features
        df = pd.DataFrame([features], columns=feature_names)
        df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]

        return jsonify(
            {
                "prediction": int(prediction),
                "probability": float(probability[1]),
                "confidence": float(probability[1] * 100),
                "status": "Fraudulent" if prediction == 1 else "Legitimate",
                "amount": amount,
                "time": time,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error="Internal server error"), 500
