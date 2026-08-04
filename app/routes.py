from flask import Blueprint, render_template, request
import joblib
import numpy as np
import pandas as pd
from app import app
import traceback

# Load model and scaler
try:
    model = joblib.load("data/xgb_model.pkl")
    scaler = joblib.load("data/scaler.pkl")
    print("✅ Model and scaler loaded successfully!")

    # Debug: Print feature names expected by model
    if hasattr(model, "feature_names_in_"):
        print(f"📊 Model expects these features: {list(model.feature_names_in_)}")
    else:
        print("⚠️ Model doesn't have feature_names_in_ attribute")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None


main = Blueprint("main", __name__)
model = load_or_train_model()


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        missing = [
            col
            for col in FEATURE_COLUMNS
            if col not in data or str(data[col]).strip() == ""
        ]
        if missing:
            return (
                render_template(
                    "index.html",
                    error=f"Please fill all fields. Missing: {', '.join(missing)}",
                ),
                400,
            )

        row = {col: float(str(data[col]).strip()) for col in FEATURE_COLUMNS}
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        prediction = int(model.predict(df)[0])
        return render_template("result.html", prediction=prediction)

    except Exception:
        return (
            render_template(
                "index.html",
                error="Something went wrong. Please enter valid numbers for every field.",
            ),
            400,
        )
