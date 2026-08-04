from flask import Blueprint, render_template, request
import pandas as pd

from app.model import FEATURE_COLUMNS, load_or_train_model


main = Blueprint("main", __name__)
model = load_or_train_model()


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        time_value = data.get("time") or data.get("Time")
        amount_value = data.get("amount") or data.get("Amount")

        if time_value is None or str(time_value).strip() == "" or amount_value is None or str(amount_value).strip() == "":
            return (
                render_template(
                    "index.html",
                    error="Please enter both Time and Amount values.",
                ),
                400,
            )

        time_value = float(str(time_value).strip())
        amount_value = float(str(amount_value).strip())

        if amount_value < 0:
            return (
                render_template(
                    "index.html",
                    error="Amount must be a non-negative number.",
                ),
                400,
            )

        row = {col: 0.0 for col in FEATURE_COLUMNS}
        row["Time"] = time_value
        row["Amount"] = amount_value
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        prediction = int(model.predict(df)[0])
        probability = (
            float(model.predict_proba(df)[0][1])
            if hasattr(model, "predict_proba")
            else (0.95 if prediction == 1 else 0.05)
        )

        confidence = round(probability * 100, 1)
        result = {
            "prediction": prediction,
            "status": "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction",
            "amount": amount_value,
            "time": time_value,
            "confidence": confidence,
            "message": (
                "This transaction looks suspicious and may be fraudulent."
                if prediction == 1
                else "This transaction appears normal based on the model's analysis."
            ),
        }
        return render_template("result.html", result=result)

    except Exception:
        return (
            render_template(
                "index.html",
                error="Something went wrong. Please enter valid numbers for every field.",
            ),
            400,
        )
