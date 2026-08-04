from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


def is_valid_model(model):
    if model is None or not hasattr(model, "predict"):
        return False
    try:
        test_df = pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
        model.predict(test_df)
        return True
    except Exception:
        return False


def load_or_train_model():
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            if is_valid_model(model):
                return model
        except Exception:
            pass

    if not DATA_PATH.exists():
        raise FileNotFoundError("The dataset file is missing.")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["Class"]

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model
