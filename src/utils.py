import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import joblib
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regular confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Legitimate", "Fraudulent"],
        yticklabels=["Legitimate", "Fraudulent"],
    )
    axes[0].set_title(title)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=axes[1],
        xticklabels=["Legitimate", "Fraudulent"],
        yticklabels=["Legitimate", "Fraudulent"],
    )
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

    return cm


def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return fpr, tpr


def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkgreen", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()

    return precision, recall


def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    Plot feature importance from tree-based model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n (int): Number of top features to display
        title (str): Plot title
    """
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(importance_df["feature"][:top_n], importance_df["importance"][:top_n])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"{title} - Top {top_n} Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return importance_df


def load_model_and_data(model_path="../data/xgb_model.pkl", data_path="../data/"):
    """
    Load trained model and test data.

    Args:
        model_path (str): Path to saved model
        data_path (str): Path to saved data

    Returns:
        tuple: model, X_test, y_test
    """
    model = joblib.load(model_path)
    X_test = joblib.load(f"{data_path}X_test.pkl")
    y_test = joblib.load(f"{data_path}y_test.pkl")

    print(f"Model loaded from {model_path}")
    print(f"Test data shape: {X_test.shape}")

    return model, X_test, y_test
