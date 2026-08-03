# Credit Card Fraud Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Detect fraudulent transactions in an imbalanced credit card dataset using **SMOTE** oversampling and **XGBoost** classification.

## 📋 Overview

This project provides a complete, end-to-end pipeline for credit card fraud detection. It addresses the critical challenge of **extreme class imbalance** (fraudulent transactions are typically ~0.17% of all data) by employing Synthetic Minority Over-sampling Technique (SMOTE) and a powerful XGBoost classifier.

### Key Features
- **Handles Imbalanced Data**: Applies SMOTE to create a balanced training set.
- **Robust Model**: Uses XGBoost with hyperparameter tuning for high performance.
- **Comprehensive Evaluation**: Provides detailed metrics (Precision, Recall, F1, ROC-AUC) and visualizations (Confusion Matrix, ROC Curve, Feature Importance).
- **Modular Code**: Clean, well-documented structure separating data processing, utilities, and model training.
- **Easy to Reproduce**: Jupyter notebooks guide you through the entire workflow.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/InfinitePraveen/Credit-Card-Fraud-Detection.git
    cd Credit-Card-Fraud-Detection
    ```

2.  **Install dependencies**
It's recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```
3.  **About the Dataset**

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.