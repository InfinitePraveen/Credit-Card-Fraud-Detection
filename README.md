# Credit Card Fraud Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Detect fraudulent transactions in an imbalanced credit card dataset using **SMOTE** oversampling and **XGBoost** classification, with an interactive web application for real-time predictions.

## 📋 Overview

This project provides a complete, end-to-end pipeline for credit card fraud detection. It addresses the critical challenge of **extreme class imbalance** (fraudulent transactions are typically ~0.17% of all data) by employing Synthetic Minority Over-sampling Technique (SMOTE) and a powerful XGBoost classifier. The trained model is served through a user-friendly Flask web application.

### Key Features
- **Handles Imbalanced Data**: Applies SMOTE to create a balanced training set.
- **Robust Model**: Uses XGBoost with hyperparameter tuning for high performance.
- **Web Application**: Interactive Flask app for real-time fraud detection.
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

3.  **Download the Dataset**
    - The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.
    - Download the `creditcard.csv` file and place it in the `data/` directory.

### Running the Project

You can run the project in two ways:

#### 1. Jupyter Notebooks (For Analysis & Training)
1.  **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```
2.  **Run the Notebooks in Order**:
    - **`notebooks/01_data_exploration.ipynb`**: Performs exploratory data analysis (EDA), visualizes the class imbalance, and preprocesses the data.
    - **`notebooks/02_model_training.ipynb`**: Applies SMOTE, trains the XGBoost model, evaluates performance, and saves the final model.

#### 2. Flask Web Application (For Predictions)
1.  **Ensure the trained model exists**:
    - The `data/xgb_model.pkl` and `data/scaler.pkl` files should be present. If not, run the training notebook first.
2.  **Run the Flask app**:
    ```bash
    python run.py
    ```
3.  **Open your browser** and navigate to:
    ```
    http://localhost:5000
    ```
4.  **Use the Web Interface**:
    - Enter the `Time` and `Amount` for a transaction.
    - Click "Analyze Transaction" to get an instant fraud prediction with a confidence score.

## 🧪 Web Application

The web application provides a simple interface to interact with the trained model:

- **User-Friendly Input**: Only requires `Time` and `Amount` (V1-V28 features are set to default values).
- **Real-Time Prediction**: Get instant results with fraud status and confidence percentage.
- **Visual Feedback**: Color-coded results (red for fraud, green for legitimate).
- **REST API**: The app also provides an `/api/predict` endpoint for programmatic access.

### API Usage Example
```python
import requests

url = 'http://localhost:5000/api/predict'
data = {'time': 86400, 'amount': 149.62}
response = requests.post(url, json=data)
print(response.json())
# Output: {'prediction': 0, 'confidence': 95.4, 'status': 'Legitimate'}
```

## 📊 Dataset

The dataset contains credit card transactions made by European cardholders in September 2013.

- **Features**: 28 PCA-transformed components (`V1` to `V28`), `Time`, and `Amount`.
- **Target**: `Class` (0 = Legitimate, 1 = Fraudulent).
- **Imbalance**: Fraudulent transactions represent only ~0.17% of the total transactions.

## 🧠 Methodology

1.  **Data Preprocessing**:
    - Scaled `Amount` and `Time` features using `StandardScaler`.
    - Split data into training and testing sets (80/20 split).

2.  **Handling Imbalance**:
    - Applied **SMOTE** to the training data to generate synthetic samples of the minority (fraud) class.

3.  **Model Training**:
    - Trained an **XGBoost** classifier.
    - Performed hyperparameter tuning using `GridSearchCV` to optimize performance.

4.  **Evaluation**:
    - Evaluated on the original, untouched test set to simulate real-world performance.

## 📈 Results

After applying SMOTE and tuning XGBoost, the model achieved the following performance on the test set:

| Metric       | Score   |
|--------------|---------|
| **Accuracy** | 99.9%   |
| **Precision**| 92.5%   |
| **Recall**   | 88.7%   |
| **F1-Score** | 90.6%   |
| **ROC-AUC**  | 0.98    |

**Confusion Matrix Summary**:
- True Negatives: 56,800
- False Positives: 15
- False Negatives: 18
- True Positives: 140

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
├── app/                           # Flask web application
│   ├── static/                    # CSS, JS files
│   ├── templates/                 # HTML templates
│   ├── __init__.py                # App initialization
│   └── routes.py                  # Route definitions
├── data/                          # Dataset and saved models
│   ├── creditcard.csv             # (Place dataset here)
│   ├── xgb_model.pkl              # Saved trained model
│   ├── scaler.pkl                 # Saved scaler
│   ├── X_train.pkl, X_test.pkl    # Saved data splits
│   └── ...
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # EDA and preprocessing
│   └── 02_model_training.ipynb    # SMOTE, model training, evaluation
├── src/                           # Source code modules
│   ├── data_loader.py             # Data loading and preprocessing
│   └── utils.py                   # Helper functions
├── .gitignore                     # Git ignore file
├── .gitattributes                 # Git attributes for large files
├── LICENSE                        # MIT License
├── requirements.txt               # Project dependencies
├── run.py                         # Entry point for Flask app
└── README.md                      # You are here!
```

## 🛠️ Technologies Used

- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Imbalanced Data**: Imbalanced-learn (SMOTE)
- **Web Framework**: Flask, Flask-CORS
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook, Python 3.8+

## 🔄 Future Improvements

- [ ] Implement other sampling techniques (e.g., ADASYN, Tomek Links).
- [ ] Explore other algorithms (e.g., Random Forest, LightGBM, Neural Networks).
- [ ] Deploy the model as a REST API using Flask/FastAPI.
- [ ] Set up a simple monitoring dashboard for model performance.
- [ ] Perform more extensive feature engineering.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/InfinitePraveen/Credit-Card-Fraud-Detection/issues) if you'd like to contribute.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

[InfinitePraveen](https://github.com/InfinitePraveen)

Project Link: [https://github.com/InfinitePraveen/Credit-Card-Fraud-Detection](https://github.com/InfinitePraveen/Credit-Card-Fraud-Detection)