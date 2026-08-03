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

1. **Clone the repository**
   ```bash
   git clone https://github.com/InfinitePraveen/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Install dependencies**
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **About the Dataset**
   - The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.
   - You can download the `creditcard.csv` file from here also.

### Running the Project

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Run the Notebooks in Order**:
   - **`notebooks/01_data_exploration.ipynb`**: Performs exploratory data analysis (EDA), visualizes the class imbalance, and preprocesses the data.
   - **`notebooks/02_model_training.ipynb`**: Applies SMOTE, trains the XGBoost model, evaluates performance, and saves the final model.

## 📊 Dataset

The dataset contains credit card transactions made by European cardholders in September 2013.

- **Features**: 28 PCA-transformed components (`V1` to `V28`), `Time`, and `Amount`.
- **Target**: `Class` (0 = Legitimate, 1 = Fraudulent).
- **Imbalance**: Fraudulent transactions represent only ~0.17% of the total transactions.

## 🧠 Methodology

1. **Data Preprocessing**:
   - Scaled `Amount` and `Time` features using `StandardScaler`.
   - Split data into training and testing sets (80/20 split).

2. **Handling Imbalance**:
   - Applied **SMOTE** to the training data to generate synthetic samples of the minority (fraud) class.

3. **Model Training**:
   - Trained an **XGBoost** classifier.
   - Performed hyperparameter tuning using `GridSearchCV` to optimize performance.

4. **Evaluation**:
   - Evaluated on the original, untouched test set to simulate real-world performance.

## 📈 Results

*Note: These are illustrative results. Update them with the actual numbers from your model run.*

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
├── data/                          # Dataset and saved models
│   ├── creditcard.csv             # (Place dataset here)
│   ├── xgb_model.pkl              # Saved trained model
│   ├── X_train.pkl, X_test.pkl    # Saved data splits
│   └── ...
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # EDA and preprocessing
│   └── 02_model_training.ipynb    # SMOTE, model training, evaluation
├── src/                           # Source code modules
│   ├── data_loader.py             # Functions for loading and preprocessing data
│   └── utils.py                   # Helper functions for plotting and evaluation
├── .gitignore                     # Git ignore file
├── .gitattributes                 # Git attributes for large files
├── requirements.txt               # Project dependencies
└── README.md                      # You are here!
```

## 🛠️ Technologies Used

- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Imbalanced Data**: Imbalanced-learn (SMOTE)
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

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` (or consider adding a LICENSE file) for more information.

## 📧 Contact

InfinitePraveen - [Link to your GitHub Profile](https://github.com/InfinitePraveen)

Project Link: [https://github.com/InfinitePraveen/Credit-Card-Fraud-Detection](https://github.com/InfinitePraveen/Credit-Card-Fraud-Detection)