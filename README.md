# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to churn.

## Project Overview

Customer churn is an important business problem for telecommunications companies. The goal of this project is to build a machine learning model that predicts customer churn based on customer information such as contract type, tenure, monthly charges, internet services, and payment method.

The project covers the complete machine learning workflow:

* Data cleaning
* Data preprocessing
* Feature engineering
* Feature selection
* Model comparison
* Hyperparameter tuning
* Model evaluation
* Probability prediction
* Model deployment

## Dataset

The project uses the **IBM Telco Customer Churn** dataset.

The dataset contains information about 7,000+ telecommunications customers, including:

* Customer demographics
* Services
* Contract information
* Tenure
* Monthly charges
* Total charges
* Churn status

The target variable is:

* `0` → Customer did not churn
* `1` → Customer churned

## Machine Learning Workflow

### 1. Data Preparation

* Loaded the dataset using Pandas
* Converted `TotalCharges` to a numeric column
* Handled missing values using `SimpleImputer`
* Removed `customerID` because it is an identifier rather than a predictive feature

### 2. Preprocessing

Numerical features were:

* Imputed using the median
* Scaled using `StandardScaler`

Categorical features were:

* Imputed using the most frequent value
* Encoded using `OneHotEncoder`

`ColumnTransformer` was used to apply the appropriate preprocessing to each feature type.

### 3. Model Comparison

Several classification algorithms were compared using 5-fold cross-validation:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### 4. Hyperparameter Tuning

XGBoost was tuned using `GridSearchCV` to find better hyperparameter values.

The search included:

* `n_estimators`
* `learning_rate`
* `max_depth`

### 5. Model Evaluation

The final model was evaluated on an unseen test set using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

The model also produces churn probabilities using `predict_proba()`.

## Deployment

The trained model is saved using Joblib:

```python
churn_model.pkl
```

A **FastAPI** backend provides a `/predict` endpoint for making predictions.

A **Streamlit** frontend provides a simple user interface where customer information can be entered and a churn prediction can be generated.

### Architecture

```text
User
  ↓
Streamlit Frontend
  ↓
FastAPI API
  ↓
Trained ML Pipeline
  ↓
Churn Prediction
```

## Project Structure

```text
ML_project_1/
│
├── train.py
├── app.py
├── frontend.py
├── churn_model.pkl
├── Telco-Customer-Churn.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This trains the model and saves the trained pipeline as:

```text
churn_model.pkl
```

### 3. Start the FastAPI backend

```bash
python -m uvicorn app:app --reload
```

The API will run locally on:

```text
http://127.0.0.1:8000
```

### 4. Start the Streamlit frontend

```bash
python -m streamlit run frontend.py
```

The Streamlit application will provide the interface for making predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib
* FastAPI
* Uvicorn
* Streamlit

## Key Concepts Practiced

This project was built to practice practical machine learning concepts including:

* Train/Test Split
* Cross-Validation
* Pipelines
* ColumnTransformer
* Missing Value Imputation
* One-Hot Encoding
* Feature Scaling
* Feature Engineering
* Feature Selection
* Model Comparison
* Hyperparameter Tuning
* Classification Metrics
* ROC-AUC
* Precision-Recall
* Probability Prediction
* Model Deployment

## Future Improvements

Possible improvements include:

* More extensive hyperparameter tuning
* Threshold optimization based on business requirements
* Model explainability using SHAP
* Improved frontend design
* Deployment to a cloud platform
* Monitoring model performance after deployment
