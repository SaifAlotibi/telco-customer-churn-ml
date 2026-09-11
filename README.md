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

`ColumnTransformer` was used to apply the appropriate preprocessing to each f
