
# ============================================================
# TELCO CUSTOMER CHURN - MACHINE LEARNING PROJECT
# ============================================================
#
# Goal:
# Predict whether a customer will churn (leave the company).
#
# Target:
#   0 = No Churn
#   1 = Churn
#
# Main workflow:
# Data → Cleaning → Train/Test Split → Preprocessing
#      → Model Comparison → Hyperparameter Tuning
#      → Final Evaluation → Save Model
#
# Important concepts learned in this project:
# - Missing values
# - Categorical encoding
# - Feature scaling
# - ColumnTransformer
# - Pipelines
# - Train/Test Split
# - Stratification
# - Cross-Validation
# - Model Comparison
# - GridSearchCV
# - Hyperparameter Tuning
# - Precision / Recall / F1
# - ROC-AUC
# - Precision-Recall Curve
# - predict_proba()
# - Threshold tuning
# - Model saving with Joblib
# - Data leakage prevention
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    r"Telco-Customer-Churn.csv"
)


# ============================================================
# 3. BASIC DATA EXPLORATION
# ============================================================

# Check target distribution.
# This helps us understand whether the classes are balanced.

print("Churn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print(df["Churn"].value_counts(normalize=True))


# Check the columns in the dataset.

print("\nColumns:")
print(df.columns.tolist())


# Check missing values.

print("\nMissing values:")
print(df.isna().sum())


# TotalCharges is originally stored as an object/string.

print("\nTotalCharges dtype before conversion:")
print(df["TotalCharges"].dtype)


# ============================================================
# 4. CLEAN DATA
# ============================================================

# TotalCharges contains blank values.
#
# We convert it to numeric.
#
# errors="coerce" means invalid values become NaN.
# The NaN values will later be handled by SimpleImputer
# inside our preprocessing pipeline.

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges dtype after conversion:")
print(df["TotalCharges"].dtype)


# ============================================================
# 5. DEFINE FEATURES (X) AND TARGET (y)
# ============================================================

# Remove:
# - Churn → because it is our target
# - customerID → because it is only an identifier and has
#   no useful predictive meaning.

X = df.drop(
    ["Churn", "customerID"],
    axis=1
)


# Convert the target from:
# "No" / "Yes"
#
# into:
# 0 / 1
#
# 0 = No Churn
# 1 = Churn

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 6. DEFINE FEATURE TYPES
# ============================================================

# Numerical features:
# These contain numerical values.

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# Categorical features:
# These contain categories such as Yes/No or different
# service types.

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ============================================================
# 7. PREPROCESSING
# ============================================================

# Numerical preprocessing:
#
# 1. Missing values → median
# 2. Scale numerical features
#
# Scaling is especially useful for models that are sensitive
# to feature magnitude.

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing:
#
# 1. Missing values → most frequent value
# 2. Convert categories into numerical columns using
#    One-Hot Encoding.
#
# handle_unknown="ignore" prevents errors when the test/new
# data contains a category that wasn't seen during training.

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ColumnTransformer applies different preprocessing to
# numerical and categorical columns.

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_features),
    ("cat", cat_pipeline, categorical_features)
])


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

# Split the data BEFORE training.
#
# The test set must remain unseen until the final evaluation.
#
# stratify=y keeps approximately the same class distribution
# in both training and test sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}


# Compare the models using 5-fold cross-validation.
#
# IMPORTANT:
# The preprocessing is inside the Pipeline.
#
# This prevents data leakage because preprocessing is fitted
# separately inside each CV training fold.

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(f"\n{name}")
    print("CV Scores:", scores)
    print("Mean CV Accuracy:", scores.mean())


# ============================================================
# 10. XGBOOST PIPELINE
# ============================================================

# We previously found XGBoost to be a strong candidate,
# so we will tune its hyperparameters.

xgboost_pipeline = Pipeline([
    ("preprocessor", preprocessor),

    ("model", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])


# ============================================================
# 11. HYPERPARAMETER TUNING - GRIDSEARCHCV
# ============================================================


param_grid = {

    "model__n_estimators": [
        50,
        100,
        150
    ],

    "model__learning_rate": [
        0.01,
        0.1,
        0.2
    ],

    "model__max_depth": [
        2,
        3,
        4
    ]
}


grid = GridSearchCV(
    xgboost_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)


grid.fit(
    X_train,
    y_train
)


print("\n================ GRID SEARCH ================\n")

print("Best Parameters:")
print(grid.best_params_)

print("\nBest CV Accuracy:")
print(grid.best_score_)


# best_estimator_ is the actual trained Pipeline using the
# best hyperparameters discovered by GridSearchCV.

best_model = grid.best_estimator_


# ============================================================
# 12. FINAL TEST SET EVALUATION
# ============================================================

# The test set has remained untouched until now.
#
# This gives us our final estimate of how the model performs
# on unseen data.

predictions = best_model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    pos_label=1
)

recall = recall_score(
    y_test,
    predictions,
    pos_label=1
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label=1
)


print("\n================ FINAL TEST ================\n")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)


# ============================================================
# 13. PREDICT PROBABILITIES
# ============================================================

# predict() gives the final class:
#
# 0 = No Churn
# 1 = Churn
#
# predict_proba() gives the probability for each class.
#
# [:, 1] selects the probability of class 1 (Churn).

churn_probability = best_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. ROC-AUC
# ============================================================

# ROC-AUC measures how well the model separates the two
# classes across different classification thresholds.
#
# IMPORTANT:
# ROC-AUC uses probabilities, not hard predictions.

auc = roc_auc_score(
    y_test,
    churn_probability
)

print("\nROC-AUC:", auc)


# ============================================================
# 15. PRECISION-RECALL CURVE
# ============================================================

# The Precision-Recall curve shows the trade-off between
# Precision an
joblib.dump(best_model, filename="churn_model.pkl")
