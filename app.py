
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# 1. CREATE FASTAPI APP
# ============================================================

app = FastAPI()


# ============================================================
# 2. CUSTOMER INPUT SCHEMA
# ============================================================

# This defines the data that our API expects from the user.
#
# The names must match the columns used by our ML pipeline.

# "Whenever someone sends customer information, this is what I expect it to look like."
class Customer(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ============================================================
# 3. LOAD TRAINED ML MODEL
# ============================================================

# We saved the complete preprocessing + model pipeline
# as churn_model.pkl during training.

model = joblib.load(
    r"churn_model.pkl"
)


# ============================================================
# 4. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Churn Prediction API is running!"
    }


# ============================================================
# 5. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: Customer):

    # Convert the customer's input into a dictionary,
    # then into a one-row Pandas DataFrame.
    #
    # The DataFrame structure matches the data used
    # during model training.

    data = pd.DataFrame([
        customer.model_dump()
    ])


    # Make the final prediction.
    #
    # 0 = No Churn
    # 1 = Churn

    prediction = model.predict(data)[0]


    # Get the probability of class 1 (Churn).
    # [[0.27, 0.73]] example of the array
    probability = model.predict_proba(data)[0][1]


    # Return the prediction and probability as JSON.

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }
