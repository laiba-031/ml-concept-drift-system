import os
import pandas as pd
from scipy.stats import ks_2samp

def run_drift_detection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "churn.csv")

    train_df = pd.read_csv(DATA_PATH)
    prod_df = pd.read_csv(DATA_PATH)

    train_df.drop("customerID", axis=1, inplace=True)
    prod_df.drop("customerID", axis=1, inplace=True)

    # Simulate drift
    prod_df["MonthlyCharges"] = prod_df["MonthlyCharges"] * 1.5

    drifted = []

    for col in train_df.columns:
        if train_df[col].dtype != "object":
            _, p = ks_2samp(train_df[col], prod_df[col])
            if p < 0.05:
                drifted.append(col)

    return drifted