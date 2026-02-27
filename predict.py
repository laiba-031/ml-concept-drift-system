import os
import pandas as pd
import joblib

# Resolve paths safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)

# Load data (acting as new incoming data)
df = pd.read_csv(DATA_PATH)

# Preprocess (SAME as training)
df.drop("customerID", axis=1, inplace=True)

for col in df.select_dtypes(include="object").columns:
    if col != "Churn":
        df[col] = df[col].astype("category").cat.codes

X = df.drop("Churn", axis=1)

# Predict churn probability
preds = model.predict_proba(X)[:, 1]

print("Sample churn probabilities:")
print(preds[:10])