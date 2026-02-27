import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.pkl")


def train_and_evaluate():
    df = pd.read_csv(DATA_PATH)
    df.drop("customerID", axis=1, inplace=True)

    for col in df.select_dtypes(include="object").columns:
        if col != "Churn":
            df[col] = LabelEncoder().fit_transform(df[col])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    new_model = LogisticRegression(max_iter=1000)
    new_model.fit(X_train, y_train)

    new_auc = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])

    if os.path.exists(MODEL_PATH):
        old_model = joblib.load(MODEL_PATH)
        old_auc = roc_auc_score(
            y_test, old_model.predict_proba(X_test)[:, 1]
        )

        if new_auc > old_auc:
            joblib.dump(new_model, MODEL_PATH)
            return f"Model retrained ✅ (AUC improved {old_auc:.3f} → {new_auc:.3f})"
        else:
            return f"Retraining skipped ℹ️ (Old AUC {old_auc:.3f} better)"
    else:
        joblib.dump(new_model, MODEL_PATH)
        return f"Initial model trained ✅ (AUC {new_auc:.3f})"


if __name__ == "__main__":
    print(train_and_evaluate())