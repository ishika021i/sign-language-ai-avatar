import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_FILE = "data/train_normalized.csv"
MODEL_FILE = "models/sign_model.joblib"


print("Loading dataset...")

df = pd.read_csv(DATA_FILE)


X = df.drop("label", axis=1)
y = df["label"]


print("Total samples:", len(X))
print("Total features:", X.shape[1])
print("Classes:", y.nunique())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining model...")


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)


print("Training complete.")


print("\nEvaluating model...")


predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)


print(f"\nAccuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


joblib.dump(
    model,
    MODEL_FILE
)


print("\nModel saved:")
print(MODEL_FILE)
