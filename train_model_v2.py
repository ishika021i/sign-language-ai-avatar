import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = "data/train_normalized_v2.csv"
MODEL_FILE = "models/sign_model_v2.joblib"

# Load data
df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["label"])
y = df["label"]

print("--------------------------------")
print("Training two-hand model")
print("--------------------------------")
print(f"Samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {y.nunique()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n--------------------------------")
print("Model evaluation")
print("--------------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_FILE)

print("--------------------------------")
print(f"Model saved: {MODEL_FILE}")
print("--------------------------------")