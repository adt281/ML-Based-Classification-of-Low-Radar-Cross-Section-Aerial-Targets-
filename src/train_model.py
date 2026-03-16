import numpy as np
from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------------------------
# Load dataset
# -------------------------------------------------

X, y = load_dataset("radar_dataset.npz")

print("Dataset shape:", X.shape)

# -------------------------------------------------
# Train / Test split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Model
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, "radar_classifier.pkl")
print("Model saved as radar_classifier.pkl")

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

y_pred = model.predict(X_test)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Empty","Aircraft","Stealth"],
            yticklabels=["Empty","Aircraft","Stealth"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -------------------------------------------------
# Feature Importance
# -------------------------------------------------

importances = model.feature_importances_

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()