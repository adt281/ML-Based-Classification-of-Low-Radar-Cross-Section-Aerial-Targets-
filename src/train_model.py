import numpy as np
from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

# -------------------------------------------------
# Setup output directory
# -------------------------------------------------
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
X, y, scene_ids = load_dataset("radar_dataset.npz")

print("Dataset shape:", X.shape)

# -------------------------------------------------
# Train / Test split (scene-level)
# -------------------------------------------------
unique_scenes = np.unique(scene_ids)

train_scenes, test_scenes = train_test_split(
    unique_scenes,
    test_size=0.2,
    random_state=42
)

train_mask = np.isin(scene_ids, train_scenes)
test_mask = np.isin(scene_ids, test_scenes)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

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

# Save model (fixed names)
joblib.dump(model, f"{OUTPUT_DIR}/radar_classifier.pkl")
joblib.dump(model, "radar_classifier.pkl")  # for result.py compatibility

print("Model saved at results/ and root directory")

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)

# ---- Classification Report ----
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_text = classification_report(y_test, y_pred)

print("\nClassification Report")
print(report_text)

with open(f"{OUTPUT_DIR}/report.json", "w") as f:
    json.dump(report_dict, f, indent=4)

with open(f"{OUTPUT_DIR}/report.txt", "w") as f:
    f.write(report_text)

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Empty","Aircraft","Stealth"],
            yticklabels=["Empty","Aircraft","Stealth"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# ---- Feature Importance ----
importances = model.feature_importances_

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances)

plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")

plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
plt.close()

# -------------------------------------------------
# Save predictions
# -------------------------------------------------
np.savez(
    f"{OUTPUT_DIR}/predictions.npz",
    y_true=y_test,
    y_pred=y_pred
)

# -------------------------------------------------
# Save metadata
# -------------------------------------------------
metadata = {
    "n_estimators": 200,
    "max_depth": None,
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test))
}

with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)