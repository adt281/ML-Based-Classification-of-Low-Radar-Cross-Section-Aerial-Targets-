import numpy as np
from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
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

print("Original Dataset shape:", X.shape)

# -------------------------------------------------
# Keep only last timesteps (t >= 50)
# -------------------------------------------------
mask = np.zeros(len(scene_ids), dtype=bool)

for sid in np.unique(scene_ids):
    idx = np.where(scene_ids == sid)[0]
    mask[idx[50:]] = True

X = X[mask]
y = y[mask]
scene_ids = scene_ids[mask]

print("Filtered Dataset shape:", X.shape)

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

# Save model
joblib.dump(model, f"{OUTPUT_DIR}/radar_classifier.pkl")
joblib.dump(model, "radar_classifier.pkl")

print("Model saved at results/ and root directory")

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ---- Standard Classification Report ----
print("\nStandard Classification Report")
print(classification_report(y_test, y_pred))

# ---- Log Loss ----
ll = log_loss(y_test, y_prob)
print("Log Loss:", ll)

# ---- Confidence stats ----
confidence = np.max(y_prob, axis=1)
print("Mean confidence:", np.mean(confidence))
print("Min confidence:", np.min(confidence))
print("Max confidence:", np.max(confidence))

# -------------------------------------------------
# STRICT THRESHOLD EVALUATION (P >= 0.9)
# -------------------------------------------------
threshold = 0.85

y_pred_strict = []

for i, p in enumerate(y_prob):
    max_prob = np.max(p)
    pred_class = np.argmax(p)

    if max_prob >= threshold:
        y_pred_strict.append(pred_class)
    else:
        # force wrong class (pick any class ≠ true label)
        true = y_test[i]
        wrong_class = (true + 1) % 3
        y_pred_strict.append(wrong_class)

y_pred_strict = np.array(y_pred_strict)

print(f"\nStrict Classification Report (P >= {threshold})")
print(classification_report(y_test, y_pred_strict))



# ---- Confusion Matrix (standard) ----
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
# Save predictions + probabilities
# -------------------------------------------------
np.savez(
    f"{OUTPUT_DIR}/predictions.npz",
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_prob,
    y_pred_strict=y_pred_strict
)

# -------------------------------------------------
# Save metadata
# -------------------------------------------------
metadata = {
    "n_estimators": 200,
    "max_depth": None,
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "log_loss": float(ll),
    "mean_confidence": float(np.mean(confidence)),
    "threshold": threshold
}

with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)