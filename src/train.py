
"""
Machine Learning Classification Layer (XGBoost)

Purpose:
    Train a supervised multi-class classifier to distinguish:
        - Bird
        - Conventional Aircraft
        - Stealth Aircraft

Input:
    Dataset generated from tracking layer.
    Each sample represents statistical behavior of one tracked target.

    Feature Vector:
        - mean_speed
        - speed_variance
        - acceleration_variance
        - mean_cov_trace
        - detection_ratio

    Label:
        - bird / aircraft / stealth

Pipeline Steps:

1) Label Encoding
    Convert categorical labels into numeric classes:
        bird → 0
        aircraft → 1
        stealth → 2

2) Train-Test Split
    Split dataset into:
        - Training set (70%)
        - Testing set (30%)
    Stratified sampling ensures equal class distribution.

3) XGBoost Classifier

    Model Type:
        Gradient Boosted Decision Trees (GBDT)

    Objective:
        Multi-class softmax probability estimation
        Minimizes multi-class log-loss:

            L = -Σ y_i log(p_i)

    Core Idea:
        - Build sequence of decision trees.
        - Each tree corrects errors of previous trees.
        - Combine trees into strong ensemble model.

    Key Hyperparameters:
        - n_estimators: number of boosting trees
        - max_depth: depth of each tree
        - learning_rate: contribution of each tree

4) Evaluation Metrics

    Accuracy:
        Overall correct classification rate.

    Confusion Matrix:
        Class-by-class prediction breakdown.

    Classification Report:
        Precision, Recall, F1-score per class.

5) Feature Importance

    Extracted from trained model.
    Indicates which behavioral features contribute most to classification.

Interpretation Context:

    - Bird vs Aircraft separation primarily driven by speed statistics.
    - Aircraft vs Stealth separation primarily driven by:
          detection_ratio and covariance behavior.
    - Acceleration statistics provide secondary refinement.

This module represents the behavioral classification layer.
It learns decision boundaries in feature space derived from radar tracking behavior.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.dataset import generate_dataset


def train_model(samples_per_class=300):

    print("Generating dataset...")
    df = generate_dataset(samples_per_class=samples_per_class)

    X = df.drop(columns=["label"])
    y = df["label"]

    # Encode labels to integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        eval_metric="mlogloss"
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    print("\nFeature Importance:")
    for feature, importance in zip(X.columns, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    train_model(samples_per_class=200)
