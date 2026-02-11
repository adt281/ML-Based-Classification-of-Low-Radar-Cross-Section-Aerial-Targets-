import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.dataset import generate_dataset


def train_stage2(samples_per_class=300):

    print("Generating dataset...")
    df = generate_dataset(samples_per_class=samples_per_class)

    # Keep only credible tracks
    df = df[df["stage1_label"] == "credible"]

    # Stage 2 labels
    X = df.drop(columns=["stage1_label", "stage2_label", "stage3_label"])
    y = df["stage2_label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    

    print("Stage2 unique labels:", df["stage2_label"].unique())
    print("Counts:\n", df["stage2_label"].value_counts())

    print("Training Stage 2 model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(model, "stage2_model.pkl")
    joblib.dump(encoder, "stage2_encoder.pkl")

    print("\nStage 2 model saved.")


if __name__ == "__main__":
    train_stage2(samples_per_class=300)
