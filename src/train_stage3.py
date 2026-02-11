import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.dataset import generate_dataset


def train_stage3(samples_per_class=300):

    print("Generating dataset...")
    df = generate_dataset("stage3", samples_per_class=samples_per_class)

    # 🔷 Only credible aircraft
    df = df[df["stage2_label"] == "aircraft"].copy()

    print("Stage3 unique labels:", df["stage3_label"].unique())
    print("Counts:\n", df["stage3_label"].value_counts())

    # Drop non-aircraft labels
    df = df[df["stage3_label"].notna()]

    # 🔷 Select ONLY detectability features
    feature_cols = [
        "detection_ratio",
        "first_detection_range",
        "mean_detection_range",
        "std_detection_range",
        "max_detection_gap",
        "mean_detection_gap",
        "mean_cov_trace",
        "cov_growth_rate"
    ]

    X = df[feature_cols]
    y = df["stage3_label"]

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

    print("Training Stage 3 model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    ))

    joblib.dump(model, "stage3_model.pkl")
    joblib.dump(encoder, "stage3_encoder.pkl")

    print("\nStage 3 model saved.")


if __name__ == "__main__":
    train_stage3(samples_per_class=300)
