import joblib
import pandas as pd

from src.simulation import simulate_target
from src.tracking import run_tracker
from src.feature_extraction import extract_features


def classify_target(simulated_type=None):

    # Load trained model
    model = joblib.load("trained_model.pkl")
    encoder = joblib.load("label_encoder.pkl")

    # Simulate new unknown target
    truth_states, detections, transition_model, measurement_model = simulate_target(
        target_type=simulated_type
    )

    track = run_tracker(
        truth_states,
        detections,
        transition_model,
        measurement_model
    )

    features = extract_features(track, detections)

    X = pd.DataFrame([features])

    prediction_encoded = model.predict(X)[0]
    prediction_label = encoder.inverse_transform([prediction_encoded])[0]

    print("\nTrue Type:", simulated_type)
    print("Predicted Type:", prediction_label)
    print("Features:", features)


if __name__ == "__main__":
    classify_target("stealth")
