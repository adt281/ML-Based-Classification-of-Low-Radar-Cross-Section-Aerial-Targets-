from src.simulation import simulate_target
from src.tracking import run_tracker
from src.feature_extraction import extract_features

truth_states, detections, transition_model, measurement_model = simulate_target(Pd=0.85)

track = run_tracker(truth_states, detections, transition_model, measurement_model)

features = extract_features(track, detections)

print("\nExtracted Features:")
for key, value in features.items():
    print(f"{key}: {value:.4f}")
