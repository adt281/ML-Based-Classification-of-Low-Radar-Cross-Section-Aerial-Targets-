import pandas as pd
from tqdm import tqdm

from src.simulation import simulate_target
from src.tracking import run_tracker
from src.feature_extraction import extract_features


def generate_dataset(samples_per_class=300):

    data = []

    target_classes = ["bird", "aircraft", "stealth"]

    for target_type in target_classes:

        print(f"\nGenerating data for: {target_type}")

        for _ in tqdm(range(samples_per_class)):

            truth_states, detections, transition_model, measurement_model, _ = simulate_target(
                target_type=target_type
            )

            track = run_tracker(
                truth_states,
                detections,
                transition_model,
                measurement_model
            )

            features = extract_features(track, detections)

            features["label"] = target_type

            data.append(features)

    df = pd.DataFrame(data)

    return df


if __name__ == "__main__":
    df = generate_dataset(samples_per_class=50)
    print(df.head())
