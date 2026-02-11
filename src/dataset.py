import pandas as pd
from tqdm import tqdm

from src.simulation import simulate_scene
from src.tracking import run_tracker
from src.feature_extraction import extract_features


# ============================================================
# Scene Selection Per Stage
# ============================================================

STAGE_SCENE_MAP = {
    "stage1": ["noise", "bird", "aircraft", "stealth"],
    "stage2": ["bird", "aircraft", "stealth"],
    "stage3": ["aircraft", "stealth"]
}


def generate_dataset(stage="stage1", samples_per_class=300):

    if stage not in STAGE_SCENE_MAP:
        raise ValueError("Invalid stage. Choose: stage1, stage2, stage3")

    data = []
    scene_types = STAGE_SCENE_MAP[stage]

    for scene_type in scene_types:

        print(f"\nGenerating data for: {scene_type}")

        for _ in tqdm(range(samples_per_class)):

            truth_states, detections, transition_model, measurement_model, metadata = simulate_scene(
                scene_type=scene_type
            )

            # -------------------------
            # Noise case (no tracker)
            # -------------------------
            if scene_type == "noise":

                features = {
                    "mean_speed": 0.0,
                    "speed_variance": 0.0,
                    "acceleration_variance": 0.0,
                    "mean_cov_trace": 0.0,
                    "cov_growth_rate": 0.0,
                    "detection_ratio": 0.0,
                    "first_detection_range": 0.0,
                    "mean_detection_range": 0.0,
                    "std_detection_range": 0.0,
                    "max_detection_gap": len(detections),
                    "mean_detection_gap": len(detections)
                }

            else:
                track = run_tracker(
                    truth_states,
                    detections,
                    transition_model,
                    measurement_model
                )

                features = extract_features(track, detections)

            # Attach hierarchical labels
            features["stage1_label"] = metadata["stage1"]
            features["stage2_label"] = metadata["stage2"]
            features["stage3_label"] = metadata["stage3"]

            data.append(features)

    df = pd.DataFrame(data)

    return df


if __name__ == "__main__":
    df = generate_dataset(stage="stage1", samples_per_class=50)
    print(df.head())
