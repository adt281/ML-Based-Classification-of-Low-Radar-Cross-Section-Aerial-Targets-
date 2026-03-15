''' .csv for smoke testing [since human readable] and .npz for actual ML training '''

import numpy as np
from feature_extraction import extract_features_timestep
from tracking import run_tracking
from feature_extraction import export_csv

# -------------------------------------------------
# Label mapping
# -------------------------------------------------

LABEL_MAP = {
    "empty": 0,
    "aircraft": 1,
    "stealth": 2
}


# -------------------------------------------------
# Generate dataset from one scene
# -------------------------------------------------

def generate_scene_dataset(scene_type, num_steps=80):

    result = run_tracking(scene_type, num_steps=num_steps)

    scene = result["scene"]
    num_steps = scene["metadata"]["num_steps"]

    X_scene = []
    y_scene = []

    label = LABEL_MAP[scene_type]

    for t in range(num_steps):

        features = extract_features_timestep(result, t)

        X_scene.append(features)
        y_scene.append(label)

    return np.array(X_scene), np.array(y_scene)


# -------------------------------------------------
# Build full dataset
# -------------------------------------------------

def build_dataset(
        aircraft_scenes=1,
        stealth_scenes=1,
        empty_scenes=1,
        num_steps=80):

    X_all = []
    y_all = []

    # Aircraft scenes
    for i in range(aircraft_scenes):

        X, y = generate_scene_dataset("aircraft", num_steps)

        X_all.append(X)
        y_all.append(y)

        if (i+1) % 50 == 0:
            print(f"Aircraft scenes generated: {i+1}")

    # Stealth scenes
    for i in range(stealth_scenes):

        X, y = generate_scene_dataset("stealth", num_steps)

        X_all.append(X)
        y_all.append(y)

        if (i+1) % 50 == 0:
            print(f"Stealth scenes generated: {i+1}")

    # Empty scenes
    for i in range(empty_scenes):

        X, y = generate_scene_dataset("empty", num_steps)

        X_all.append(X)
        y_all.append(y)

        if (i+1) % 50 == 0:
            print(f"Empty scenes generated: {i+1}")

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    return X_all, y_all


# -------------------------------------------------
# Save dataset
# -------------------------------------------------

def save_dataset(X, y, filename="radar_dataset.npz"):

    np.savez_compressed(
        filename,
        X=X,
        y=y
    )

    print("Dataset saved:", filename)
    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])


# -------------------------------------------------
# Load dataset
# -------------------------------------------------

def load_dataset(filename="radar_dataset.npz"):

    data = np.load(filename)

    X = data["X"]
    y = data["y"]

    return X, y


# -------------------------------------------------
# Main (dataset generation)
# -------------------------------------------------
if __name__ == "__main__":

    X, y = build_dataset(
        aircraft_scenes=1,
        stealth_scenes=1,
        empty_scenes=1
    )

    save_dataset(X, y)

    export_csv(X, y)