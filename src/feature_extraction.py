import numpy as np


def extract_features(track, detections):

    velocities = []
    cov_traces = []
    detection_count = 0

    for state, detection in zip(track[1:], detections):

        vx = state.state_vector[1, 0]
        vy = state.state_vector[3, 0]

        speed = np.sqrt(vx**2 + vy**2)
        velocities.append(speed)

        cov_traces.append(np.trace(state.covar))

        if detection is not None:
            detection_count += 1

    velocities = np.array(velocities)

    accelerations = np.diff(velocities)

    features = {
        "mean_speed": np.mean(velocities),
        "speed_variance": np.var(velocities),
        "acceleration_variance": np.var(accelerations) if len(accelerations) > 0 else 0.0,
        "mean_cov_trace": np.mean(cov_traces),
        "detection_ratio": detection_count / len(detections)
    }

    return features
