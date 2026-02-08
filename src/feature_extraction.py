import numpy as np


def extract_features(track, detections):

    vx_list = []
    vy_list = []
    cov_traces = []
    detection_count = 0

    for state, detection in zip(track[1:], detections):

        vx = state.state_vector[1, 0]
        vy = state.state_vector[3, 0]

        vx_list.append(vx)
        vy_list.append(vy)

        cov_traces.append(np.trace(state.covar))

        if detection is not None:
            detection_count += 1

    vx_list = np.array(vx_list)
    vy_list = np.array(vy_list)

    speed = np.sqrt(vx_list**2 + vy_list**2)

    ax = np.diff(vx_list)
    ay = np.diff(vy_list)

    acceleration = np.sqrt(ax**2 + ay**2) / speed[:-1]


    features = {
        "mean_speed": np.mean(speed),
        "speed_variance": np.var(speed),
        "acceleration_variance": np.var(acceleration) if len(acceleration) > 0 else 0.0,
        "mean_cov_trace": np.mean(cov_traces),
        "detection_ratio": detection_count / len(detections)
    }

    return features
