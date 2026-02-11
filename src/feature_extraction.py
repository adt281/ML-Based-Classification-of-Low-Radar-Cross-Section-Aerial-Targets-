import numpy as np


def extract_features(track, detections):

    if track is None:
        # Noise-only scene
        return {
            "mean_speed": 0.0,
            "speed_variance": 0.0,
            "acceleration_variance": 0.0,
            "mean_cov_trace": 0.0,
            "cov_growth_rate": 0.0,
            "detection_ratio": 0.0,
            "first_detection_range": 0.0,
            "mean_detection_range": 0.0,
            "std_detection_range": 0.0,
            "max_detection_gap": 0,
            "mean_detection_gap": 0
        }

    vx_list = []
    vy_list = []
    cov_traces = []

    detection_ranges = []
    detection_count = 0

    for i, (state, detection_list) in enumerate(zip(track[1:], detections)):

        vx = state.state_vector[1, 0]
        vy = state.state_vector[3, 0]

        vx_list.append(vx)
        vy_list.append(vy)

        cov_traces.append(np.trace(state.covar))

        if detection_list is not None and len(detection_list) > 0:
            detection_count += 1

            # Use first detection (already gated by tracker)
            det = detection_list[0]
            range_ = det.state_vector[1, 0]
            detection_ranges.append(range_)

    vx_list = np.array(vx_list)
    vy_list = np.array(vy_list)
    cov_traces = np.array(cov_traces)

    speed = np.sqrt(vx_list**2 + vy_list**2)

    ax = np.diff(vx_list)
    ay = np.diff(vy_list)

    if len(speed) > 1:
        acceleration = np.sqrt(ax**2 + ay**2) / (speed[:-1] + 1e-6)
        acceleration_variance = np.var(acceleration)
    else:
        acceleration_variance = 0.0

    detection_ratio = detection_count / len(detections)

    if len(detection_ranges) > 0:
        first_detection_range = detection_ranges[0]
        mean_detection_range = np.mean(detection_ranges)
        std_detection_range = np.std(detection_ranges)
    else:
        first_detection_range = 0.0
        mean_detection_range = 0.0
        std_detection_range = 0.0

    gaps = []
    current_gap = 0

    for d_list in detections:
        if d_list is None or len(d_list) == 0:
            current_gap += 1
        else:
            if current_gap > 0:
                gaps.append(current_gap)
                current_gap = 0

    if current_gap > 0:
        gaps.append(current_gap)

    if len(gaps) > 0:
        max_gap = np.max(gaps)
        mean_gap = np.mean(gaps)
    else:
        max_gap = 0
        mean_gap = 0

    if len(cov_traces) > 1:
        cov_growth = np.polyfit(range(len(cov_traces)), cov_traces, 1)[0]
    else:
        cov_growth = 0.0

    return {
        "mean_speed": np.mean(speed),
        "speed_variance": np.var(speed),
        "acceleration_variance": acceleration_variance,
        "mean_cov_trace": np.mean(cov_traces),
        "cov_growth_rate": cov_growth,
        "detection_ratio": detection_ratio,
        "first_detection_range": first_detection_range,
        "mean_detection_range": mean_detection_range,
        "std_detection_range": std_detection_range,
        "max_detection_gap": max_gap,
        "mean_detection_gap": mean_gap
    }
