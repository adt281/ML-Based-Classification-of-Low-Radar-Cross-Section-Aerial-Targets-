import numpy as np
import matplotlib.pyplot as plt

from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from numpy.linalg import inv


# ============================================================
# Utility: Cartesian conversion
# ============================================================

def polar_to_cartesian(det):
    bearing = det.state_vector[0, 0]
    range_ = det.state_vector[1, 0]
    x = range_ * np.cos(bearing)
    y = range_ * np.sin(bearing)
    return x, y


# ============================================================
# Phase 6: Ambiguity + SNR-based association
# ============================================================

def select_best_detection(prediction,
                          detection_list,
                          updater,
                          gate_threshold=9.0):

    if detection_list is None or len(detection_list) == 0:
        return None, None

    valid_detections = []
    mahalanobis_distances = []
    snr_values = []

    for det in detection_list:

        innovation = updater.measurement_model.function(
            prediction,
            noise=False
        ) - det.state_vector

        H = updater.measurement_model.jacobian(prediction)
        P = prediction.covar
        R = updater.measurement_model.covar()

        S = H @ P @ H.T + R

        d2 = innovation.T @ inv(S) @ innovation
        d2 = d2.item()

        if d2 < gate_threshold:
            valid_detections.append(det)
            mahalanobis_distances.append(d2)
            snr_values.append(det.metadata.get("snr_db", 30))

    if len(valid_detections) == 0:
        return None, None

    sorted_indices = np.argsort(mahalanobis_distances)
    best_idx = sorted_indices[0]
    selected_idx = best_idx

    if len(sorted_indices) > 1:

        second_idx = sorted_indices[1]

        d1 = mahalanobis_distances[best_idx]
        d2 = mahalanobis_distances[second_idx]
        current_snr_db = snr_values[best_idx]

        ambiguity_threshold = 2.0
        ambiguity_factor = max(0, 1 - abs(d2 - d1) / ambiguity_threshold)
        ambiguity_factor = np.clip(ambiguity_factor, 0, 1)

        snr_threshold_db = 15
        snr_factor = max(0, 1 - (current_snr_db / snr_threshold_db))
        snr_factor = np.clip(snr_factor, 0, 1)

        P_wrong = 0.6 * (0.5 * ambiguity_factor + 0.5 * snr_factor)

        if np.random.rand() < P_wrong:
            selected_idx = second_idx

    return valid_detections[selected_idx], mahalanobis_distances[selected_idx]


# ============================================================
# Main Tracker
# ============================================================

def run_tracker(truth_states,
                detections,
                transition_model,
                measurement_model,
                plot=False):

    if truth_states is None or len(truth_states) == 0:
        return None

    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model)

    first_truth = truth_states[0]

    prior = GaussianState(
        StateVector([
            first_truth.state_vector[0, 0],
            first_truth.state_vector[1, 0],
            first_truth.state_vector[2, 0],
            first_truth.state_vector[3, 0],
        ]),
        np.diag([200, 50, 200, 50]),
        timestamp=first_truth.timestamp
    )

    track = Track([prior])

    truth_x, truth_y = [], []
    est_x, est_y = [], []
    meas_x, meas_y = [], []

    # NEW: store all detections for plotting
    all_meas_x, all_meas_y = [], []

    innovation_norms = []
    cov_trace_history = []
    rms_error_history = []

    hit_counter = 0
    miss_streak = 0

    for truth, detection_list in zip(truth_states, detections):

        tx = truth.state_vector[0, 0]
        ty = truth.state_vector[2, 0]

        truth_x.append(tx)
        truth_y.append(ty)

        prediction = predictor.predict(
            track[-1],
            timestamp=truth.timestamp
        )

        # Store ALL detections for visualization
        if detection_list is not None:
            for det in detection_list:
                dx, dy = polar_to_cartesian(det)
                all_meas_x.append(dx)
                all_meas_y.append(dy)

        detection, dist = select_best_detection(
            prediction,
            detection_list,
            updater
        )

        if detection is not None:

            hypothesis = SingleHypothesis(prediction, detection)
            posterior = updater.update(hypothesis)

            hit_counter += 1
            miss_streak = 0

            meas_x_val, meas_y_val = polar_to_cartesian(detection)
            meas_x.append(meas_x_val)
            meas_y.append(meas_y_val)

            innovation_norms.append(dist)

        else:
            posterior = prediction
            miss_streak += 1
            innovation_norms.append(0)

        track.append(posterior)

        ex = posterior.state_vector[0, 0]
        ey = posterior.state_vector[2, 0]

        est_x.append(ex)
        est_y.append(ey)

        cov_trace_history.append(np.trace(posterior.covar))

        rms_error = np.sqrt((ex - tx)**2 + (ey - ty)**2)
        rms_error_history.append(rms_error)

    # ============================================================
    # Summary
    # ============================================================

    print("\n==== Tracking Summary ====")
    print(f"Total hits: {hit_counter}")
    print(f"Final miss streak: {miss_streak}")
    print(f"Final covariance trace: {cov_trace_history[-1]:.2f}")
    print(f"Mean RMS error: {np.mean(rms_error_history):.2f}")
    print(f"Mean innovation distance: {np.mean(innovation_norms):.2f}")

    # ============================================================
    # Diagnostics Plot
    # ============================================================

    if plot:

        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle("Tracking Diagnostics", fontsize=14)

        # Scene
        axs[0, 0].plot(truth_x, truth_y, label="Truth")
        axs[0, 0].scatter(all_meas_x, all_meas_y,
                          s=10, alpha=0.2, color='gray',
                          label="All Detections")
        axs[0, 0].scatter(meas_x, meas_y,
                          marker='x', color='blue',
                          label="Selected Detections")
        axs[0, 0].plot(est_x, est_y,
                       linestyle='--', color='orange',
                       label="EKF Estimate")
        axs[0, 0].set_title("Tracking Scene")
        axs[0, 0].legend()

        axs[0, 1].plot(innovation_norms)
        axs[0, 1].set_title("Innovation Distance")

        axs[1, 0].plot(cov_trace_history)
        axs[1, 0].set_title("Covariance Trace")

        axs[1, 1].plot(rms_error_history)
        axs[1, 1].set_title("RMS Position Error")

        cumulative_hits = np.cumsum(
            [1 if x > 0 else 0 for x in innovation_norms]
        )
        axs[2, 0].plot(cumulative_hits)
        axs[2, 0].set_title("Cumulative Hits")

        miss_indicator = [1 if x == 0 else 0 for x in innovation_norms]
        axs[2, 1].step(range(len(miss_indicator)),
                       miss_indicator,
                       where='mid')
        axs[2, 1].set_title("Miss Indicator")

        for ax in axs.flat:
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return track


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    from src.simulation import simulate_scene

    for target in ["aircraft", "stealth", "bird"]:

        print(f"\nRunning tracker for: {target}")

        truth_states, detections, transition_model, measurement_model, _ = simulate_scene(
            scene_type=target
        )

        run_tracker(
            truth_states,
            detections,
            transition_model,
            measurement_model,
            plot=True
        )
