"""
Extended Kalman Filter (EKF) Tracking Layer

Purpose:
    Estimate the target state X = [x, vx, y, vy]^T
    using noisy radar measurements z = [θ, r].

Why EKF?
    The radar measurement model is nonlinear:

        r = sqrt(x^2 + y^2)
        θ = arctan2(y, x)

    Therefore a linear Kalman Filter cannot be used.
    EKF linearizes the nonlinear measurement function locally.

System Models:

1) Motion Model (State Transition)
    X_{k+1} = F X_k + w_k

    F = Constant Velocity transition matrix
    w_k ~ N(0, Q)  (process noise)

    This model:
        - Propagates state forward (prediction step)
        - Handles motion continuity
        - Enables tracking during missed detections

2) Measurement Model
    z_k = h(X_k) + v_k

    h(X) = nonlinear Cartesian → Polar conversion
    v_k ~ N(0, R)  (measurement noise)

    This model:
        - Maps predicted state to expected radar measurement
        - Enables correction step using actual detection

EKF Algorithm Per Time Step:

Prediction:
    X̂_{k|k-1} = F X̂_{k-1}
    P_{k|k-1} = F P_{k-1} F^T + Q

Linearization:
    H_k = ∂h/∂X evaluated at predicted state

Kalman Gain:
    K_k = P H^T (H P H^T + R)^{-1}

Update (if detection exists):
    X̂_k = X̂_{k|k-1} + K (z - h(X̂_{k|k-1}))
    P_k = (I - K H) P_{k|k-1}

If detection is missing:
    X̂_k = X̂_{k|k-1}
    P_k = P_{k|k-1}

Key Insight:
    - Motion model predicts where target should be.
    - Measurement model corrects prediction using radar data.
    - Missed detections cause covariance growth.
    - Stealth targets exhibit higher uncertainty due to fewer updates.

This module represents the radar tracking estimator.
It reconstructs target motion from noisy range–bearing measurements.
"""
import numpy as np
import matplotlib.pyplot as plt

from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater


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
# Detection selection (nearest neighbor gating)
# ============================================================

def select_best_detection(prediction, detection_list):

    if detection_list is None or len(detection_list) == 0:
        return None, None

    pred_x = prediction.state_vector[0, 0]
    pred_y = prediction.state_vector[2, 0]

    min_dist = np.inf
    best_detection = None

    for det in detection_list:
        x, y = polar_to_cartesian(det)
        dist = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)

        if dist < min_dist:
            min_dist = dist
            best_detection = det

    return best_detection, min_dist


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

        detection, dist = select_best_detection(prediction, detection_list)

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
    # Summary Statistics
    # ============================================================

    print("\n==== Tracking Summary ====")
    print(f"Total hits: {hit_counter}")
    print(f"Final miss streak: {miss_streak}")
    print(f"Final covariance trace: {cov_trace_history[-1]:.2f}")
    print(f"Mean RMS error: {np.mean(rms_error_history):.2f}")
    print(f"Mean innovation distance: {np.mean(innovation_norms):.2f}")

    # ============================================================
    # Unified Diagnostics Plot
    # ============================================================

    if plot:

        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle("Tracking Diagnostics", fontsize=14)

        # Scene
        axs[0, 0].plot(truth_x, truth_y, label="Truth")
        axs[0, 0].scatter(meas_x, meas_y, marker='x', label="Selected Detections")
        axs[0, 0].plot(est_x, est_y, linestyle='--', label="EKF Estimate")
        axs[0, 0].set_title("Tracking Scene")
        axs[0, 0].legend()

        # Innovation
        axs[0, 1].plot(innovation_norms)
        axs[0, 1].set_title("Innovation Distance")

        # Covariance
        axs[1, 0].plot(cov_trace_history)
        axs[1, 0].set_title("Covariance Trace")

        # RMS error
        axs[1, 1].plot(rms_error_history)
        axs[1, 1].set_title("RMS Position Error")

        # Hit ratio cumulative
        cumulative_hits = np.cumsum(
            [1 if x > 0 else 0 for x in innovation_norms]
        )
        axs[2, 0].plot(cumulative_hits)
        axs[2, 0].set_title("Cumulative Hits")

        # Miss indicator
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

