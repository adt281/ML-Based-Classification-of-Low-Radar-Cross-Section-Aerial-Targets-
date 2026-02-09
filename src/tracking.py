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


def run_tracker(truth_states,
                detections,
                transition_model,
                measurement_model,
                plot=False):

    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model)

    prior = GaussianState(
        StateVector([0, 18, 0, 12]),
        np.diag([100, 10, 100, 10]),
        timestamp=truth_states[0].timestamp
    )

    track = Track([prior])

    truth_x, truth_y = [], []
    meas_x, meas_y = [], []
    est_x, est_y = [], []

    for truth, detection in zip(truth_states, detections):

        truth_x.append(truth.state_vector[0, 0])
        truth_y.append(truth.state_vector[2, 0])

        prediction = predictor.predict(
            track[-1],
            timestamp=truth.timestamp
        )

        if detection is not None:
            hypothesis = SingleHypothesis(prediction, detection)
            posterior = updater.update(hypothesis)

            # Convert polar detection to Cartesian for plotting
            bearing = detection.state_vector[0, 0]
            range_ = detection.state_vector[1, 0]

            meas_x.append(range_ * np.cos(bearing))
            meas_y.append(range_ * np.sin(bearing))

        else:
            posterior = prediction

        track.append(posterior)

        est_x.append(posterior.state_vector[0, 0])
        est_y.append(posterior.state_vector[2, 0])

    if plot:
        plt.figure()
        plt.plot(truth_x, truth_y, label="Truth")
        plt.scatter(meas_x, meas_y, marker='x', label="Measurements")
        plt.plot(est_x, est_y, linestyle='--', label="EKF Estimate")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Tracking Debug")
        plt.legend()
        plt.grid(True)
        plt.show()

    return track


# --- Standalone Debug Mode ---
if __name__ == "__main__":

    from src.simulation import simulate_target

    truth_states, detections, transition_model, measurement_model = simulate_target(Pd=0.85)

    run_tracker(truth_states,
                detections,
                transition_model,
                measurement_model,
                plot=True)
