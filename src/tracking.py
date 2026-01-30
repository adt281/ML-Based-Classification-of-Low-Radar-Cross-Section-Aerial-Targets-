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
