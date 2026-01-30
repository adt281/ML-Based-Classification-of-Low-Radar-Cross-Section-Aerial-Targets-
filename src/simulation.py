import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection


def simulate_target(Pd=0.85, num_steps=50, dt=1.0, plot=False):

    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(0.1),
        ConstantVelocity(0.1)
    ])

    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(1.0), 10.0])
    )

    truth = GaussianState(
        StateVector([0, 20, 0, 10]),
        np.diag([1, 1, 1, 1]),
        timestamp=datetime.now()
    )

    truth_states = []
    detections = []

    truth_x, truth_y = [], []
    meas_x, meas_y = [], []

    for _ in range(num_steps):

        truth_states.append(truth)

        truth_x.append(truth.state_vector[0, 0])
        truth_y.append(truth.state_vector[2, 0])

        if np.random.rand() <= Pd:

            measurement_vector = measurement_model.function(
                truth,
                noise=True
            )

            bearing = measurement_vector[0, 0]
            range_ = measurement_vector[1, 0]

            meas_x.append(range_ * np.cos(bearing))
            meas_y.append(range_ * np.sin(bearing))

            detection = Detection(
                measurement_vector,
                timestamp=truth.timestamp,
                measurement_model=measurement_model
            )

            detections.append(detection)

        else:
            detections.append(None)

        # propagate truth
        new_truth_vector = transition_model.function(
            truth,
            noise=True,
            time_interval=timedelta(seconds=dt)
        )

        truth = GaussianState(
            new_truth_vector,
            truth.covar,
            timestamp=truth.timestamp + timedelta(seconds=dt)
        )

    if plot:
        plt.figure()
        plt.plot(truth_x, truth_y, label="True Trajectory")
        plt.scatter(meas_x, meas_y, marker='x', label="Radar Measurements")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"Simulation Debug (Pd = {Pd})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return truth_states, detections, transition_model, measurement_model


# --- Allow standalone debugging ---
if __name__ == "__main__":
    simulate_target(Pd=0.85, plot=True)
