import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection


def simulate_target(target_type="aircraft",
                    num_steps=50,
                    dt=1.0,
                    plot=False):

    # ---------------- Target Class Configuration ----------------

    if target_type == "bird":
        speed = 20.0
        process_noise = 2.0        # higher maneuver variability
        Pd = 0.80

    elif target_type == "aircraft":
        speed = 250.0
        process_noise = 0.1       # very smooth motion
        Pd = 0.95

    elif target_type == "stealth":
        speed = 250.0
        process_noise = 0.1       # identical motion to aircraft
        Pd = 0.70                  # lower detection reliability

    else:
        raise ValueError("Invalid target_type")

    # ---------------- Motion Model ----------------

    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(process_noise),
        ConstantVelocity(process_noise)
    ])

    # ---------------- Radar Model ----------------

    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(1.0), 10.0])
    )

    # Slight random heading variation for realism
    heading_angle = np.random.uniform(-np.pi/6, np.pi/6)

    vx = speed * np.cos(heading_angle)
    vy = speed * np.sin(heading_angle)

    truth = GaussianState(
        StateVector([0, vx, 0, vy]),
        np.diag([1, 1, 1, 1]),
        timestamp=datetime.now()
    )

    truth_states = []
    detections = []

    truth_x, truth_y = [], []
    meas_x, meas_y = [], []

    # ---------------- Simulation Loop ----------------

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
        plt.plot(truth_x, truth_y, label="Truth")
        plt.scatter(meas_x, meas_y, marker='x', label="Measurements")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"Simulation Debug ({target_type})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return truth_states, detections, transition_model, measurement_model


if __name__ == "__main__":
    simulate_target(target_type="bird", plot=True)
