"""
High-Level Radar Tracking Simulation

State Vector:
    X = [x, vx, y, vy]^T

This module simulates a moving aerial target observed by a 2D radar.

System Components:
1. Motion Model (Constant Velocity):
    - Propagates the true state over time.
    - Adds Gaussian process noise to simulate maneuver uncertainty.
    - Generates evolving Cartesian state [x, vx, y, vy].


It generates these values using: 
    Xk+1 = FXk + wk

    F = constant velocity transition matrix
    F= 1 0 0 0 
       dt 1 0 0
       0 0 1 0 
       0 0 dt 1

    wk= Gaussian process noise

2. Radar Measurement Model:
    - Converts Cartesian position (x, y) to polar coordinates:
          r = sqrt(x^2 + y^2)
          θ = arctan(y / x)
    - Adds Gaussian measurement noise.
    
    - Produces radar detections (range, bearing).

3. Detection Probability (Pd):
    - Models missed detections.
    - Lower Pd simulates stealth behavior.

Target Classes:
    - Bird: low speed, higher maneuver noise.
    - Aircraft: high speed, smooth motion.
    - Stealth: same motion as aircraft, lower Pd.

Output:
    - Ground truth states over time
    - Radar detections (with missed detections)
    - Motion and measurement models for tracking stage

This module simulates the physical and sensor layer only.
Tracking and classification are handled separately.
"""

"""
High-Level Radar Tracking Simulation (Realistic Fighter-Class Radar)

State Vector:
    X = [x, vx, y, vy]^T

This module simulates aerial targets observed by a modern 2D tracking radar.

Motion Model:
    X_{k+1} = F X_k + w_k
    F = Constant velocity transition
    w_k = Gaussian process noise

Radar Measurement Model:
    z = h(X) + v
    h(X) = nonlinear Cartesian → Polar conversion
           r = sqrt(x^2 + y^2)
           θ = arctan2(y, x)
    v = Gaussian measurement noise

This module simulates:
    - Target kinematics
    - Radar detection behavior
    - Missed detections
Tracking and classification are handled separately.
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection


# ============================================================
# TARGET MODEL
# ============================================================

class Target:

    def __init__(self, target_type="aircraft"):

        if target_type == "bird":
            self.speed = 20.0
            self.process_noise = 2.0
            self.Pd = 0.80

        elif target_type == "aircraft":
            self.speed = 250.0
            self.process_noise = 0.1
            self.Pd = 0.95

        elif target_type == "stealth":
            self.speed = 250.0
            self.process_noise = 0.1
            self.Pd = 0.70

        else:
            raise ValueError("Invalid target_type")

        self.target_type = target_type


# ============================================================
# RADAR MODEL (No physics yet)
# ============================================================

class Radar:

    def __init__(self):

        self.bearing_std_deg = 1.0
        self.range_std_m = 10.0

        self.measurement_model = CartesianToBearingRange(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.diag([
                np.radians(self.bearing_std_deg) ** 2,
                self.range_std_m ** 2
            ])
        )

    def detect(self, truth_state, Pd):

        if np.random.rand() <= Pd:

            measurement_vector = self.measurement_model.function(
                truth_state,
                noise=True
            )

            detection = Detection(
                measurement_vector,
                timestamp=truth_state.timestamp,
                measurement_model=self.measurement_model
            )

            return detection

        return None


# ============================================================
# SIMULATION FUNCTION
# ============================================================

def simulate_target(target_type="aircraft",
                    num_steps=50,
                    dt=1.0,
                    plot=False):

    # ---------------- Initialize Target ----------------

    target = Target(target_type)

    # ---------------- Motion Model ----------------

    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(target.process_noise),
        ConstantVelocity(target.process_noise)
    ])

    # ---------------- Radar ----------------

    radar = Radar()

    # ---------------- Initial State ----------------

    heading_angle = np.random.uniform(-np.pi/12, np.pi/12)

    vx = target.speed * np.cos(heading_angle)
    vy = target.speed * np.sin(heading_angle)

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

        detection = radar.detect(truth, target.Pd)

        if detection is not None:

            bearing = detection.state_vector[0, 0]
            range_ = detection.state_vector[1, 0]

            meas_x.append(range_ * np.cos(bearing))
            meas_y.append(range_ * np.sin(bearing))

        detections.append(detection)

        # Propagate truth
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

    return truth_states, detections, transition_model, radar.measurement_model


# Standalone debug
if __name__ == "__main__":
    simulate_target(target_type="bird", plot=True)
