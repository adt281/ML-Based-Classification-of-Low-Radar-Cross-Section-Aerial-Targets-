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
# Target Configuration
# ============================================================

TARGET_CONFIG = {

    "bird": {
        "speed": 20.0,
        "process_noise": 2.0,
        "rcs": 0.01      # m²
    },

    "aircraft": {
        "speed": 250.0,
        "process_noise": 0.1,
        "rcs": 5.0      # m²
    },

    "stealth": {
        "speed": 250.0,
        "process_noise": 0.1,
        "rcs": 0.5       # m²
    }
}


# ============================================================
# Radar Configuration
# ============================================================

RADAR_CONFIG = {
    "Pt": 1e4,
    "noise_floor": 1e-6,
    "bearing_std_deg": 0.8,
    "range_std_m": 20.0
}



# ============================================================
# Simulation
# ============================================================

def simulate_target(target_type="aircraft",
                    num_steps=80,
                    dt=1.0,
                    plot=False):

    if target_type not in TARGET_CONFIG:
        raise ValueError("Invalid target_type")

    config = TARGET_CONFIG[target_type]

    speed = config["speed"]
    process_noise = config["process_noise"]
    rcs = config["rcs"]

    # ---------------- Motion Model ----------------

    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(process_noise),
        ConstantVelocity(process_noise)
    ])

    # ---------------- Radar Model ----------------

    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([
            np.radians(RADAR_CONFIG["bearing_std_deg"])**2,
            RADAR_CONFIG["range_std_m"]**2
        ])
    )

    # Random heading
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

        x = truth.state_vector[0, 0]
        y = truth.state_vector[2, 0]

        truth_x.append(x)
        truth_y.append(y)

        # ---------------- Radar Physics ----------------

        R = np.sqrt(x**2 + y**2) + 1e-6

        Pt = RADAR_CONFIG["Pt"]
        N0 = RADAR_CONFIG["noise_floor"]

        # Radar equation (simplified)
        received_power = (Pt * rcs) / (R**4)

        SNR_linear = received_power / N0
        SNR_dB = 10 * np.log10(SNR_linear + 1e-12)

        # Detection threshold around 13 dB typical
        threshold_dB = 10
        steepness = 0.6


        Pd = 1.0 / (1.0 + np.exp(-steepness * (SNR_dB - threshold_dB)))

        # ---------------- Detection Decision ----------------

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
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title(f"Radar Simulation ({target_type})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return truth_states, detections, transition_model, measurement_model
