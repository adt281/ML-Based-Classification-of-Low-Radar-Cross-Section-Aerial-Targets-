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
        "rcs": 0.01,
        "stage1": "credible",
        "stage2": "bird",
        "stage3": None
    },

    "aircraft": {
        "speed": 250.0,
        "process_noise": 0.1,
        "rcs": 5.0,
        "stage1": "credible",
        "stage2": "aircraft",
        "stage3": "strong"
    },

    "stealth": {
        "speed": 250.0,
        "process_noise": 0.1,
        "rcs": 0.5,
        "stage1": "credible",
        "stage2": "aircraft",
        "stage3": "weak"
    }
}


# ============================================================
# Radar Configuration
# ============================================================

RADAR_CONFIG = {
    "Pt": 1e9,
    "noise_floor": 1e-9,
    "bearing_std_deg": 1.6,
    "range_std_m": 20.0,
    "clutter_rate": 2.0
}


# ============================================================
# Main Scene Simulation
# ============================================================

def simulate_scene(scene_type="aircraft",
                   num_steps=80,
                   dt=1.0,
                   plot=False,
                   diagnostics=False):

    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([
            np.radians(RADAR_CONFIG["bearing_std_deg"])**2,
            RADAR_CONFIG["range_std_m"]**2
        ])
    )

    truth_states = []
    detections = []

    range_history = []
    snr_history = []
    pd_history = []
    rcs_history = []
    detection_binary = []

    # --------------------------------------------------------
    # NOISE ONLY SCENE
    # --------------------------------------------------------

    if scene_type == "noise":

        timestamp = datetime.now()
        clutter_x = []
        clutter_y = []

        for _ in range(num_steps):

            clutter_count = np.random.poisson(RADAR_CONFIG["clutter_rate"])
            step_detections = []

            for _ in range(clutter_count):

                bearing = np.random.uniform(-np.pi, np.pi)
                range_ = np.random.uniform(0, 10000)

                measurement_vector = StateVector([bearing, range_])

                detection = Detection(
                    measurement_vector,
                    timestamp=timestamp,
                    measurement_model=measurement_model
                )

                step_detections.append(detection)

                clutter_x.append(range_ * np.cos(bearing))
                clutter_y.append(range_ * np.sin(bearing))

            detections.append(step_detections if step_detections else None)
            timestamp += timedelta(seconds=dt)

        if plot:
            plt.figure()
            plt.scatter(clutter_x, clutter_y, s=10)
            plt.title("Noise / Clutter Only Scene")
            plt.xlabel("X Position (m)")
            plt.ylabel("Y Position (m)")
            plt.grid(True)
            plt.show()

        metadata = {
            "stage1": "non_credible",
            "stage2": None,
            "stage3": None
        }

        return truth_states, detections, None, measurement_model, metadata


    # --------------------------------------------------------
    # REAL TARGET SCENE
    # --------------------------------------------------------

    if scene_type not in TARGET_CONFIG:
        raise ValueError("Invalid scene_type")

    config = TARGET_CONFIG[scene_type]

    speed = config["speed"]
    process_noise = config["process_noise"]
    rcs_mean = config["rcs"]

    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(process_noise),
        ConstantVelocity(process_noise)
    ])

    heading_angle = np.random.uniform(-np.pi/6, np.pi/6)
    vx = speed * np.cos(heading_angle)
    vy = speed * np.sin(heading_angle)

    truth = GaussianState(
        StateVector([0, vx, 0, vy]),
        np.diag([1, 1, 1, 1]),
        timestamp=datetime.now()
    )

    truth_x, truth_y = [], []
    meas_x, meas_y = [], []

    for _ in range(num_steps):

        truth_states.append(truth)

        x = truth.state_vector[0, 0]
        y = truth.state_vector[2, 0]

        truth_x.append(x)
        truth_y.append(y)

        R = np.sqrt(x**2 + y**2) + 1e-6

        Pt = RADAR_CONFIG["Pt"]
        N0 = RADAR_CONFIG["noise_floor"]

        # 🔷 Swerling I RCS fluctuation
        rcs_fluct = np.random.exponential(scale=rcs_mean)

        received_power = (Pt * rcs_fluct) / (R**4)
        SNR_linear = received_power / N0
        SNR_dB = 10 * np.log10(SNR_linear + 1e-12)

        threshold_dB = 10
        steepness = 0.6

        Pd = 1.0 / (1.0 + np.exp(-steepness * (SNR_dB - threshold_dB)))

        range_history.append(R)
        snr_history.append(SNR_dB)
        pd_history.append(Pd)
        rcs_history.append(rcs_fluct)

        step_detections = []
        detected = False

        if np.random.rand() <= Pd:

            detected = True

            measurement_vector = measurement_model.function(
                truth,
                noise=True
            )

            detection = Detection(
                measurement_vector,
                timestamp=truth.timestamp,
                measurement_model=measurement_model
            )

            step_detections.append(detection)

            bearing = measurement_vector[0, 0]
            range_ = measurement_vector[1, 0]

            meas_x.append(range_ * np.cos(bearing))
            meas_y.append(range_ * np.sin(bearing))

        detection_binary.append(1 if detected else 0)

        # Clutter
        clutter_count = np.random.poisson(RADAR_CONFIG["clutter_rate"])

        for _ in range(clutter_count):
            bearing = np.random.uniform(-np.pi, np.pi)
            range_ = np.random.uniform(0, 10000)
            measurement_vector = StateVector([bearing, range_])
            detection = Detection(
                measurement_vector,
                timestamp=truth.timestamp,
                measurement_model=measurement_model
            )
            step_detections.append(detection)

        detections.append(step_detections if step_detections else None)

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

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------
    if plot or diagnostics:

        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f"Radar Diagnostics - {scene_type}", fontsize=14, y=1.02)

        # --- Scene ---
        axs[0, 0].plot(truth_x, truth_y, label="Truth")
        axs[0, 0].scatter(meas_x, meas_y, marker='x', label="Detections")
        axs[0, 0].set_title("Scene (Top View)", pad=10)
        axs[0, 0].set_xlabel("X Position (m)")
        axs[0, 0].set_ylabel("Y Position (m)")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # --- RCS ---
        axs[0, 1].plot(rcs_history)
        axs[0, 1].set_title("RCS Fluctuation", pad=10)
        axs[0, 1].set_xlabel("Time Step")
        axs[0, 1].set_ylabel("Instantaneous RCS")
        axs[0, 1].grid(True)

        # --- SNR ---
        axs[1, 0].plot(snr_history)
        axs[1, 0].set_title("SNR (dB)", pad=10)
        axs[1, 0].set_xlabel("Time Step")
        axs[1, 0].set_ylabel("SNR (dB)")
        axs[1, 0].grid(True)

        # --- Detection Probability ---
        axs[1, 1].plot(pd_history)
        axs[1, 1].set_title("Detection Probability", pad=10)
        axs[1, 1].set_xlabel("Time Step")
        axs[1, 1].set_ylabel("Pd")
        axs[1, 1].grid(True)

        # --- Detection Timeline ---
        axs[2, 0].step(range(len(detection_binary)), detection_binary, where='mid')
        axs[2, 0].set_title("Detection Occurrence (1=Detected)", pad=10)
        axs[2, 0].set_xlabel("Time Step")
        axs[2, 0].set_ylabel("Detection")
        axs[2, 0].set_ylim(-0.1, 1.1)
        axs[2, 0].grid(True)

        # Hide unused subplot
        axs[2, 1].axis('off')

        # Leave room for suptitle
        # Increase spacing ONLY
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        plt.show()


    metadata = {
        "stage1": config["stage1"],
        "stage2": config["stage2"],
        "stage3": config["stage3"]
    }

    return truth_states, detections, transition_model, measurement_model, metadata


if __name__ == "__main__":

    simulate_scene(
        scene_type="stealth",
        plot=True,
        diagnostics=True
    )
