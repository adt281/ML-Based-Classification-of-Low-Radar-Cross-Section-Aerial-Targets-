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

    # Clutter realism
    "clutter_clusters": 3,
    "clutter_spread_m": 800,
    "clutter_drift_std": 80,

    "base_clutter_rate": 5,
    "ground_clutter_boost_range": 2500,
    "ground_clutter_multiplier": 2.5,

    # Heavy-tailed amplitude (gamma parameters)
    "clutter_shape_k": 0.8,
    "clutter_scale_theta": 1.0
}


# ============================================================
# Helper: Heavy-Tailed Clutter Power
# ============================================================

def sample_clutter_amplitude():
    return np.random.gamma(
        RADAR_CONFIG["clutter_shape_k"],
        RADAR_CONFIG["clutter_scale_theta"]
    )


# ============================================================
# Helper: Generate Clustered Clutter
# ============================================================

def generate_clustered_clutter(timestamp,
                               measurement_model,
                               cluster_centers,
                               clutter_spread,
                               num_points):

    detections = []
    clutter_x = []
    clutter_y = []

    for _ in range(num_points):

        cx, cy = cluster_centers[np.random.randint(len(cluster_centers))]

        x = np.random.normal(cx, clutter_spread)
        y = np.random.normal(cy, clutter_spread)

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # heavy-tailed amplitude
        amplitude = sample_clutter_amplitude()

        # only create detection if amplitude exceeds minimal threshold
        if amplitude > 0.2:

            measurement_vector = StateVector([theta, r])

            detection = Detection(
                measurement_vector,
                timestamp=timestamp,
                measurement_model=measurement_model
            )

            detections.append(detection)
            clutter_x.append(x)
            clutter_y.append(y)

    return detections, clutter_x, clutter_y


# ============================================================
# Main Simulation
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

    # ============================================================
    # NOISE SCENE (MULTI-LAYER CLUTTER)
    # ============================================================

    if scene_type == "noise":

        timestamp = datetime.now()

        cluster_centers = [
            (np.random.uniform(-8000, 8000),
             np.random.uniform(-8000, 8000))
            for _ in range(RADAR_CONFIG["clutter_clusters"])
        ]

        clutter_x_all = []
        clutter_y_all = []

        persistent_points = []

        for _ in range(num_steps):

            # drift clusters (weather motion)
            new_centers = []
            for cx, cy in cluster_centers:
                cx += np.random.normal(0, RADAR_CONFIG["clutter_drift_std"])
                cy += np.random.normal(0, RADAR_CONFIG["clutter_drift_std"])
                new_centers.append((cx, cy))
            cluster_centers = new_centers

            # base clutter count
            clutter_count = np.random.poisson(
                RADAR_CONFIG["base_clutter_rate"]
            )

            # ground clutter boost near radar
            if np.random.rand() < 0.5:
                clutter_count *= RADAR_CONFIG["ground_clutter_multiplier"]

            clutter_count = int(clutter_count)

            step_detections, cx_list, cy_list = generate_clustered_clutter(
                timestamp,
                measurement_model,
                cluster_centers,
                RADAR_CONFIG["clutter_spread_m"],
                clutter_count
            )

            # persistence (some clutter survives to next scan)
            persistent_points = cx_list[:5]
            clutter_x_all.extend(cx_list)
            clutter_y_all.extend(cy_list)

            detections.append(step_detections if step_detections else None)
            timestamp += timedelta(seconds=dt)

        if plot:

            plt.figure(figsize=(8, 6))
            plt.scatter(clutter_x_all, clutter_y_all, s=8)
            plt.title("Multi-Layer Realistic Clutter Scene")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.grid(True)
            plt.show()

        metadata = {
            "stage1": "non_credible",
            "stage2": None,
            "stage3": None
        }

        return truth_states, detections, None, measurement_model, metadata


    # ============================================================
    # REAL TARGET SCENE
    # ============================================================

    config = TARGET_CONFIG[scene_type]

    speed = config["speed"]
    process_noise = config["process_noise"]
    rcs_mean = config["rcs"]

    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(process_noise),
        ConstantVelocity(process_noise)
    ])

    heading = np.random.uniform(-np.pi/6, np.pi/6)
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    truth = GaussianState(
        StateVector([0, vx, 0, vy]),
        np.diag([1, 1, 1, 1]),
        timestamp=datetime.now()
    )

    truth_x, truth_y = [], []
    meas_x, meas_y = [], []
    detection_binary = []
    snr_history = []

    for _ in range(num_steps):

        truth_states.append(truth)

        x = truth.state_vector[0, 0]
        y = truth.state_vector[2, 0]

        truth_x.append(x)
        truth_y.append(y)

        R = np.sqrt(x**2 + y**2) + 1e-6

        Pt = RADAR_CONFIG["Pt"]
        N0 = RADAR_CONFIG["noise_floor"]

        rcs_fluct = np.random.exponential(scale=rcs_mean)

        received_power = (Pt * rcs_fluct) / (R**4)
        SNR_linear = received_power / N0
        SNR_dB = 10 * np.log10(SNR_linear + 1e-12)

        Pd = 1 / (1 + np.exp(-0.6 * (SNR_dB - 10)))

        snr_history.append(SNR_dB)

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
            r_meas = measurement_vector[1, 0]

            meas_x.append(r_meas * np.cos(bearing))
            meas_y.append(r_meas * np.sin(bearing))

        detection_binary.append(1 if detected else 0)

        detections.append(step_detections if step_detections else None)

        truth = GaussianState(
            transition_model.function(
                truth,
                noise=True,
                time_interval=timedelta(seconds=dt)
            ),
            truth.covar,
            timestamp=truth.timestamp + timedelta(seconds=dt)
        )

    if plot:

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(truth_x, truth_y)
        axs[0, 0].scatter(meas_x, meas_y, marker='x')
        axs[0, 0].set_title("Scene")

        axs[0, 1].plot(snr_history)
        axs[0, 1].set_title("SNR")

        axs[1, 0].step(range(len(detection_binary)), detection_binary)
        axs[1, 0].set_title("Detection Timeline")

        axs[1, 1].axis('off')

        for ax in axs.flat:
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    metadata = {
        "stage1": config["stage1"],
        "stage2": config["stage2"],
        "stage3": config["stage3"]
    }

    return truth_states, detections, transition_model, measurement_model, metadata


if __name__ == "__main__":

    simulate_scene(
        scene_type="noise",
        plot=True
    )
