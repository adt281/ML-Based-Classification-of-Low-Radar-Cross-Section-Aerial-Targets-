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
    "clutter_birth_rate": 3,
    "clutter_velocity_std": 5.0,
}


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

    # Target diagnostics
    range_history = []
    snr_history = []
    pd_history = []
    rcs_history = []
    detection_binary = []

    # ============================================================
    # NOISE SCENE WITH PERSISTENT CLUTTER OBJECTS
    # ============================================================

    if scene_type == "noise":

        timestamp = datetime.now()

        active_clutter = []
        clutter_x_all = []
        clutter_y_all = []

        clutter_vel_history = []
        clutter_lifetime_history = []
        active_counts = []

        for _ in range(num_steps):

            # Birth new clutter objects
            for _ in range(np.random.poisson(RADAR_CONFIG["clutter_birth_rate"])):
                x = np.random.uniform(-8000, 8000)
                y = np.random.uniform(-8000, 8000)
                vx = np.random.normal(0, RADAR_CONFIG["clutter_velocity_std"])
                vy = np.random.normal(0, RADAR_CONFIG["clutter_velocity_std"])
                lifetime = np.random.randint(5, 11)

                active_clutter.append({
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "life": lifetime
                })

            step_detections = []
            new_active = []

            for obj in active_clutter:

                # Update position
                obj["x"] += obj["vx"] * dt + np.random.normal(0, 2)
                obj["y"] += obj["vy"] * dt + np.random.normal(0, 2)
                obj["life"] -= 1

                clutter_x_all.append(obj["x"])
                clutter_y_all.append(obj["y"])

                vel_mag = np.sqrt(obj["vx"]**2 + obj["vy"]**2)
                clutter_vel_history.append(vel_mag)
                clutter_lifetime_history.append(obj["life"])

                r = np.sqrt(obj["x"]**2 + obj["y"]**2)
                theta = np.arctan2(obj["y"], obj["x"])

                measurement_vector = StateVector([theta, r])

                step_detections.append(
                    Detection(
                        measurement_vector,
                        timestamp=timestamp,
                        measurement_model=measurement_model
                    )
                )

                if obj["life"] > 0:
                    new_active.append(obj)

            active_clutter = new_active
            active_counts.append(len(active_clutter))

            detections.append(step_detections if step_detections else None)
            timestamp += timedelta(seconds=dt)

        # ================= PLOTTING =================

        if plot or diagnostics:

            fig, axs = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle("Advanced Clutter Diagnostics", fontsize=14)

            # Spatial distribution
            axs[0, 0].scatter(clutter_x_all, clutter_y_all, s=10)
            axs[0, 0].set_title("Persistent Clutter Motion")

            # Velocity magnitude
            axs[0, 1].plot(clutter_vel_history)
            axs[0, 1].set_title("Clutter Velocity Magnitude")

            # Lifetime histogram
            axs[1, 0].hist(clutter_lifetime_history, bins=10)
            axs[1, 0].set_title("Clutter Lifetime Distribution")

            # Active clutter count
            axs[1, 1].plot(active_counts)
            axs[1, 1].set_title("Active Clutter Per Scan")

            # Density map
            heatmap, _, _ = np.histogram2d(
                clutter_x_all,
                clutter_y_all,
                bins=50
            )
            axs[2, 0].imshow(heatmap.T, origin='lower')
            axs[2, 0].set_title("Clutter Density Map")

            axs[2, 1].axis('off')

            for ax in axs.flat:
                ax.grid(True)

            plt.tight_layout()
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

    for _ in range(num_steps):

        truth_states.append(truth)

        x = truth.state_vector[0, 0]
        y = truth.state_vector[2, 0]

        truth_x.append(x)
        truth_y.append(y)

        R = np.sqrt(x**2 + y**2) + 1e-6

        rcs_fluct = np.random.exponential(scale=rcs_mean)
        received_power = (RADAR_CONFIG["Pt"] * rcs_fluct) / (R**4)
        SNR_linear = received_power / RADAR_CONFIG["noise_floor"]
        SNR_dB = 10 * np.log10(SNR_linear + 1e-12)

        Pd = 1 / (1 + np.exp(-0.6 * (SNR_dB - 10)))

        range_history.append(R)
        snr_history.append(SNR_dB)
        pd_history.append(Pd)
        rcs_history.append(rcs_fluct)

        detected = False
        step_detections = []

        if np.random.rand() <= Pd:

            detected = True

            measurement_vector = measurement_model.function(
                truth,
                noise=True
            )

            step_detections.append(
                Detection(
                    measurement_vector,
                    timestamp=truth.timestamp,
                    measurement_model=measurement_model
                )
            )

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

    # Target diagnostics
    if plot or diagnostics:

        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f"Radar Diagnostics - {scene_type}", fontsize=14)

        axs[0, 0].plot(truth_x, truth_y)
        axs[0, 0].scatter(meas_x, meas_y, marker='x')
        axs[0, 0].set_title("Target Scene")

        axs[0, 1].plot(rcs_history)
        axs[0, 1].set_title("RCS Fluctuation")

        axs[1, 0].plot(snr_history)
        axs[1, 0].set_title("SNR (dB)")

        axs[1, 1].plot(pd_history)
        axs[1, 1].set_title("Detection Probability")

        axs[2, 0].step(range(len(detection_binary)),
                       detection_binary,
                       where='mid')
        axs[2, 0].set_title("Detection Timeline")
        axs[2, 0].set_ylim(-0.1, 1.1)

        axs[2, 1].axis('off')

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
    simulate_scene(scene_type="noise", plot=True, diagnostics=True)
