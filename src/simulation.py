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

    # Bias parameters (Phase 2)
    "bias_snr_threshold_db": 15,
    "range_bias_fraction": 0.03,
    "bearing_bias_deg": 0.5,

    # Clutter parameters (Phase 3)
    "clutter_birth_rate": 3,
    "clutter_velocity_std": 5.0,
    "max_range": 15000
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

    config = TARGET_CONFIG[scene_type]

    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(config["process_noise"]),
        ConstantVelocity(config["process_noise"])
    ])

    heading = np.random.uniform(-np.pi/6, np.pi/6)
    vx = config["speed"] * np.cos(heading)
    vy = config["speed"] * np.sin(heading)

    truth = GaussianState(
        StateVector([0, vx, 0, vy]),
        np.diag([1, 1, 1, 1]),
        timestamp=datetime.now()
    )

    truth_states = []
    detections = []

    # Diagnostics
    truth_x, truth_y = [], []
    meas_x, meas_y = [], []
    snr_history, pd_history, rcs_history = [], [], []
    detection_binary = []
    range_noise_std_history = []
    bearing_noise_std_history = []
    range_bias_history = []
    bearing_bias_history = []

    # Persistent clutter store
    persistent_clutter = []

    for _ in range(num_steps):

        truth_states.append(truth)

        x = truth.state_vector[0, 0]
        y = truth.state_vector[2, 0]

        truth_x.append(x)
        truth_y.append(y)

        R = np.sqrt(x**2 + y**2) + 1e-6

        # ---------------- Radar Equation ----------------
        rcs_fluct = np.random.exponential(scale=config["rcs"])
        received_power = (RADAR_CONFIG["Pt"] * rcs_fluct) / (R**4)
        SNR_linear = received_power / RADAR_CONFIG["noise_floor"]
        SNR_dB = 10 * np.log10(SNR_linear + 1e-12)
        Pd = 1 / (1 + np.exp(-0.6 * (SNR_dB - 10)))

        snr_history.append(SNR_dB)
        pd_history.append(Pd)
        rcs_history.append(rcs_fluct)

        step_detections = []

        # ============================================================
        # TRUE TARGET DETECTION (Phase 1 + 2)
        # ============================================================

        if np.random.rand() <= Pd:

            SNR_linear_safe = max(SNR_linear, 1e-6)

            base_range_std = RADAR_CONFIG["range_std_m"]
            base_bearing_std = np.radians(RADAR_CONFIG["bearing_std_deg"])

            range_std = base_range_std * np.sqrt(10 / SNR_linear_safe)
            bearing_std = base_bearing_std * np.sqrt(10 / SNR_linear_safe)

            range_noise_std_history.append(range_std)
            bearing_noise_std_history.append(np.degrees(bearing_std))

            true_range = R
            true_bearing = np.arctan2(y, x)

            # Phase 2 Bias
            if SNR_dB < RADAR_CONFIG["bias_snr_threshold_db"]:
                range_bias = RADAR_CONFIG["range_bias_fraction"] * true_range
                bearing_bias = np.radians(RADAR_CONFIG["bearing_bias_deg"])
            else:
                range_bias = 0
                bearing_bias = 0

            range_bias_history.append(range_bias)
            bearing_bias_history.append(np.degrees(bearing_bias))

            noisy_range = true_range + range_bias + np.random.normal(0, range_std)
            noisy_bearing = true_bearing + bearing_bias + np.random.normal(0, bearing_std)

            step_detections.append(
                Detection(
                    StateVector([noisy_bearing, noisy_range]),
                    timestamp=truth.timestamp,
                    measurement_model=measurement_model
                )
            )

            meas_x.append(noisy_range * np.cos(noisy_bearing))
            meas_y.append(noisy_range * np.sin(noisy_bearing))

            detection_binary.append(1)

        else:
            detection_binary.append(0)
            range_noise_std_history.append(0)
            bearing_noise_std_history.append(0)
            range_bias_history.append(0)
            bearing_bias_history.append(0)

        # ============================================================
        # BACKGROUND CLUTTER (Range-scaled)
        # ============================================================

        clutter_rate = 2 + 6 * (R / RADAR_CONFIG["max_range"])
        clutter_count = np.random.poisson(clutter_rate)

        for _ in range(clutter_count):
            cr = np.random.uniform(0, RADAR_CONFIG["max_range"])
            cb = np.random.uniform(-np.pi, np.pi)
            step_detections.append(
                Detection(
                    StateVector([cb, cr]),
                    timestamp=truth.timestamp,
                    measurement_model=measurement_model
                )
            )

        # ============================================================
        # LOCAL COMPETITION CLUTTER (Ambiguity)
        # ============================================================

        if np.random.rand() < 0.5:
            for _ in range(np.random.randint(1, 3)):
                local_range = R + np.random.normal(0, 200)
                local_bearing = np.arctan2(y, x) + np.random.normal(0, np.radians(2))
                step_detections.append(
                    Detection(
                        StateVector([local_bearing, local_range]),
                        timestamp=truth.timestamp,
                        measurement_model=measurement_model
                    )
                )

        # ============================================================
        # PERSISTENT CLUTTER (Drifting slow objects)
        # ============================================================

        # Birth
        for _ in range(np.random.poisson(RADAR_CONFIG["clutter_birth_rate"])):
            persistent_clutter.append({
                "x": np.random.uniform(-8000, 8000),
                "y": np.random.uniform(-8000, 8000),
                "vx": np.random.normal(0, RADAR_CONFIG["clutter_velocity_std"]),
                "vy": np.random.normal(0, RADAR_CONFIG["clutter_velocity_std"]),
                "life": np.random.randint(5, 10)
            })

        new_persistent = []

        for obj in persistent_clutter:
            obj["x"] += obj["vx"] * dt + np.random.normal(0, 2)
            obj["y"] += obj["vy"] * dt + np.random.normal(0, 2)
            obj["life"] -= 1

            r = np.sqrt(obj["x"]**2 + obj["y"]**2)
            theta = np.arctan2(obj["y"], obj["x"])

            step_detections.append(
                Detection(
                    StateVector([theta, r]),
                    timestamp=truth.timestamp,
                    measurement_model=measurement_model
                )
            )

            if obj["life"] > 0:
                new_persistent.append(obj)

        persistent_clutter = new_persistent

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

    # ============================================================
    # Diagnostics Plot
    # ============================================================

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

        axs[2, 1].plot(range_noise_std_history, label="Range Noise Std")
        axs[2, 1].plot(bearing_noise_std_history, label="Bearing Noise Std (deg)")
        axs[2, 1].plot(range_bias_history, linestyle='--', label="Range Bias")
        axs[2, 1].plot(bearing_bias_history, linestyle='--', label="Bearing Bias (deg)")
        axs[2, 1].set_title("SNR-Dependent Noise + Bias")
        axs[2, 1].legend()

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
    for target in ["aircraft", "stealth", "bird"]:
        print(f"\nRunning simulation for: {target}")
        simulate_scene(scene_type=target, plot=True, diagnostics=True)
