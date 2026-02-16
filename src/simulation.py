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
    "bird": {"speed": 20.0, "process_noise": 2.0, "rcs": 0.01},
    "aircraft": {"speed": 250.0, "process_noise": 0.1, "rcs": 5.0},
    "stealth": {"speed": 250.0, "process_noise": 0.1, "rcs": 0.5},
}

MANEUVER_CONFIG = {
    "aircraft": {
        "maneuver_prob": 0.05,
        "maneuver_min_deg": 3,
        "maneuver_max_deg": 6,
        "maneuver_duration_min": 8,
        "maneuver_duration_max": 12
    },
    "stealth": {
        "maneuver_prob": 0.04,
        "maneuver_min_deg": 4,
        "maneuver_max_deg": 8,
        "maneuver_duration_min": 6,
        "maneuver_duration_max": 10
    },
    "bird": {
        "maneuver_prob": 0.15,
        "maneuver_min_deg": 20,
        "maneuver_max_deg": 60,
        "maneuver_duration_min": 1,
        "maneuver_duration_max": 3
    }
}


# ============================================================
# Radar Configuration
# ============================================================

RADAR_CONFIG = {
    "Pt": 1e9,
    "noise_floor": 1e-9,
    "detection_threshold_db": 13,
    "bearing_std_deg": 1.6,
    "range_std_m": 20.0,
    "max_range": 15000,
    "cell_range_m": 300,
    "cell_bearing_deg": 3
}


# ============================================================
# Main Simulation
# ============================================================

def simulate_scene(scene_type="aircraft",
                   num_steps=80,
                   dt=1.0,
                   plot=False):

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

    truth_x, truth_y = [], []
    clutter_x_all, clutter_y_all = [], []
    detect_x_all, detect_y_all = [], []
    snr_history = []
    clutter_count_history = []


    maneuver_steps_remaining = 0
    current_turn_rate = 0.0  # rad/sec


    for step in range(num_steps):

        truth_states.append(truth)

        x = truth.state_vector[0, 0]
        y = truth.state_vector[2, 0]
        R = np.sqrt(x**2 + y**2) + 1e-6

        truth_x.append(x)
        truth_y.append(y)

        step_returns = []

        # ============================================================
        # TARGET RETURN
        # ============================================================

        rcs_fluct = np.random.exponential(scale=config["rcs"])
        target_power = (RADAR_CONFIG["Pt"] * rcs_fluct) / (R**4)
        target_snr = target_power / RADAR_CONFIG["noise_floor"]
        target_snr_db = 10 * np.log10(target_snr + 1e-12)

        snr_history.append(target_snr_db)

        step_returns.append({
            "range": R,
            "bearing": np.arctan2(y, x),
            "snr_db": target_snr_db
        })

        # ============================================================
        # CLUTTER RETURNS
        # ============================================================

        clutter_rate = 2 + 6 * (R / RADAR_CONFIG["max_range"])
        clutter_count = np.random.poisson(clutter_rate)
        clutter_count_history.append(clutter_count)

        for _ in range(clutter_count):

            cr = np.random.uniform(0, RADAR_CONFIG["max_range"])
            cb = np.random.uniform(-np.pi, np.pi)

            clutter_rcs = np.random.exponential(scale=1.0)
            clutter_power = (RADAR_CONFIG["Pt"] * clutter_rcs) / (cr**4 + 1e-6)
            clutter_snr = clutter_power / RADAR_CONFIG["noise_floor"]
            clutter_snr_db = 10 * np.log10(clutter_snr + 1e-12)

            clutter_x_all.append(cr * np.cos(cb))
            clutter_y_all.append(cr * np.sin(cb))

            step_returns.append({
                "range": cr,
                "bearing": cb,
                "snr_db": clutter_snr_db
            })

        # ============================================================
        # THRESHOLDING
        # ============================================================

        valid_returns = [
            r for r in step_returns
            if r["snr_db"] >= RADAR_CONFIG["detection_threshold_db"]
        ]

        # ============================================================
        # LOCAL COMPETITION
        # ============================================================

        survivors = []
        used = np.zeros(len(valid_returns), dtype=bool)

        cell_range = RADAR_CONFIG["cell_range_m"]
        cell_bearing = np.radians(RADAR_CONFIG["cell_bearing_deg"])

        for i, r in enumerate(valid_returns):
            if used[i]:
                continue

            competitors = [i]

            for j in range(i+1, len(valid_returns)):
                if used[j]:
                    continue

                dr = abs(valid_returns[j]["range"] - r["range"])
                db = abs(valid_returns[j]["bearing"] - r["bearing"])

                if dr < cell_range and db < cell_bearing:
                    competitors.append(j)

            best_idx = max(competitors,
                           key=lambda k: valid_returns[k]["snr_db"])

            for k in competitors:
                used[k] = True

            survivors.append(valid_returns[best_idx])

        # ============================================================
        # MEASUREMENT GENERATION (NO TYPE LEAKAGE)
        # ============================================================

        step_detections = []

        for ret in survivors:

            snr_linear = 10 ** (ret["snr_db"] / 10)

            range_std = RADAR_CONFIG["range_std_m"] * \
                        np.sqrt(10 / max(snr_linear, 1e-6))
            bearing_std = np.radians(RADAR_CONFIG["bearing_std_deg"]) * \
                          np.sqrt(10 / max(snr_linear, 1e-6))

            noisy_range = ret["range"] + np.random.normal(0, range_std)
            noisy_bearing = ret["bearing"] + np.random.normal(0, bearing_std)

            detect_x_all.append(noisy_range * np.cos(noisy_bearing))
            detect_y_all.append(noisy_range * np.sin(noisy_bearing))

            step_detections.append(
                Detection(
                    StateVector([noisy_bearing, noisy_range]),
                    timestamp=truth.timestamp,
                    measurement_model=measurement_model,
                    metadata={"snr_db": ret["snr_db"]}
                )
            )

        detections.append(step_detections if step_detections else None)

        # ============================================================
        # Coordinated Turn Truth Propagation
        # ============================================================

        # Extract state
        x = truth.state_vector[0, 0]
        vx = truth.state_vector[1, 0]
        y = truth.state_vector[2, 0]
        vy = truth.state_vector[3, 0]

        speed = np.sqrt(vx**2 + vy**2)

        # Decide if new maneuver starts
        
        # ============================================================
        # Maneuver Configuration Per Target Type
        # ============================================================

        maneuver_cfg = MANEUVER_CONFIG[scene_type]

        # Extract current state
        x = truth.state_vector[0, 0]
        vx = truth.state_vector[1, 0]
        y = truth.state_vector[2, 0]
        vy = truth.state_vector[3, 0]

        speed = np.sqrt(vx**2 + vy**2)
        heading = np.arctan2(vy, vx)

        # ------------------------------------------------------------
        # Decide if new maneuver starts
        # ------------------------------------------------------------

        if maneuver_steps_remaining == 0:
            if np.random.rand() < maneuver_cfg["maneuver_prob"]:
                maneuver_steps_remaining = np.random.randint(
                    maneuver_cfg["maneuver_duration_min"],
                    maneuver_cfg["maneuver_duration_max"]
                )

                turn_deg = np.random.uniform(
                    maneuver_cfg["maneuver_min_deg"],
                    maneuver_cfg["maneuver_max_deg"]
                )

                turn_sign = np.random.choice([-1, 1])
                current_turn_rate = np.radians(turn_deg) * turn_sign

        # ------------------------------------------------------------
        # Apply maneuver if active
        # ------------------------------------------------------------

        if maneuver_steps_remaining > 0:
            heading += current_turn_rate * dt
            maneuver_steps_remaining -= 1

        # ------------------------------------------------------------
        # Bird-specific speed fluctuation (biological realism)
        # ------------------------------------------------------------

        if scene_type == "bird":
            speed += np.random.normal(0, 2.0)
            speed = max(speed, 5.0)  # prevent negative/near-zero speed

        # ------------------------------------------------------------
        # Recompute velocity from updated heading & speed
        # ------------------------------------------------------------

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        # Small process noise (environmental disturbance)
        vx += np.random.normal(0, config["process_noise"])
        vy += np.random.normal(0, config["process_noise"])

        # Position update
        x += vx * dt
        y += vy * dt

        # Update truth state
        truth = GaussianState(
            StateVector([x, vx, y, vy]),
            truth.covar,
            timestamp=truth.timestamp + timedelta(seconds=dt)
        )



    # ============================================================
    # PLOTTING
    # ============================================================

    if plot:

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Radar Diagnostics - {scene_type}", fontsize=14)

        axs[0, 0].plot(truth_x, truth_y, linewidth=2, label="Truth")
        axs[0, 0].scatter(clutter_x_all, clutter_y_all,
                          s=6, alpha=0.15, label="Clutter")
        axs[0, 0].scatter(detect_x_all, detect_y_all,
                          s=20, marker='x', label="All Detections")
        axs[0, 0].legend()
        axs[0, 0].set_title("Full Scene (Anonymous Detections)")

        axs[0, 1].plot(snr_history)
        axs[0, 1].set_title("Target SNR (dB)")

        axs[1, 0].plot(clutter_count_history)
        axs[1, 0].set_title("Clutter Count Per Scan")

        heatmap, _, _ = np.histogram2d(clutter_x_all,
                                       clutter_y_all,
                                       bins=50)
        axs[1, 1].imshow(heatmap.T, origin='lower')
        axs[1, 1].set_title("Clutter Density Map")

        for ax in axs.flat:
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return truth_states, detections, transition_model, measurement_model, {}
    

if __name__ == "__main__":
    for target in ["aircraft", "stealth", "bird"]:
        print(f"\nRunning simulation for: {target}")
        simulate_scene(scene_type=target, plot=True)
