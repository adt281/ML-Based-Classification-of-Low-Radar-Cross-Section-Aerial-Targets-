import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection


# ============================================================
# Target Configuration
# ============================================================

TARGET_CONFIG = {
    "bird": {"speed": 50, "process_noise": 2.0, "rcs": 0.01},
    "aircraft": {"speed": 250.0, "process_noise": 0.1, "rcs": 5.0},
    "stealth": {"speed": 250.0, "process_noise": 0.1, "rcs": 0.09},
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
        "maneuver_prob": 0.4,
        "maneuver_min_deg": 5,
        "maneuver_max_deg": 25,
        "maneuver_duration_min": 1,
        "maneuver_duration_max": 4
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
    "max_range": 15000,

    "num_range_bins": 200,
    "num_bearing_bins": 180,

    # CFAR
    "cfar_training": 8,
    "cfar_guard": 3,
    "cfar_pfa": 1e-4
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

    max_range = RADAR_CONFIG["max_range"]
    Nr = RADAR_CONFIG["num_range_bins"]
    Nb = RADAR_CONFIG["num_bearing_bins"]

    range_bins = np.linspace(0, max_range, Nr)
    bearing_bins = np.linspace(-np.pi, np.pi, Nb)

    for step in range(num_steps):

        truth_states.append(truth)

        x = truth.state_vector[0, 0]
        y = truth.state_vector[2, 0]
        vx = truth.state_vector[1, 0]
        vy = truth.state_vector[3, 0]

        R = np.sqrt(x**2 + y**2) + 1e-6
        B = np.arctan2(y, x)

        truth_x.append(x)
        truth_y.append(y)

        power_map = np.ones((Nr, Nb)) * RADAR_CONFIG["noise_floor"]

        # ============================================================
        # TARGET
        # ============================================================

        rcs_fluct = np.random.exponential(scale=config["rcs"])
        target_power = (RADAR_CONFIG["Pt"] * rcs_fluct) / (R**4)

        snr = target_power / RADAR_CONFIG["noise_floor"]
        snr_history.append(10 * np.log10(snr + 1e-12))

        r_idx = np.argmin(np.abs(range_bins - R))
        b_idx = np.argmin(np.abs(bearing_bins - B))

        power_map[r_idx, b_idx] += target_power

        # ============================================================
        # CLUTTER
        # ============================================================

        clutter_rate = 2 + 10 * (R / max_range)**2
        clutter_count = np.random.poisson(clutter_rate)
        clutter_count_history.append(clutter_count)

        # ============================================================
        # CLUTTER - realistic patches + speckle
        # ============================================================

        # base speckle (exponential / Rayleigh-like) — stochastic noise floor
        noise_map = np.random.exponential(scale=RADAR_CONFIG["noise_floor"], size=(Nr, Nb))
        power_map += noise_map

        # add a few patchy clutter regions (2D Gaussian blobs in range-bearing bins)
        num_patches = np.random.randint(4, 10)  # how many clutter patches this scan
        for _p in range(num_patches):
            # center in physical coordinates
            cr = max_range * np.sqrt(np.random.rand())         # range (area-consistent)
            cb = np.random.vonmises(0, 0.0) if False else np.random.uniform(-np.pi, np.pi)
            # amplitude: heavier tail but capped
            amp = np.random.gamma(shape=1.5, scale=0.6)  # modifies local RCS of patch

            # convert to bin centers
            r_c = np.argmin(np.abs(range_bins - cr))
            b_c = np.argmin(np.abs(bearing_bins - cb))

            # patch widths in bins (scale with range for realism)
            sigma_r = max(2, int(5 + (r_c / Nr) * 10))      # wider patches at larger ranges
            sigma_b = max(1, int(2 + (b_c / Nb) * 4))

            # create local 2D Gaussian bump and add it
            r_indices = np.arange(max(0, r_c - 4*sigma_r), min(Nr, r_c + 4*sigma_r))
            b_indices = np.arange(max(0, b_c - 4*sigma_b), min(Nb, b_c + 4*sigma_b))

            rr, bb = np.meshgrid(r_indices, b_indices, indexing='ij')
            dr = (rr - r_c) / max(1.0, sigma_r)
            db = (bb - b_c) / max(1.0, sigma_b)
            bump = amp * np.exp(-0.5 * (dr**2 + db**2))

            # scale bump by inverse-range^4 effect at patch center (approx)
            patch_range = range_bins[r_c] + 1e-6
            bump *= (RADAR_CONFIG["Pt"] / (patch_range**4))

            power_map[r_indices[:, None], b_indices[None, :]] += bump

            # store some example clutter scatter points for plotting
            # sample a few points from the bump to keep scatter realistic
            flat_idx = np.argwhere(bump > 0)
            if flat_idx.size:
                samples = flat_idx[np.random.choice(flat_idx.shape[0],
                                                    size=min(4, flat_idx.shape[0]),
                                                    replace=False)]
                for s in samples:
                    rr_s = int(rr[s[0], s[1]])
                    bb_s = int(bb[s[0], s[1]])
                    cr_phys = range_bins[rr_s]
                    cb_phys = bearing_bins[bb_s]
                    clutter_x_all.append(cr_phys * np.cos(cb_phys))
                    clutter_y_all.append(cr_phys * np.sin(cb_phys))

        # Add a small number of independent point-clutter hits (like spiky birds)
        point_clutter_count = np.random.poisson(3)
        for _ in range(point_clutter_count):
            cr = max_range * np.sqrt(np.random.rand())
            cb = np.random.uniform(-np.pi, np.pi)
            clutter_rcs = np.random.gamma(shape=1.0, scale=0.4)
            clutter_power = (RADAR_CONFIG["Pt"] * clutter_rcs) / (cr**4 + 1e-6)
            r_i = np.argmin(np.abs(range_bins - cr))
            b_i = np.argmin(np.abs(bearing_bins - cb))
            power_map[r_i, b_i] += clutter_power
            clutter_x_all.append(cr * np.cos(cb))
            clutter_y_all.append(cr * np.sin(cb))

        # ============================================================
        # 2D CA-CFAR
        # ============================================================

        T = RADAR_CONFIG["cfar_training"]
        G = RADAR_CONFIG["cfar_guard"]
        Pfa = RADAR_CONFIG["cfar_pfa"]

        total_window = ((2*(T+G)+1)**2)
        guard_window = (2*G+1)**2
        N = total_window - guard_window

        alpha = N * (Pfa**(-1/N) - 1)

        detected_cells = []

        for i in range(T+G, Nr-T-G):
            for j in range(T+G, Nb-T-G):

                cut_power = power_map[i, j]

                window = power_map[i-T-G:i+T+G+1,
                                   j-T-G:j+T+G+1]

                guard = power_map[i-G:i+G+1,
                                  j-G:j+G+1]

                training_cells = window.flatten()

                #*** OS-CFAR Implementation !! *** 
                # remove guard cells
                guard_mask = np.zeros_like(window, dtype=bool)
                guard_mask[G:-G or None, G:-G or None] = True
                training_cells = training_cells[~guard_mask.flatten()]

                # Flatten window and remove guard cells
                window_flat = window.flatten()
                # Build guard mask (same shape as window)
                gmask = np.zeros_like(window, dtype=bool)
                # mark guard cells True inside center (guard block)
                gmask[G:-G or None, G:-G or None] = True
                training_flat = window_flat[~gmask.flatten()]

                # If too few training cells skip
                if training_flat.size < max(3, int(0.5 * N)):
                    continue

                # Ordered-statistic CFAR: pick k-th order statistic as noise estimate
                k = int(0.75 * training_flat.size)  # 75th percentile cell
                sorted_cells = np.sort(training_flat)
                noise_est_os = sorted_cells[min(k, len(sorted_cells)-1)]

                # Use existing alpha (note: alpha formula assumes CA-CFAR; OS-CFAR will require tuning)
                threshold = alpha * noise_est_os

                if cut_power > threshold:
                    detected_cells.append((i, j))

        # ============================================================
        # LOCAL PEAK FILTER (stable clustering)
        # ============================================================

        filtered = []

        for (i, j) in detected_cells:

            local = power_map[i-1:i+2, j-1:j+2]
            if power_map[i, j] == np.max(local):
                filtered.append((i, j))

        detected_cells = filtered

        # ============================================================
        # MEASUREMENT GENERATION
        # ============================================================

        step_detections = []

        target_vr = (x*vx + y*vy) / (R + 1e-6)

        for i, j in detected_cells:

            det_range = range_bins[i]
            det_bearing = bearing_bins[j]

            snr_linear = power_map[i, j] / RADAR_CONFIG["noise_floor"]
            snr_db = 10*np.log10(snr_linear + 1e-12)

            if snr_db < 10:
                continue

            dx = det_range * np.cos(det_bearing)
            dy = det_range * np.sin(det_bearing)

            vr_est = (dx*vx + dy*vy) / (det_range + 1e-6)

            if abs(vr_est - target_vr) > 50:
                continue

            range_std = RADAR_CONFIG["range_std_m"] * \
                        np.sqrt(10 / max(snr_linear, 1e-6))
            bearing_std = np.radians(RADAR_CONFIG["bearing_std_deg"]) * \
                          np.sqrt(10 / max(snr_linear, 1e-6))

            noisy_range = det_range + np.random.normal(0, range_std)
            noisy_bearing = det_bearing + np.random.normal(0, bearing_std)

            detect_x_all.append(noisy_range * np.cos(noisy_bearing))
            detect_y_all.append(noisy_range * np.sin(noisy_bearing))

            step_detections.append(
                Detection(
                    StateVector([noisy_bearing, noisy_range]),
                    timestamp=truth.timestamp,
                    measurement_model=measurement_model
                )
            )

        detections.append(step_detections if step_detections else None)

        # ============================================================
        # MOTION PROPAGATION (unchanged)
        # ============================================================

        speed = np.sqrt(vx**2 + vy**2)
        heading = np.arctan2(vy, vx)

        maneuver_cfg = MANEUVER_CONFIG[scene_type]

        if maneuver_steps_remaining == 0:
            if np.random.rand() < maneuver_cfg["maneuver_prob"]:
                maneuver_steps_remaining = np.random.randint(
                    maneuver_cfg["maneuver_duration_min"],
                    maneuver_cfg["maneuver_duration_max"]
                )

        if maneuver_steps_remaining > 0:
            heading += np.radians(np.random.uniform(-20, 20))
            maneuver_steps_remaining -= 1

        if scene_type == "bird":
            speed += np.random.normal(0, 2.0)
            speed = max(speed, 5.0)

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        vx += np.random.normal(0, config["process_noise"])
        vy += np.random.normal(0, config["process_noise"])

        x += vx * dt
        y += vy * dt

        truth = GaussianState(
            StateVector([x, vx, y, vy]),
            truth.covar,
            timestamp=truth.timestamp + timedelta(seconds=dt)
        )

    # ============================================================
    # PLOTTING (UNCHANGED)
    # ============================================================

    if plot:

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Radar Diagnostics - {scene_type}", fontsize=14)

        if scene_type == "bird":
            margin = 5000
            axs[0, 0].set_xlim(min(truth_x)-margin, max(truth_x)+margin)
            axs[0, 0].set_ylim(min(truth_y)-margin, max(truth_y)+margin)

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

    return truth_states, detections, None, measurement_model, {}


if __name__ == "__main__":
    for target in ["aircraft", "stealth", "bird"]:
        print(f"\nRunning simulation for: {target}")
        simulate_scene(scene_type=target, plot=True)
