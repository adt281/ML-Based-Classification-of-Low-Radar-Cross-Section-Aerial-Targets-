import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity
)
from stonesoup.types.hypothesis import SingleHypothesis

from simulation import simulate_scene


class SingleTargetTracker:

    def __init__(self, dt, measurement_model, process_noise_scale=1.0):

        self.dt = dt
        self.measurement_model = measurement_model

        # ---------------------------
        # CV transition model
        # ---------------------------
        self.transition_model = CombinedLinearGaussianTransitionModel([
            ConstantVelocity(process_noise_scale),
            ConstantVelocity(process_noise_scale)
        ])

        self.predictor = ExtendedKalmanPredictor(self.transition_model)
        self.updater = ExtendedKalmanUpdater(measurement_model)

        self.state = None
        self.initialized = False
        self.status = "tentative"

        self.consecutive_misses = 0
        self.hit_count = 0
        self.init_buffer = []

        # Logging
        self.estimate_history = []
        self.cov_trace_history = []
        self.innovation_history = []
        self.gated_count_history = []
        self.miss_flag_history = []
        self.status_history = []
        self.predicted_range_history = []
        self.predicted_speed_history = []

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(self):

        new_timestamp = self.state.timestamp + timedelta(seconds=self.dt)

        self.state = self.predictor.predict(
            self.state,
            timestamp=new_timestamp
        )

    # --------------------------------------------------
    # Gating
    # --------------------------------------------------

    def gate(self, detections):

        if not detections:
            return [], []

        gated = []
        innovations = []

        measurement_prediction = self.updater.predict_measurement(self.state)
        S = measurement_prediction.covar
        Sinv = np.linalg.inv(S)

        for det in detections:

            innovation = det.state_vector - measurement_prediction.state_vector

            # Normalize bearing innovation
            innovation[0, 0] = np.arctan2(
                np.sin(innovation[0, 0]),
                np.cos(innovation[0, 0])
            )

            d2 = (innovation.T @ Sinv @ innovation).item()

            gate_threshold = 9.21  # 99% chi-square, 2 DOF

            if d2 < gate_threshold:
                gated.append(det)
                innovations.append(d2)

        return gated, innovations

    # --------------------------------------------------
    # Association
    # --------------------------------------------------

    def associate(self, gated_detections, innovation_values):

        if not gated_detections:
            return None

        idx = np.argmin(innovation_values)
        return gated_detections[idx]

    # --------------------------------------------------
    # Update
    # --------------------------------------------------

    def update(self, detection):

        measurement_prediction = self.updater.predict_measurement(self.state)

        hypothesis = SingleHypothesis(
            self.state,
            detection,
            measurement_prediction
        )

        updated = self.updater.update(hypothesis)

        innovation = detection.state_vector - measurement_prediction.state_vector
        innovation[0, 0] = np.arctan2(
            np.sin(innovation[0, 0]),
            np.cos(innovation[0, 0])
        )

        innovation[0, 0] = np.arctan2(
            np.sin(innovation[0, 0]),
            np.cos(innovation[0, 0])
        )

        innovation_norm = np.linalg.norm(innovation).item()

        self.state = updated

        self.consecutive_misses = 0
        self.hit_count += 1

        if self.status == "tentative" and self.hit_count >= 3:
            self.status = "confirmed"

        return innovation_norm

    # --------------------------------------------------
    # Miss Handling
    # --------------------------------------------------

    def handle_miss(self):

        self.consecutive_misses += 1

        if self.consecutive_misses > 12:
            self.status = "deleted"

    # --------------------------------------------------
    # Step
    # --------------------------------------------------

    def step(self, detections):

        # -----------------------
        # Initialization
        # -----------------------
        if not self.initialized:

            if detections:
                self.init_buffer.append(detections[0])

            if len(self.init_buffer) >= 2:

                d1 = self.init_buffer[-2]
                d2 = self.init_buffer[-1]

                b1, r1 = d1.state_vector.flatten()
                b2, r2 = d2.state_vector.flatten()

                x1 = r1 * np.cos(b1)
                y1 = r1 * np.sin(b1)

                x2 = r2 * np.cos(b2)
                y2 = r2 * np.sin(b2)

                vx = (x2 - x1) / self.dt
                vy = (y2 - y1) / self.dt

                speed = np.sqrt(vx**2 + vy**2)

                # Reject unrealistic initializations
                if speed > 400:
                    self.init_buffer.pop(0)   # discard oldest
                    return

                initial_state = StateVector([x2, vx, y2, vy])

                P = np.diag([100**2, 200**2, 100**2, 200**2])

                self.state = GaussianState(
                    initial_state,
                    P,
                    timestamp=d2.timestamp
                )

                self.initialized = True
                self.hit_count = 2

            return

        # -----------------------
        # Normal tracking
        # -----------------------
        self.predict()

        gated, innovations = self.gate(detections)

        if gated:
            best = self.associate(gated, innovations)
            innovation_norm = self.update(best)
            miss_flag = 0
        else:
            innovation_norm = 0.0
            self.handle_miss()
            miss_flag = 1

        x = self.state.state_vector.flatten()
        speed = np.sqrt(x[1]**2 + x[3]**2)
        rng = np.sqrt(x[0]**2 + x[2]**2)

        self.estimate_history.append(x.copy())
        self.cov_trace_history.append(np.trace(self.state.covar))
        self.innovation_history.append(innovation_norm)
        self.gated_count_history.append(len(gated))
        self.miss_flag_history.append(miss_flag)
        self.status_history.append(self.status)
        self.predicted_range_history.append(rng)
        self.predicted_speed_history.append(speed)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    def get_results(self):

        return {
            "estimates": np.array(self.estimate_history),
            "cov_trace": np.array(self.cov_trace_history),
            "innovation_norm": np.array(self.innovation_history),
            "gated_count": np.array(self.gated_count_history),
            "miss_flags": np.array(self.miss_flag_history),
            "status_history": self.status_history,
            "predicted_range": np.array(self.predicted_range_history),
            "predicted_speed": np.array(self.predicted_speed_history)
        }


# ============================================================
# Standalone Execution
# ============================================================

if __name__ == "__main__":

    scene = simulate_scene(scene_type="stealth", plot=False)

    tracker = SingleTargetTracker(
        dt=scene["metadata"]["dt"],
        measurement_model=scene["measurements"]["measurement_model"],
        process_noise_scale=1.0
    )

    for detections in scene["measurements"]["detections"]:
        tracker.step(detections)

    results = tracker.get_results()

    # ======================================================
    # PLOT 1 — Simulation Scene + EKF Track
    # ======================================================

    plt.figure(figsize=(10, 8))

    plt.plot(scene["truth"]["x"],
             scene["truth"]["y"],
             linewidth=2,
             label="Truth")

    plt.scatter(scene["plot_data"]["clutter_x"],
                scene["plot_data"]["clutter_y"],
                s=6,
                alpha=0.15,
                label="Clutter")

    plt.scatter(scene["plot_data"]["detect_x"],
                scene["plot_data"]["detect_y"],
                s=20,
                marker='x',
                label="All Detections")

    est = results["estimates"]

    if len(est) > 0:
        plt.plot(est[:, 0],
                 est[:, 2],
                 linestyle='--',
                 linewidth=2,
                 label="EKF Track")

    plt.legend()
    plt.title("Radar Scene with EKF Track")
    plt.grid(True)
    plt.show()

    # ======================================================
    # PLOT 2 — Tracker Diagnostics
    # ======================================================

    t = np.arange(len(results["cov_trace"]))

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Tracker Diagnostics", fontsize=14)

    axs[0, 0].plot(t, results["innovation_norm"])
    axs[0, 0].set_title("Innovation Norm")
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, results["cov_trace"])
    axs[0, 1].set_title("Covariance Trace")
    axs[0, 1].grid(True)

    axs[0, 2].plot(t, results["gated_count"])
    axs[0, 2].set_title("Gated Detection Count")
    axs[0, 2].grid(True)

    axs[1, 0].plot(t, results["miss_flags"])
    axs[1, 0].set_title("Miss Flags")
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, results["predicted_range"])
    axs[1, 1].set_title("Predicted Range")
    axs[1, 1].grid(True)

    axs[1, 2].plot(t, results["predicted_speed"])
    axs[1, 2].set_title("Predicted Speed")
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()