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
from stonesoup.models.transition.nonlinear import ConstantTurn
from simulation import simulate_scene
from scipy.special import logsumexp
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.transition.nonlinear import ConstantTurn

class CVTracker:

    def __init__(self, dt, measurement_model, process_noise_scale=5.0):

        self.dt = dt
        self.measurement_model = measurement_model

        self.transition_model = CombinedLinearGaussianTransitionModel([
            ConstantVelocity(process_noise_scale),
            ConstantVelocity(process_noise_scale)
        ])

        self.predictor = ExtendedKalmanPredictor(self.transition_model)
        self.updater = ExtendedKalmanUpdater(measurement_model)

        self.base_bearing_std = np.radians(1.6)
        self.base_range_std = 20.0
        self.snr_ref = 10.0  # reference SNR scaling constant

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

    # ---------------- Prediction ----------------

    def predict(self):
        new_timestamp = self.state.timestamp + timedelta(seconds=self.dt)
        self.state = self.predictor.predict(
            self.state,
            timestamp=new_timestamp
        )

    # ---------------- Gating ----------------
    def gate(self, detections):

        if not detections:
            return [], []

        gated = []
        innovations = []

        for det in detections:

            snr_linear = det.metadata.get("snr_linear", 10.0)

            # Clamp SNR to avoid numerical explosion
            snr_linear = np.clip(snr_linear, 0.1, 1e4)

            scale = np.sqrt(self.snr_ref / snr_linear)

            range_std = self.base_range_std * scale
            bearing_std = self.base_bearing_std * scale

            R_dynamic = np.diag([bearing_std**2, range_std**2])

            dynamic_measurement_model = type(self.measurement_model)(
                ndim_state=self.measurement_model.ndim_state,
                mapping=self.measurement_model.mapping,
                noise_covar=R_dynamic
            )

            # IMPORTANT: do NOT modify detection object here
            measurement_prediction = self.updater.predict_measurement(
                self.state,
                measurement_model=dynamic_measurement_model
            )

            innovation = det.state_vector - measurement_prediction.state_vector

            innovation[0, 0] = np.arctan2(
                np.sin(innovation[0, 0]),
                np.cos(innovation[0, 0])
            )

            S = measurement_prediction.covar
            Sinv = np.linalg.inv(S)

            d2 = (innovation.T @ Sinv @ innovation).item()

            if d2 < 9.21:
                gated.append(det)
                innovations.append(d2)

        return gated, innovations

    # ---------------- Update ----------------
    def update(self, detection):

        snr_linear = detection.metadata.get("snr_linear", 10.0)
        snr_linear = np.clip(snr_linear, 0.1, 1e4)

        scale = np.sqrt(self.snr_ref / snr_linear)

        range_std = self.base_range_std * scale
        bearing_std = self.base_bearing_std * scale

        R_dynamic = np.diag([bearing_std**2, range_std**2])

        dynamic_measurement_model = type(self.measurement_model)(
            ndim_state=self.measurement_model.ndim_state,
            mapping=self.measurement_model.mapping,
            noise_covar=R_dynamic
        )

        # Attach dynamic model ONLY here
        detection.measurement_model = dynamic_measurement_model

        measurement_prediction = self.updater.predict_measurement(self.state)

        innovation = detection.state_vector - measurement_prediction.state_vector

        innovation[0, 0] = np.arctan2(
            np.sin(innovation[0, 0]),
            np.cos(innovation[0, 0])
        )

        S = measurement_prediction.covar
        Sinv = np.linalg.inv(S)
        d2 = (innovation.T @ Sinv @ innovation).item()

        hypothesis = SingleHypothesis(
            self.state,
            detection,
            measurement_prediction
        )

        self.state = self.updater.update(hypothesis)

        self.consecutive_misses = 0
        self.hit_count += 1

        if self.status == "tentative" and self.hit_count >= 3:
            self.status = "confirmed"

        return d2

    # ---------------- Miss Handling ----------------

    def handle_miss(self):
        self.consecutive_misses += 1
        if self.consecutive_misses > 6:
            self.status = "deleted"
            self.initialized = False

    # ---------------- Step ----------------

    def step(self, detections):

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

                if speed > 400:
                    self.init_buffer.pop(0)
                    return

                initial_state = StateVector([x2, vx, y2, vy])
                P = np.diag([200**2, 50**2, 200**2, 50**2])

                self.state = GaussianState(
                    initial_state,
                    P,
                    timestamp=d2.timestamp
                )

                self.initialized = True
                self.hit_count = 2

            return

        # Normal tracking

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

    def associate(self, gated_detections, innovation_values):

        if not gated_detections:
            return None

        idx = np.argmin(np.array(innovation_values))
        return gated_detections[idx]

class CTTracker:

    def __init__(self, dt, measurement_model):

        self.dt = dt
        self.measurement_model = measurement_model

        # STRONGER maneuver capability
        self.transition_model = ConstantTurn(
            linear_noise_coeffs=[1, 1],   # allow velocity diffusion
            turn_noise_coeff=0.1             # allow omega to move
        )

        self.predictor = ExtendedKalmanPredictor(self.transition_model)
        self.updater = ExtendedKalmanUpdater(measurement_model)

        self.base_bearing_std = np.radians(1.6)
        self.base_range_std = 20.0
        self.snr_ref = 10.0  # reference SNR scaling constant

        self.state = None
        self.initialized = False
        self.status = "tentative"

        self.consecutive_misses = 0
        self.hit_count = 0
        self.init_buffer = []

        self.estimate_history = []
        self.cov_trace_history = []
        self.innovation_history = []
        self.gated_count_history = []
        self.miss_flag_history = []
        self.status_history = []
        self.predicted_range_history = []
        self.predicted_speed_history = []

    # ---------------- Prediction ----------------

    def predict(self):
        new_timestamp = self.state.timestamp + timedelta(seconds=self.dt)
        self.state = self.predictor.predict(self.state, timestamp=new_timestamp)

    # ---------------- Gating ----------------
    def gate(self, detections):

        if not detections:
            return [], []

        gated = []
        innovations = []

        for det in detections:

            snr_linear = det.metadata.get("snr_linear", 10.0)
            snr_linear = np.clip(snr_linear, 0.1, 1e4)

            scale = np.sqrt(self.snr_ref / snr_linear)

            range_std = self.base_range_std * scale
            bearing_std = self.base_bearing_std * scale

            R_dynamic = np.diag([bearing_std**2, range_std**2])

            dynamic_measurement_model = type(self.measurement_model)(
                ndim_state=self.measurement_model.ndim_state,
                mapping=self.measurement_model.mapping,
                noise_covar=R_dynamic
            )

            measurement_prediction = self.updater.predict_measurement(
                self.state,
                measurement_model=dynamic_measurement_model
            )

            innovation = det.state_vector - measurement_prediction.state_vector

            innovation[0, 0] = np.arctan2(
                np.sin(innovation[0, 0]),
                np.cos(innovation[0, 0])
            )

            S = measurement_prediction.covar
            Sinv = np.linalg.inv(S)

            d2 = (innovation.T @ Sinv @ innovation).item()

            if d2 < 11.83:  # relaxed gate for CT
                gated.append(det)
                innovations.append(d2)

        return gated, innovations
    # ---------------- Update ----------------
    def update(self, detection):

        snr_linear = detection.metadata.get("snr_linear", 10.0)
        snr_linear = np.clip(snr_linear, 0.1, 1e4)

        scale = np.sqrt(self.snr_ref / snr_linear)

        range_std = self.base_range_std * scale
        bearing_std = self.base_bearing_std * scale

        R_dynamic = np.diag([bearing_std**2, range_std**2])

        dynamic_measurement_model = type(self.measurement_model)(
            ndim_state=self.measurement_model.ndim_state,
            mapping=self.measurement_model.mapping,
            noise_covar=R_dynamic
        )

        detection.measurement_model = dynamic_measurement_model

        measurement_prediction = self.updater.predict_measurement(self.state)

        innovation = detection.state_vector - measurement_prediction.state_vector

        innovation[0, 0] = np.arctan2(
            np.sin(innovation[0, 0]),
            np.cos(innovation[0, 0])
        )

        S = measurement_prediction.covar
        Sinv = np.linalg.inv(S)
        d2 = (innovation.T @ Sinv @ innovation).item()

        hypothesis = SingleHypothesis(
            self.state,
            detection,
            measurement_prediction
        )

        self.state = self.updater.update(hypothesis)

        self.consecutive_misses = 0
        self.hit_count += 1

        if self.status == "tentative" and self.hit_count >= 3:
            self.status = "confirmed"

        return d2
    # ---------------- Miss Handling ----------------

    def handle_miss(self):
        self.consecutive_misses += 1
        if self.consecutive_misses > 8:
            self.status = "deleted"

    # ---------------- Initialization ----------------

    def initialize_from_detections(self):

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

        if np.sqrt(vx**2 + vy**2) > 400:
            self.init_buffer.pop(0)
            return False

        omega = 0.0

        initial_state = StateVector([x2, vx, y2, vy, omega])

        # Much larger omega uncertainty
        P = np.diag([
            200**2,
            50**2,
            200**2,
            50**2,
            np.radians(20)**2   # was 2°, now 20°
        ])

        self.state = GaussianState(
            initial_state,
            P,
            timestamp=d2.timestamp
        )

        self.initialized = True
        self.hit_count = 2

        return True

    # ---------------- Step ----------------

    def step(self, detections):

        if not self.initialized:

            if detections:
                self.init_buffer.append(detections[0])

            if len(self.init_buffer) >= 2:
                success = self.initialize_from_detections()
                if not success:
                    return

            return

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

    def associate(self, gated_detections, innovation_values):

        if not gated_detections:
            return None

        idx = np.argmin(np.array(innovation_values))
        return gated_detections[idx]

class IMMTracker:

    def __init__(self, cv_tracker, ct_tracker):

        self.cv = cv_tracker
        self.ct = ct_tracker

        # Mode probabilities: [CV, CT]
        self.mu = np.array([0.5, 0.5])

        # Transition probability matrix
        self.PI = np.array([
            [0.92, 0.08],
            [0.08, 0.92]
        ])

        self.mu_history = []
        self.fused_history = []

    def interaction(self):

        if not (self.cv.initialized and self.ct.initialized):
            return

        mu_prev = self.mu
        c = self.PI.T @ mu_prev

        mu_ij = np.zeros((2,2))

        for j in range(2):
            for i in range(2):
                mu_ij[i,j] = self.PI[i,j] * mu_prev[i] / c[j]

        # Expand CV state to 5D for CT compatibility
        x_cv = self.cv.state.state_vector.flatten()
        P_cv = self.cv.state.covar

        x_cv5 = np.array([
            x_cv[0], x_cv[1],
            x_cv[2], x_cv[3],
            0.0
        ]).reshape(-1,1)

        P_cv5 = np.zeros((5,5))
        P_cv5[:4,:4] = P_cv
        P_cv5[4,4] = np.radians(5)**2

        x_ct = self.ct.state.state_vector
        P_ct = self.ct.state.covar

        X = [x_cv5, x_ct]
        P = [P_cv5, P_ct]

        mixed_states = []
        mixed_covs = []

        for j in range(2):
            x0 = mu_ij[0,j]*X[0] + mu_ij[1,j]*X[1]

            P0 = (
                mu_ij[0,j]*(P[0] + (X[0]-x0)@(X[0]-x0).T) +
                mu_ij[1,j]*(P[1] + (X[1]-x0)@(X[1]-x0).T)
            )

            mixed_states.append(x0)
            mixed_covs.append(P0)

        # Assign back
        self.cv.state.state_vector = mixed_states[0][:4]
        self.cv.state.covar = mixed_covs[0][:4,:4]

        self.ct.state.state_vector = mixed_states[1]
        self.ct.state.covar = mixed_covs[1]

    def update_mode_probabilities(self, d2_cv, d2_ct):

        mu_prior = self.PI.T @ self.mu
        log_mu_prior = np.log(mu_prior + 1e-12)

        log_likelihoods = np.array([
            -0.5 * d2_cv,
            -0.5 * d2_ct
        ])

        log_mu_post = log_mu_prior + log_likelihoods
        log_mu_post -= logsumexp(log_mu_post)

        self.mu = np.exp(log_mu_post)
        self.mu_history.append(self.mu.copy())

    def fuse(self):

        x_cv = self.cv.state.state_vector.flatten()
        x_ct = self.ct.state.state_vector.flatten()

        # Expand CV to 5D
        x_cv5 = np.array([
            x_cv[0], x_cv[1],
            x_cv[2], x_cv[3],
            0.0
        ])

        x = self.mu[0]*x_cv5 + self.mu[1]*x_ct

        self.fused_history.append(x.copy())

if __name__ == "__main__":

    for scene_type in ["stealth", "aircraft"]:
        scene = simulate_scene(scene_type, plot=False)

        cv = CVTracker(
            dt=scene["metadata"]["dt"],
            measurement_model=scene["measurements"]["measurement_model"],
            process_noise_scale=2.0
        )

        ct = CTTracker(
            dt=scene["metadata"]["dt"],
            measurement_model=scene["measurements"]["measurement_model"]
        )

        imm = IMMTracker(cv, ct)

        for detections in scene["measurements"]["detections"]:

            # ----------------------------
            # Initialization phase
            # ----------------------------
            if not cv.initialized:
                cv.step(detections)

            if not ct.initialized:
                ct.step(detections)

            if not (cv.initialized and ct.initialized):
                continue

            # ----------------------------
            # IMM Interaction
            # ----------------------------
            imm.interaction()

            # ----------------------------
            # Predict
            # ----------------------------
            cv.predict()
            ct.predict()

            # ----------------------------
            # CV model
            # ----------------------------
            gated_cv, innov_cv = cv.gate(detections)

            if len(gated_cv) > 0:
                best_cv = cv.associate(gated_cv, innov_cv)
                d2_cv = cv.update(best_cv)
            else:
                cv.handle_miss()
                d2_cv = 100.0   # large penalty

            # ----------------------------
            # CT model
            # ----------------------------
            gated_ct, innov_ct = ct.gate(detections)

            if len(gated_ct) > 0:
                best_ct = ct.associate(gated_ct, innov_ct)
                d2_ct = ct.update(best_ct)
            else:
                ct.handle_miss()
                d2_ct = 100.0   # large penalty

            # ----------------------------
            # Update mode probabilities
            # ----------------------------
            imm.update_mode_probabilities(d2_cv, d2_ct)

            # ----------------------------
            # Fuse state
            # ----------------------------
            imm.fuse()

        # ==========================================
        # Convert fused history
        # ==========================================
        fused = np.array(imm.fused_history)
        mu = np.array(imm.mu_history)

        # ==========================================
        # PLOT 1 — Radar Scene + IMM Track
        # ==========================================
        plt.figure(figsize=(10,8))

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

        if len(fused) > 0:
            plt.plot(fused[:,0],
                    fused[:,2],
                    color='green',
                    linewidth=2,
                    label="IMM (CV+CT) Track")

        plt.legend()
        plt.title("Radar Scene with IMM Track (CV + CT)")
        plt.grid(True)
        plt.show()

        # ==========================================
        # PLOT 2 — Mode Probabilities
        # ==========================================
        if len(mu) > 0:
            plt.figure()
            plt.plot(mu[:,0], label="CV Probability")
            plt.plot(mu[:,1], label="CT Probability")
            plt.legend()
            plt.title("IMM Mode Probabilities")
            plt.grid(True)
            plt.show()