# ML-Based Classification of Low Radar Cross-Section Aerial Targets
---

## Overview

This project simulates a physics-based radar environment and applies machine learning to detect and classify aerial targets under realistic sensing conditions.

The system models radar propagation physics, RCS fluctuation, SNR-driven detection probability, clutter, and missed detections. Targets are tracked using an Extended Kalman Filter (EKF), and statistical track features are used to classify credible aircraft — including low-RCS stealth targets — from background radar noise.

*The project is currently under active development.

---

## System Architecture

Simulation → Tracking → Feature Extraction → Hierarchical ML Classification

### 1. Radar Simulation
- Constant velocity motion model
- Cartesian → Polar measurement model
- Swerling I RCS fluctuation
- SNR-based detection probability
- Clutter modeling
- Missed detections (stealth behavior)

### 2. Tracking Layer
- Extended Kalman Filter (EKF)
- Handles nonlinear bearing-range measurements
- Covariance growth during missed detections
- Observability degradation modeling

### 3. Feature Engineering
Extracted track-level features:
- Kinematic statistics (speed, acceleration variance)
- Detection statistics (detection ratio, gaps)
- Estimation uncertainty metrics (covariance trace, growth rate)

### 4. Hierarchical Classification
- **Stage 1:** Noise vs Credible Target
- **Stage 2:** Bird vs Aircraft
- **Stage 3:** Strong RCS vs Weak RCS (Stealth Detection)

Models: XGBoost (binary classifiers)

---

## Project Status

- Radar simulation layer: Implemented
- EKF tracking layer: Implemented
- Feature extraction pipeline: Implemented
- Hierarchical ML training: Implemented
- Multi-target scenarios: Planned
- Adaptive radar parameter randomization: Planned
- Performance benchmarking: Ongoing

---

## Tech Stack

- Python
- NumPy, Pandas
- XGBoost
- scikit-learn
- Stone Soup (tracking framework)
- Matplotlib

---

## Research Direction

The core objective is to investigate how observability degradation — caused by low radar cross-section (RCS) — can be learned statistically from track behavior rather than raw signal amplitude alone.

This work aims to bridge radar physics, state estimation, and machine learning into a unified detection framework.

---

## Note

This project is an academic simulation developed for research and educational purposes as part of an undergraduate 8th semester project at VIT Vellore.
