"""Demo: 5-state vehicle estimator (x, y, yaw, u, v), Python bindings.

Mirrors core/executables/5_states_estimator/main.cpp: the predict step
integrates noisy IMU measurements (ax, ay, yaw_rate); the correction step
fuses a noisy [x, y, yaw, vg] measurement. Uses the class' default sensor
covariances (no explicit set_state_covariance/set_process_noise/
set_measurement_noise calls), then runs the RTS smoother over the recorded
history and reports the filtered vs. smoothed RMSE against ground truth.
"""

import math

import numpy as np

from kflib_estimators import FiveStateEstimator

DT = 0.01  # IMU rate: 100 Hz
CORRECT_EVERY = 10  # pose+speed correction at 10 Hz
STEPS = 2000


def main() -> None:
    rng = np.random.default_rng(12345)

    estimator = FiveStateEstimator()
    estimator.set_state(0.0, 0.0, 0.0, 1.0, 0.0)

    # True vehicle state, integrated with the same kinematic model.
    x, y, yaw, u, v = 0.0, 0.0, 0.0, 1.0, 0.0

    filtered_states = [estimator.get_state()]
    filtered_covariances = [estimator.get_covariance()]
    inputs = []
    true_states = [(x, y, yaw)]

    for i in range(STEPS):
        t = i * DT

        # True control inputs: mild acceleration and a slow S-curve turn.
        true_ax = 0.2 * math.sin(0.05 * t)
        true_ay = 0.0
        true_yaw_rate = 0.3 * math.sin(0.1 * t)

        # Integrate ground truth with the same kinematic model.
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        x += (u * cos_yaw - v * sin_yaw) * DT
        y += (u * sin_yaw + v * cos_yaw) * DT
        yaw += true_yaw_rate * DT
        u += (true_ax + true_yaw_rate * v) * DT
        v += (true_ay - true_yaw_rate * u) * DT

        # Noisy IMU measurement drives the predict step.
        meas_ax = true_ax + rng.normal(0.0, 0.05)
        meas_ay = true_ay + rng.normal(0.0, 0.05)
        meas_yaw_rate = true_yaw_rate + rng.normal(0.0, 0.01)
        estimator.predict(meas_ax, meas_ay, meas_yaw_rate, DT)
        inputs.append(FiveStateEstimator.make_input(meas_ax, meas_ay, meas_yaw_rate, DT))

        if i % CORRECT_EVERY == CORRECT_EVERY - 1:
            meas_x = x + rng.normal(0.0, 1.5)
            meas_y = y + rng.normal(0.0, 1.5)
            meas_yaw = yaw + rng.normal(0.0, math.radians(2.0))
            meas_vg = math.hypot(u, v) + rng.normal(0.0, 0.1)
            estimator.correct(meas_x, meas_y, meas_yaw, meas_vg)

        filtered_states.append(estimator.get_state())
        filtered_covariances.append(estimator.get_covariance())
        true_states.append((x, y, yaw))

    def position_rmse(states):
        errors = [
            math.hypot(s[0] - t[0], s[1] - t[1])
            for s, t in zip(states, true_states)
        ]
        return math.sqrt(sum(e * e for e in errors) / len(errors))

    print(f"filtered position RMSE: {position_rmse(filtered_states):.4f} m")

    smoothed_states, _ = estimator.smooth(filtered_states, filtered_covariances, inputs)
    print(f"smoothed position RMSE: {position_rmse(smoothed_states):.4f} m")

    print("\n%5s, %10s, %10s, %10s, %10s" % ("t", "true_x", "true_y", "filt_x", "smooth_x"))
    for i in range(0, STEPS + 1, 200):
        t = i * DT
        print(
            "%5.2f, %10.4f, %10.4f, %10.4f, %10.4f"
            % (t, true_states[i][0], true_states[i][1], filtered_states[i][0], smoothed_states[i][0])
        )


if __name__ == "__main__":
    main()
