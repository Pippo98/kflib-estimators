# kflib-estimators

## Introduction

`kflib-estimators` provides `FiveStateEstimator`, an Extended Kalman Filter
for estimating the planar pose and body-frame velocities of a vehicle:

- `x`, `y` — position in the world frame.
- `yaw` — heading angle (zero points along the world +x axis).
- `u` — longitudinal velocity in the vehicle frame (positive forward).
- `v` — lateral velocity in the vehicle frame (positive left).

The filter **predicts** by integrating IMU measurements (`ax`, `ay`,
`yaw_rate`) through the vehicle's kinematic model, and **corrects** with an
absolute measurement `[x, y, yaw, vg]` (`vg = hypot(u, v)`, e.g. from GPS).
It ships with sensible default sensor covariances so it runs out of the
box, and includes a Rauch-Tung-Striebel smoother for offline post-processing
of a whole recorded trip.

The estimator is implemented once in C++, on top of
[kflib](https://github.com/Pippo98/kflib)'s Kalman filter library, and
exposed to Python via [nanobind](https://github.com/wjakob/nanobind)
bindings — the same tested code runs natively from C++ or from Python.

## Installation

```bash
pip install "git+https://github.com/Pippo98/kflib-estimators.git"
```

This builds the C++ extension from source (via
[scikit-build-core](https://github.com/scikit-build/scikit-build-core) +
CMake), so it needs a C++20 compiler and CMake on the machine doing the
install; `pip` fetches the rest (including `cmake`/`ninja`) itself. Git
submodules (kflib, nanobind, Eigen) are fetched automatically by pip's git
backend. Requires Python >= 3.10.

For local development, clone with submodules and install in editable mode:

```bash
git clone --recursive https://github.com/Pippo98/kflib-estimators.git
cd kflib-estimators
pip install -e .
```

## Most probable use

The typical workflow is to run the filter over a whole logged trip in one
call with `estimate_and_smooth_from_series()`, passing plain columns (e.g.
straight from a pandas DataFrame) and a `measurement_dt` that sets how often
to correct with the — usually lower-rate — pose/speed measurement; every
other sample still contributes to prediction. Set the measurement noise and
the true initial state first for best results; everything else already has
a workable default.

```python
fse = kfe.FiveStateEstimator()
fse.set_measurement_noise(np.array([
    [0.5, 0.0, 0.0, 0.00], # x
    [0.0, 0.5, 0.0, 0.00], # y
    [0.0, 0.0, 0.2, 0.00], # yaw
    [0.0, 0.0, 0.0, 0.01], # vg
]))
fse.set_state(df.loc[0,"x"], df.loc[0,"y"], df.loc[0,"course"], df.loc[0,"vx"], df.loc[0,"vy"])
states, covariances = fse.estimate_and_smooth_from_series(
        df["time"].to_numpy(),
        df["ax"].to_numpy(),
        df["ay"].to_numpy(),
        df["yaw_rate"].to_numpy(),
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        df["course"].to_numpy(),
        df["vg"].to_numpy(),
        float(mean_dt) * 5.0 # correct once every 5 measurements
)
states = np.array(states)  # shape (N, 5): columns are [x, y, yaw, u, v]
```

## Other uses and features

**Streaming / online use** — feed the filter one reading at a time instead
of a whole recorded batch, e.g. as IMU/GPS messages arrive:

```python
fse = kfe.FiveStateEstimator()
fse.set_state(x0, y0, yaw0, u0, v0)
while True:
    ax, ay, yaw_rate, dt = read_imu()
    fse.predict(ax, ay, yaw_rate, dt)
    if new_pose_available():
        x, y, yaw, vg = read_pose()
        fse.correct(x, y, yaw, vg)
    x, y, yaw, u, v = fse.get_state()
```

**Finer control over what gets corrected** — `estimate()`/`smooth()` take
pre-built `measurements`/`inputs` lists (via `make_measurement()`/
`make_input()`) instead of columns + `measurement_dt`, for cases where the
`*_from_series()` decimation logic doesn't fit (e.g. correcting on
irregularly-timed or sensor-flagged samples). Skip a correction at a given
timestep by putting an empty array in its slot in `measurements`:

```python
measurements = [
    kfe.FiveStateEstimator.make_measurement(x, y, yaw, vg) if has_fix else np.array([])
    for x, y, yaw, vg, has_fix in zip(xs, ys, yaws, vgs, fixes)
]
inputs = [kfe.FiveStateEstimator.make_input(ax, ay, r, dt) for ax, ay, r, dt in zip(axs, ays, rs, dts)]
states, covariances = fse.estimate(measurements, inputs)
states, covariances = fse.smooth(states, covariances, inputs)
```

`estimate_from_series()`/`smooth_from_series()` sit in between: same
column/`measurement_dt` inputs as `estimate_and_smooth_from_series()`, but
run (and let you inspect) the filtered and smoothed results separately.

**Defaults** — a freshly constructed `FiveStateEstimator` already has
ballpark process/measurement covariances (roughly consumer-grade automotive
IMU + GPS) and a zero initial state, so it runs without any configuration;
override with `set_state`, `set_state_covariance`, `set_process_noise` and
`set_measurement_noise` once real sensor characteristics are known.

**From C++** — the same class (`core/inc/5_state_estimator.hpp`,
`core/src/5_state_estimator.cpp`) is usable directly with no Python
involved; see `core/executables/5_states_estimator/main.cpp` for a runnable
example (predict/correct loop, then `smooth()`). A runnable Python example
is at `python/examples/example_five_state_estimator.py`.

**Other filters** — the underlying [kflib](https://github.com/Pippo98/kflib)
submodule (`external/kflib`) also provides standalone linear (`KalmanFilter`)
and unscented (`UnscentedKalmanFilter`) filter implementations alongside the
`ExtendedKalmanFilter` that `FiveStateEstimator` itself builds on, in case a
different vehicle/sensor model is needed.
