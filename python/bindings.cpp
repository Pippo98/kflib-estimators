#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "4_state_estimator.hpp"
#include "5_state_estimator.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Shared building blocks for the *_from_series() convenience bindings:
// builds the `inputs`/`measurements` vectors estimate()/smooth() expect
// from plain time-series column vectors (e.g. pandas DataFrame columns).
// One set per estimator, since each has a different input/measurement
// field list.

// --- FiveStateEstimator ---

// One input per step from sample i to i+1: dt from consecutive timestamps,
// IMU reading held from the start of the step.
std::vector<Eigen::VectorXd> buildFiveStateInputsFromSeries(
    const Eigen::VectorXd &time, const Eigen::VectorXd &ax,
    const Eigen::VectorXd &ay, const Eigen::VectorXd &yawRate) {
  const Eigen::Index n = time.size();
  if (n == 0 || ax.size() != n || ay.size() != n || yawRate.size() != n) {
    throw std::invalid_argument(
        "time, ax, ay, yaw_rate must all be non-empty and the same length");
  }
  std::vector<Eigen::VectorXd> inputs;
  inputs.reserve(static_cast<size_t>(n - 1));
  for (Eigen::Index i = 0; i + 1 < n; ++i) {
    inputs.push_back(FiveStateEstimator::makeInput(ax(i), ay(i), yawRate(i),
                                                     time(i + 1) - time(i)));
  }
  return inputs;
}

// Correct only close to every `measurementDt`, always including the first
// sample; every other sample gets an empty (skipped) measurement slot, per
// estimate()'s contract.
std::vector<Eigen::VectorXd> buildFiveStateMeasurementsFromSeries(
    const Eigen::VectorXd &time, const Eigen::VectorXd &x,
    const Eigen::VectorXd &y, const Eigen::VectorXd &yaw,
    const Eigen::VectorXd &vg, double measurementDt) {
  const Eigen::Index n = time.size();
  if (x.size() != n || y.size() != n || yaw.size() != n || vg.size() != n) {
    throw std::invalid_argument(
        "x, y, yaw, vg must be the same length as `time`");
  }
  std::vector<Eigen::VectorXd> measurements;
  measurements.reserve(static_cast<size_t>(n));
  double lastCorrectionTime = time(0);
  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == 0 || time(i) - lastCorrectionTime >= measurementDt) {
      measurements.push_back(
          FiveStateEstimator::makeMeasurement(x(i), y(i), yaw(i), vg(i)));
      lastCorrectionTime = time(i);
    } else {
      measurements.emplace_back();
    }
  }
  return measurements;
}

// Shared body for smooth(): converts the default-allocator vectors nanobind
// hands us to/from the aligned-allocator lists FiveStateEstimator::smooth()
// needs.
std::pair<std::vector<FiveStateEstimator::StateVector>,
          std::vector<FiveStateEstimator::StateCovariance>>
smoothFiveStateImpl(
    FiveStateEstimator &self,
    const std::vector<FiveStateEstimator::StateVector> &statesIn,
    const std::vector<FiveStateEstimator::StateCovariance> &covariancesIn,
    const std::vector<Eigen::VectorXd> &inputs) {
  FiveStateEstimator::StateVectorList states(statesIn.begin(), statesIn.end());
  FiveStateEstimator::StateCovarianceList covariances(covariancesIn.begin(),
                                                        covariancesIn.end());
  self.smooth(states, covariances, inputs);
  return {std::vector<FiveStateEstimator::StateVector>(states.begin(),
                                                         states.end()),
          std::vector<FiveStateEstimator::StateCovariance>(
              covariances.begin(), covariances.end())};
}

// --- FourStateEstimator ---

std::vector<Eigen::VectorXd>
buildFourStateInputsFromSeries(const Eigen::VectorXd &time,
                                const Eigen::VectorXd &ax,
                                const Eigen::VectorXd &yawRate) {
  const Eigen::Index n = time.size();
  if (n == 0 || ax.size() != n || yawRate.size() != n) {
    throw std::invalid_argument(
        "time, ax, yaw_rate must all be non-empty and the same length");
  }
  std::vector<Eigen::VectorXd> inputs;
  inputs.reserve(static_cast<size_t>(n - 1));
  for (Eigen::Index i = 0; i + 1 < n; ++i) {
    inputs.push_back(FourStateEstimator::makeInput(ax(i), yawRate(i),
                                                     time(i + 1) - time(i)));
  }
  return inputs;
}

std::vector<Eigen::VectorXd> buildFourStateMeasurementsFromSeries(
    const Eigen::VectorXd &time, const Eigen::VectorXd &x,
    const Eigen::VectorXd &y, const Eigen::VectorXd &yaw,
    const Eigen::VectorXd &vg, const Eigen::VectorXd &ay,
    double measurementDt) {
  const Eigen::Index n = time.size();
  if (x.size() != n || y.size() != n || yaw.size() != n || vg.size() != n ||
      ay.size() != n) {
    throw std::invalid_argument(
        "x, y, yaw, vg, ay must be the same length as `time`");
  }
  std::vector<Eigen::VectorXd> measurements;
  measurements.reserve(static_cast<size_t>(n));
  double lastCorrectionTime = time(0);
  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == 0 || time(i) - lastCorrectionTime >= measurementDt) {
      measurements.push_back(FourStateEstimator::makeMeasurement(
          x(i), y(i), yaw(i), vg(i), ay(i)));
      lastCorrectionTime = time(i);
    } else {
      measurements.emplace_back();
    }
  }
  return measurements;
}

std::pair<std::vector<FourStateEstimator::StateVector>,
          std::vector<FourStateEstimator::StateCovariance>>
smoothFourStateImpl(
    FourStateEstimator &self,
    const std::vector<FourStateEstimator::StateVector> &statesIn,
    const std::vector<FourStateEstimator::StateCovariance> &covariancesIn,
    const std::vector<Eigen::VectorXd> &inputs) {
  FourStateEstimator::StateVectorList states(statesIn.begin(), statesIn.end());
  FourStateEstimator::StateCovarianceList covariances(covariancesIn.begin(),
                                                        covariancesIn.end());
  self.smooth(states, covariances, inputs);
  return {std::vector<FourStateEstimator::StateVector>(states.begin(),
                                                         states.end()),
          std::vector<FourStateEstimator::StateCovariance>(
              covariances.begin(), covariances.end())};
}

} // namespace

NB_MODULE(kflib_estimators, m) {
  m.doc() = "Python bindings for kflib-estimators' vehicle EKF estimators: "
            "FiveStateEstimator (x, y, yaw, u, v) and FourStateEstimator "
            "(x, y, yaw, vg, no lateral velocity).";

  nb::class_<FiveStateEstimator>(m, "FiveStateEstimator")
      .def(nb::init<>(), "Construct with default sensor covariances.")
      .def("set_state", &FiveStateEstimator::setState, "x"_a, "y"_a, "yaw"_a,
           "u"_a, "v"_a, "Set the state vector [x, y, yaw, u, v].")
      .def("set_state_covariance", &FiveStateEstimator::setStateCovariance,
           "P"_a, "Set the 5x5 state covariance matrix.")
      .def("set_process_noise", &FiveStateEstimator::setProcessNoise, "Q"_a,
           "Set the 5x5 process noise covariance matrix.")
      .def("set_measurement_noise", &FiveStateEstimator::setMeasurementNoise,
           "R"_a, "Set the 4x4 measurement noise covariance matrix.")
      .def("predict", &FiveStateEstimator::predict, "ax"_a, "ay"_a,
           "yaw_rate"_a, "dt"_a,
           "Predict step: integrate IMU measurements (ax, ay, yaw_rate) "
           "over dt.")
      .def("correct", &FiveStateEstimator::correct, "x"_a, "y"_a, "yaw"_a,
           "vg"_a,
           "Correction step with a [x, y, yaw, vg] measurement, where vg "
           "= hypot(u, v).")
      .def("get_state", &FiveStateEstimator::getState,
           "Return the state vector [x, y, yaw, u, v].")
      .def("get_covariance", &FiveStateEstimator::getCovariance,
           "Return the 5x5 state covariance matrix.")
      .def_static("make_input", &FiveStateEstimator::makeInput, "ax"_a,
                   "ay"_a, "yaw_rate"_a, "dt"_a,
                   "Build an EKF input vector [ax, ay, yaw_rate, dt], for "
                   "use with the `inputs` argument of estimate()/smooth().")
      .def_static("make_measurement", &FiveStateEstimator::makeMeasurement,
                   "x"_a, "y"_a, "yaw"_a, "vg"_a,
                   "Build a measurement vector [x, y, yaw, vg], for use "
                   "with the `measurements` argument of estimate().")
      .def("estimate", &FiveStateEstimator::estimate, "measurements"_a,
           "inputs"_a,
           "Run the filter over a whole recorded sequence: correct() with "
           "measurements[0] first (no predict before it), then alternate "
           "predict(inputs[i]) / correct(measurements[i + 1]) for the "
           "rest. Overwrites the estimator's current state/covariance as "
           "it goes. `inputs` must have `len(measurements) - 1` entries. "
           "Returns (states, covariances), one entry per measurement, in "
           "the layout smooth() expects (with these same `inputs`), so "
           "the two chain directly.")
      .def(
          "estimate_from_series",
          [](FiveStateEstimator &self, const Eigen::VectorXd &time,
             const Eigen::VectorXd &ax, const Eigen::VectorXd &ay,
             const Eigen::VectorXd &yawRate, const Eigen::VectorXd &x,
             const Eigen::VectorXd &y, const Eigen::VectorXd &yaw,
             const Eigen::VectorXd &vg, double measurementDt) {
            auto inputs = buildFiveStateInputsFromSeries(time, ax, ay, yawRate);
            auto measurements = buildFiveStateMeasurementsFromSeries(
                time, x, y, yaw, vg, measurementDt);
            return self.estimate(measurements, inputs);
          },
          "time"_a, "ax"_a, "ay"_a, "yaw_rate"_a, "x"_a, "y"_a, "yaw"_a,
          "vg"_a, "measurement_dt"_a,
          "Convenience wrapper around estimate() for time-series data "
          "(e.g. columns from a pandas DataFrame): pass absolute "
          "timestamps `time` plus one column vector per IMU/measurement "
          "field (all the same length as `time`), and a `measurement_dt` "
          "giving the desired spacing between correction steps. Every "
          "sample is used for prediction; only the first sample and "
          "samples at least `measurement_dt` after the last correction "
          "are used to correct() (the rest are skipped, per estimate()'s "
          "empty-measurement contract). Builds the `inputs`/`measurements` "
          "arguments to estimate() internally and calls it, so the return "
          "value is the same (states, covariances) pair.")
      .def(
          "smooth",
          [](FiveStateEstimator &self,
             const std::vector<FiveStateEstimator::StateVector> &states,
             const std::vector<FiveStateEstimator::StateCovariance>
                 &covariances,
             const std::vector<Eigen::VectorXd> &inputs) {
            return smoothFiveStateImpl(self, states, covariances, inputs);
          },
          "states"_a, "covariances"_a, "inputs"_a,
          "Rauch-Tung-Striebel smoother over a recorded, chronologically "
          "ordered sequence of filtered states/covariances and the inputs "
          "used at each predict() call (see make_input()). Returns "
          "(smoothed_states, smoothed_covariances); does not mutate the "
          "estimator's current state.")
      .def(
          "smooth_from_series",
          [](FiveStateEstimator &self,
             const std::vector<FiveStateEstimator::StateVector> &states,
             const std::vector<FiveStateEstimator::StateCovariance>
                 &covariances,
             const Eigen::VectorXd &time, const Eigen::VectorXd &ax,
             const Eigen::VectorXd &ay, const Eigen::VectorXd &yawRate) {
            auto inputs = buildFiveStateInputsFromSeries(time, ax, ay, yawRate);
            return smoothFiveStateImpl(self, states, covariances, inputs);
          },
          "states"_a, "covariances"_a, "time"_a, "ax"_a, "ay"_a,
          "yaw_rate"_a,
          "Same as smooth(), but for time-series data: `states`/"
          "`covariances` are unchanged (as returned by estimate()/"
          "estimate_from_series()), while the `inputs` argument is built "
          "from `time`/`ax`/`ay`/`yaw_rate` the same way "
          "estimate_from_series() does. Returns (smoothed_states, "
          "smoothed_covariances).")
      .def(
          "estimate_and_smooth_from_series",
          [](FiveStateEstimator &self, const Eigen::VectorXd &time,
             const Eigen::VectorXd &ax, const Eigen::VectorXd &ay,
             const Eigen::VectorXd &yawRate, const Eigen::VectorXd &x,
             const Eigen::VectorXd &y, const Eigen::VectorXd &yaw,
             const Eigen::VectorXd &vg, double measurementDt) {
            auto inputs = buildFiveStateInputsFromSeries(time, ax, ay, yawRate);
            auto measurements = buildFiveStateMeasurementsFromSeries(
                time, x, y, yaw, vg, measurementDt);
            auto [states, covariances] = self.estimate(measurements, inputs);
            return smoothFiveStateImpl(
                self,
                std::vector<FiveStateEstimator::StateVector>(states.begin(),
                                                               states.end()),
                std::vector<FiveStateEstimator::StateCovariance>(
                    covariances.begin(), covariances.end()),
                inputs);
          },
          "time"_a, "ax"_a, "ay"_a, "yaw_rate"_a, "x"_a, "y"_a, "yaw"_a,
          "vg"_a, "measurement_dt"_a,
          "Runs estimate_from_series() followed by smooth_from_series() "
          "(sharing the same built `inputs`, rather than rebuilding them "
          "twice) and returns its (smoothed_states, smoothed_covariances). "
          "See estimate_from_series() for the argument semantics.");

  nb::class_<FourStateEstimator>(m, "FourStateEstimator")
      .def(nb::init<>(), "Construct with default sensor covariances.")
      .def("set_state", &FourStateEstimator::setState, "x"_a, "y"_a, "yaw"_a,
           "vg"_a, "Set the state vector [x, y, yaw, vg].")
      .def("set_state_covariance", &FourStateEstimator::setStateCovariance,
           "P"_a, "Set the 4x4 state covariance matrix.")
      .def("set_process_noise", &FourStateEstimator::setProcessNoise, "Q"_a,
           "Set the 4x4 process noise covariance matrix.")
      .def("set_measurement_noise", &FourStateEstimator::setMeasurementNoise,
           "R"_a, "Set the 5x5 measurement noise covariance matrix.")
      .def("predict", &FourStateEstimator::predict, "ax"_a, "yaw_rate"_a,
           "dt"_a,
           "Predict step: integrate IMU measurements (ax, yaw_rate) over "
           "dt (no-lateral-velocity model: no `ay` input).")
      .def("correct", &FourStateEstimator::correct, "x"_a, "y"_a, "yaw"_a,
           "vg"_a, "ay"_a,
           "Correction step with a [x, y, yaw, vg, ay] measurement, where "
           "ay = vg * yaw_rate is the lateral acceleration implied by the "
           "no-lateral-velocity assumption (yaw_rate from the preceding "
           "predict() call).")
      .def("get_state", &FourStateEstimator::getState,
           "Return the state vector [x, y, yaw, vg].")
      .def("get_covariance", &FourStateEstimator::getCovariance,
           "Return the 4x4 state covariance matrix.")
      .def_static("make_input", &FourStateEstimator::makeInput, "ax"_a,
                   "yaw_rate"_a, "dt"_a,
                   "Build an EKF input vector [ax, yaw_rate, dt], for use "
                   "with the `inputs` argument of estimate()/smooth().")
      .def_static("make_measurement", &FourStateEstimator::makeMeasurement,
                   "x"_a, "y"_a, "yaw"_a, "vg"_a, "ay"_a,
                   "Build a measurement vector [x, y, yaw, vg, ay], for "
                   "use with the `measurements` argument of estimate().")
      .def("estimate", &FourStateEstimator::estimate, "measurements"_a,
           "inputs"_a,
           "Run the filter over a whole recorded sequence: correct() with "
           "measurements[0] first (no predict before it), then alternate "
           "predict(inputs[i]) / correct(measurements[i + 1]) for the "
           "rest. Overwrites the estimator's current state/covariance as "
           "it goes. `inputs` must have `len(measurements) - 1` entries. "
           "Returns (states, covariances), one entry per measurement, in "
           "the layout smooth() expects (with these same `inputs`), so "
           "the two chain directly.")
      .def(
          "estimate_from_series",
          [](FourStateEstimator &self, const Eigen::VectorXd &time,
             const Eigen::VectorXd &ax, const Eigen::VectorXd &yawRate,
             const Eigen::VectorXd &x, const Eigen::VectorXd &y,
             const Eigen::VectorXd &yaw, const Eigen::VectorXd &vg,
             const Eigen::VectorXd &ay, double measurementDt) {
            auto inputs = buildFourStateInputsFromSeries(time, ax, yawRate);
            auto measurements = buildFourStateMeasurementsFromSeries(
                time, x, y, yaw, vg, ay, measurementDt);
            return self.estimate(measurements, inputs);
          },
          "time"_a, "ax"_a, "yaw_rate"_a, "x"_a, "y"_a, "yaw"_a, "vg"_a,
          "ay"_a, "measurement_dt"_a,
          "Convenience wrapper around estimate() for time-series data "
          "(e.g. columns from a pandas DataFrame): pass absolute "
          "timestamps `time` plus one column vector per IMU/measurement "
          "field (all the same length as `time`), and a `measurement_dt` "
          "giving the desired spacing between correction steps. Every "
          "sample is used for prediction; only the first sample and "
          "samples at least `measurement_dt` after the last correction "
          "are used to correct() (the rest are skipped, per estimate()'s "
          "empty-measurement contract). Builds the `inputs`/`measurements` "
          "arguments to estimate() internally and calls it, so the return "
          "value is the same (states, covariances) pair.")
      .def(
          "smooth",
          [](FourStateEstimator &self,
             const std::vector<FourStateEstimator::StateVector> &states,
             const std::vector<FourStateEstimator::StateCovariance>
                 &covariances,
             const std::vector<Eigen::VectorXd> &inputs) {
            return smoothFourStateImpl(self, states, covariances, inputs);
          },
          "states"_a, "covariances"_a, "inputs"_a,
          "Rauch-Tung-Striebel smoother over a recorded, chronologically "
          "ordered sequence of filtered states/covariances and the inputs "
          "used at each predict() call (see make_input()). Returns "
          "(smoothed_states, smoothed_covariances); does not mutate the "
          "estimator's current state.")
      .def(
          "smooth_from_series",
          [](FourStateEstimator &self,
             const std::vector<FourStateEstimator::StateVector> &states,
             const std::vector<FourStateEstimator::StateCovariance>
                 &covariances,
             const Eigen::VectorXd &time, const Eigen::VectorXd &ax,
             const Eigen::VectorXd &yawRate) {
            auto inputs = buildFourStateInputsFromSeries(time, ax, yawRate);
            return smoothFourStateImpl(self, states, covariances, inputs);
          },
          "states"_a, "covariances"_a, "time"_a, "ax"_a, "yaw_rate"_a,
          "Same as smooth(), but for time-series data: `states`/"
          "`covariances` are unchanged (as returned by estimate()/"
          "estimate_from_series()), while the `inputs` argument is built "
          "from `time`/`ax`/`yaw_rate` the same way estimate_from_series() "
          "does. Returns (smoothed_states, smoothed_covariances).")
      .def(
          "estimate_and_smooth_from_series",
          [](FourStateEstimator &self, const Eigen::VectorXd &time,
             const Eigen::VectorXd &ax, const Eigen::VectorXd &yawRate,
             const Eigen::VectorXd &x, const Eigen::VectorXd &y,
             const Eigen::VectorXd &yaw, const Eigen::VectorXd &vg,
             const Eigen::VectorXd &ay, double measurementDt) {
            auto inputs = buildFourStateInputsFromSeries(time, ax, yawRate);
            auto measurements = buildFourStateMeasurementsFromSeries(
                time, x, y, yaw, vg, ay, measurementDt);
            auto [states, covariances] = self.estimate(measurements, inputs);
            return smoothFourStateImpl(
                self,
                std::vector<FourStateEstimator::StateVector>(states.begin(),
                                                               states.end()),
                std::vector<FourStateEstimator::StateCovariance>(
                    covariances.begin(), covariances.end()),
                inputs);
          },
          "time"_a, "ax"_a, "yaw_rate"_a, "x"_a, "y"_a, "yaw"_a, "vg"_a,
          "ay"_a, "measurement_dt"_a,
          "Runs estimate_from_series() followed by smooth_from_series() "
          "(sharing the same built `inputs`, rather than rebuilding them "
          "twice) and returns its (smoothed_states, smoothed_covariances). "
          "See estimate_from_series() for the argument semantics.");
}
