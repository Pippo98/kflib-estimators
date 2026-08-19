#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "5_state_estimator.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(kflib_estimators, m) {
  m.doc() = "Python bindings for kflib-estimators' FiveStateEstimator "
            "(5-state vehicle EKF: x, y, yaw, u, v).";

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
            const Eigen::Index n = time.size();
            if (n == 0 || ax.size() != n || ay.size() != n ||
                yawRate.size() != n || x.size() != n || y.size() != n ||
                yaw.size() != n || vg.size() != n) {
              throw std::invalid_argument(
                  "time, ax, ay, yaw_rate, x, y, yaw, vg must all be "
                  "non-empty and the same length");
            }

            // One input per step from sample i to i+1: dt from consecutive
            // timestamps, IMU reading held from the start of the step.
            std::vector<Eigen::VectorXd> inputs;
            inputs.reserve(static_cast<size_t>(n - 1));
            for (Eigen::Index i = 0; i + 1 < n; ++i) {
              inputs.push_back(FiveStateEstimator::makeInput(
                  ax(i), ay(i), yawRate(i), time(i + 1) - time(i)));
            }

            // Correct only close to every `measurement_dt`, always
            // including the first sample; every other sample gets an
            // empty (skipped) measurement slot, per estimate()'s contract.
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
             const std::vector<FiveStateEstimator::StateVector> &statesIn,
             const std::vector<FiveStateEstimator::StateCovariance>
                 &covariancesIn,
             const std::vector<Eigen::VectorXd> &inputs) {
            FiveStateEstimator::StateVectorList states(statesIn.begin(),
                                                        statesIn.end());
            FiveStateEstimator::StateCovarianceList covariances(
                covariancesIn.begin(), covariancesIn.end());
            self.smooth(states, covariances, inputs);
            return std::make_tuple(
                std::vector<FiveStateEstimator::StateVector>(states.begin(),
                                                               states.end()),
                std::vector<FiveStateEstimator::StateCovariance>(
                    covariances.begin(), covariances.end()));
          },
          "states"_a, "covariances"_a, "inputs"_a,
          "Rauch-Tung-Striebel smoother over a recorded, chronologically "
          "ordered sequence of filtered states/covariances and the inputs "
          "used at each predict() call (see make_input()). Returns "
          "(smoothed_states, smoothed_covariances); does not mutate the "
          "estimator's current state.");
}
