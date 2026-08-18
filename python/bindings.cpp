#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

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
                   "use with the `inputs` argument of smooth().")
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
