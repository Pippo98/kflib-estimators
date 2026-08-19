#include "5_state_estimator.hpp"

#include <cmath>
#include <numbers>

namespace {

constexpr double kPi = std::numbers::pi;

// Ballpark defaults so the filter can run before the caller supplies
// characterized covariances, roughly matching consumer-grade automotive
// IMU + GPS sensor specs. Callers with actual sensor specs should override
// these via setStateCovariance()/setProcessNoise()/setMeasurementNoise().

// Initial state uncertainty.
constexpr double kDefaultPositionVar = 10.0;   // [m^2]
constexpr double kDefaultYawVar = (10.0 * kPi / 180.0) * (10.0 * kPi / 180.0); // [rad^2], 10 deg std
constexpr double kDefaultVelocityVar = 1.0;    // [(m/s)^2]

// Process noise, added once per predict() call (not scaled by dt).
constexpr double kDefaultPositionProcessVar = 1e-4;
constexpr double kDefaultYawProcessVar = 1e-5;
constexpr double kDefaultVelocityProcessVar = 1e-3;

// Measurement noise: consumer-grade GPS position/heading/speed.
constexpr double kDefaultGpsPositionVar = 2.25; // [m^2], 1.5 m std
constexpr double kDefaultGpsYawVar = (2.0 * kPi / 180.0) * (2.0 * kPi / 180.0); // [rad^2], 2 deg std
constexpr double kDefaultGpsSpeedVar = 0.01;    // [(m/s)^2], 0.1 m/s std

} // namespace

FiveStateEstimator::FiveStateEstimator() {
  filter_.setStateUpdateFunction(&FiveStateEstimator::stateFunction);
  filter_.setStateJacobian(&FiveStateEstimator::stateJacobian);
  filter_.setMeasurementFunction(&FiveStateEstimator::measurementFunction);
  filter_.setMeasurementJacobian(&FiveStateEstimator::measurementJacobian);

  filter_.setState(StateVector::Zero());

  StateCovariance P0 = StateCovariance::Zero();
  P0(STATE_X, STATE_X) = kDefaultPositionVar;
  P0(STATE_Y, STATE_Y) = kDefaultPositionVar;
  P0(STATE_YAW, STATE_YAW) = kDefaultYawVar;
  P0(STATE_U, STATE_U) = kDefaultVelocityVar;
  P0(STATE_V, STATE_V) = kDefaultVelocityVar;
  filter_.setStateCovariance(P0);

  StateCovariance Q0 = StateCovariance::Zero();
  Q0(STATE_X, STATE_X) = kDefaultPositionProcessVar;
  Q0(STATE_Y, STATE_Y) = kDefaultPositionProcessVar;
  Q0(STATE_YAW, STATE_YAW) = kDefaultYawProcessVar;
  Q0(STATE_U, STATE_U) = kDefaultVelocityProcessVar;
  Q0(STATE_V, STATE_V) = kDefaultVelocityProcessVar;
  filter_.setProcessCovariance(Q0);

  Eigen::Matrix4d R0 = Eigen::Matrix4d::Zero();
  R0(MEAS_X, MEAS_X) = kDefaultGpsPositionVar;
  R0(MEAS_Y, MEAS_Y) = kDefaultGpsPositionVar;
  R0(MEAS_YAW, MEAS_YAW) = kDefaultGpsYawVar;
  R0(MEAS_VG, MEAS_VG) = kDefaultGpsSpeedVar;
  filter_.setMeasurementCovariance(R0);
}

void FiveStateEstimator::setState(double x, double y, double yaw, double u,
                                   double v) {
  StateVector state;
  state(STATE_X) = x;
  state(STATE_Y) = y;
  state(STATE_YAW) = yaw;
  state(STATE_U) = u;
  state(STATE_V) = v;
  filter_.setState(state);
}

void FiveStateEstimator::setStateCovariance(const StateCovariance &P) {
  filter_.setStateCovariance(P);
}

void FiveStateEstimator::setProcessNoise(const StateCovariance &Q) {
  filter_.setProcessCovariance(Q);
}

void FiveStateEstimator::setMeasurementNoise(const Eigen::Matrix4d &R) {
  filter_.setMeasurementCovariance(R);
}

void FiveStateEstimator::predict(double ax, double ay, double yawRate,
                                  double dt) {
  filter_.predict(makeInput(ax, ay, yawRate, dt));
}

void FiveStateEstimator::correct(double x, double y, double yaw, double vg) {
  filter_.update(makeMeasurement(x, y, yaw, vg));
}

void FiveStateEstimator::smooth(StateVectorList &states,
                                 StateCovarianceList &covariances,
                                 const std::vector<Eigen::VectorXd> &inputs) {
  filter_.RTSSmoother(states, covariances, inputs);
}

std::pair<FiveStateEstimator::StateVectorList,
          FiveStateEstimator::StateCovarianceList>
FiveStateEstimator::estimate(const std::vector<Eigen::VectorXd> &measurements,
                              const std::vector<Eigen::VectorXd> &inputs) {
  StateVectorList states;
  StateCovarianceList covariances;
  states.reserve(measurements.size());
  covariances.reserve(measurements.size());

  for (size_t i = 0; i < measurements.size(); ++i) {
    if (i > 0) {
      filter_.predict(inputs[i - 1]);
    }
    if (measurements.size() == FiveStateEstimator::NUM_MEASUREMENTS) {
      filter_.update(measurements[i]);
    }
    states.push_back(filter_.getState());
    covariances.push_back(filter_.getCovariance());
  }
  return {states, covariances};
}

Eigen::VectorXd FiveStateEstimator::makeInput(double ax, double ay,
                                               double yawRate, double dt) {
  Eigen::VectorXd input(NUM_INPUTS);
  input(INPUT_AX) = ax;
  input(INPUT_AY) = ay;
  input(INPUT_YAW_RATE) = yawRate;
  input(INPUT_DT) = dt;
  return input;
}

Eigen::VectorXd FiveStateEstimator::makeMeasurement(double x, double y,
                                                     double yaw, double vg) {
  Eigen::VectorXd measurement(NUM_MEASUREMENTS);
  measurement(MEAS_X) = x;
  measurement(MEAS_Y) = y;
  measurement(MEAS_YAW) = yaw;
  measurement(MEAS_VG) = vg;
  return measurement;
}

Eigen::VectorXd FiveStateEstimator::stateFunction(const Eigen::VectorXd &state,
                                                   const Eigen::VectorXd &input,
                                                   void * /*userData*/) {
  const double yaw = state(STATE_YAW);
  const double u = state(STATE_U);
  const double v = state(STATE_V);

  const double ax = input(INPUT_AX);
  const double ay = input(INPUT_AY);
  const double yawRate = input(INPUT_YAW_RATE);
  const double dt = input(INPUT_DT);

  const double cosYaw = std::cos(yaw);
  const double sinYaw = std::sin(yaw);

  Eigen::VectorXd next(NUM_STATES);
  next(STATE_X) = state(STATE_X) + (u * cosYaw - v * sinYaw) * dt;
  next(STATE_Y) = state(STATE_Y) + (u * sinYaw + v * cosYaw) * dt;
  next(STATE_YAW) = yaw + yawRate * dt;
  next(STATE_U) = u + (ax + yawRate * v) * dt;
  next(STATE_V) = v + (ay - yawRate * u) * dt;
  return next;
}

Eigen::MatrixXd FiveStateEstimator::stateJacobian(const Eigen::VectorXd &state,
                                                   const Eigen::VectorXd &input,
                                                   void * /*userData*/) {
  const double yaw = state(STATE_YAW);
  const double u = state(STATE_U);
  const double v = state(STATE_V);

  const double yawRate = input(INPUT_YAW_RATE);
  const double dt = input(INPUT_DT);

  const double cosYaw = std::cos(yaw);
  const double sinYaw = std::sin(yaw);

  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(NUM_STATES, NUM_STATES);

  F(STATE_X, STATE_YAW) = (-u * sinYaw - v * cosYaw) * dt;
  F(STATE_X, STATE_U) = cosYaw * dt;
  F(STATE_X, STATE_V) = -sinYaw * dt;

  F(STATE_Y, STATE_YAW) = (u * cosYaw - v * sinYaw) * dt;
  F(STATE_Y, STATE_U) = sinYaw * dt;
  F(STATE_Y, STATE_V) = cosYaw * dt;

  F(STATE_U, STATE_V) = yawRate * dt;
  F(STATE_V, STATE_U) = -yawRate * dt;

  return F;
}

Eigen::VectorXd
FiveStateEstimator::measurementFunction(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd & /*input*/,
                                         void * /*userData*/) {
  Eigen::VectorXd z(NUM_MEASUREMENTS);
  z(MEAS_X) = state(STATE_X);
  z(MEAS_Y) = state(STATE_Y);
  z(MEAS_YAW) = state(STATE_YAW);
  z(MEAS_VG) = std::hypot(state(STATE_U), state(STATE_V));
  return z;
}

Eigen::MatrixXd
FiveStateEstimator::measurementJacobian(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd & /*input*/,
                                         void * /*userData*/) {
  const double u = state(STATE_U);
  const double v = state(STATE_V);
  const double vg = std::hypot(u, v);

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(NUM_MEASUREMENTS, NUM_STATES);
  H(MEAS_X, STATE_X) = 1.0;
  H(MEAS_Y, STATE_Y) = 1.0;
  H(MEAS_YAW, STATE_YAW) = 1.0;
  // Gradient of hypot(u, v) is undefined at vg == 0; leave the row zero
  // there rather than dividing by zero.
  if (vg > 1e-6) {
    H(MEAS_VG, STATE_U) = u / vg;
    H(MEAS_VG, STATE_V) = v / vg;
  }
  return H;
}
