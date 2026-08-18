#include "5_state_estimator.hpp"

#include <cmath>

FiveStateEstimator::FiveStateEstimator() {
  filter_.setStateUpdateFunction(&FiveStateEstimator::stateFunction);
  filter_.setStateJacobian(&FiveStateEstimator::stateJacobian);
  filter_.setMeasurementFunction(&FiveStateEstimator::measurementFunction);
  filter_.setMeasurementJacobian(&FiveStateEstimator::measurementJacobian);

  filter_.setState(StateVector::Zero());
  filter_.setStateCovariance(StateCovariance::Identity());
  filter_.setProcessCovariance(StateCovariance::Identity());
  filter_.setMeasurementCovariance(
      Eigen::MatrixXd::Identity(NUM_MEASUREMENTS, NUM_MEASUREMENTS));
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
  Eigen::VectorXd input(NUM_INPUTS);
  input(INPUT_AX) = ax;
  input(INPUT_AY) = ay;
  input(INPUT_YAW_RATE) = yawRate;
  input(INPUT_DT) = dt;
  filter_.predict(input);
}

void FiveStateEstimator::correct(double x, double y, double yaw, double vg) {
  Eigen::VectorXd measurement(NUM_MEASUREMENTS);
  measurement(MEAS_X) = x;
  measurement(MEAS_Y) = y;
  measurement(MEAS_YAW) = yaw;
  measurement(MEAS_VG) = vg;
  filter_.update(measurement);
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
