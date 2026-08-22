#include "4_state_estimator.hpp"

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
constexpr double kDefaultVgVar = 1.0;          // [(m/s)^2]

// Process noise, added once per predict() call (not scaled by dt).
constexpr double kDefaultPositionProcessVar = 1e-4;
constexpr double kDefaultYawProcessVar = 1e-5;
constexpr double kDefaultVgProcessVar = 1e-3;

// Measurement noise: consumer-grade GPS position/heading/speed, plus a
// consumer-grade MEMS IMU lateral-acceleration channel.
constexpr double kDefaultGpsPositionVar = 2.25; // [m^2], 1.5 m std
constexpr double kDefaultGpsYawVar = (2.0 * kPi / 180.0) * (2.0 * kPi / 180.0); // [rad^2], 2 deg std
constexpr double kDefaultGpsSpeedVar = 0.01;    // [(m/s)^2], 0.1 m/s std
constexpr double kDefaultAyVar = 0.0025;        // [(m/s^2)^2], 0.05 m/s^2 std

} // namespace

FourStateEstimator::FourStateEstimator() {
  filter_.setStateUpdateFunction(&FourStateEstimator::stateFunction);
  filter_.setStateJacobian(&FourStateEstimator::stateJacobian);
  filter_.setMeasurementFunction(&FourStateEstimator::measurementFunction);
  filter_.setMeasurementJacobian(&FourStateEstimator::measurementJacobian);

  filter_.setState(StateVector::Zero());

  StateCovariance P0 = StateCovariance::Zero();
  P0(STATE_X, STATE_X) = kDefaultPositionVar;
  P0(STATE_Y, STATE_Y) = kDefaultPositionVar;
  P0(STATE_YAW, STATE_YAW) = kDefaultYawVar;
  P0(STATE_VG, STATE_VG) = kDefaultVgVar;
  filter_.setStateCovariance(P0);

  StateCovariance Q0 = StateCovariance::Zero();
  Q0(STATE_X, STATE_X) = kDefaultPositionProcessVar;
  Q0(STATE_Y, STATE_Y) = kDefaultPositionProcessVar;
  Q0(STATE_YAW, STATE_YAW) = kDefaultYawProcessVar;
  Q0(STATE_VG, STATE_VG) = kDefaultVgProcessVar;
  filter_.setProcessCovariance(Q0);

  MeasurementCovariance R0 = MeasurementCovariance::Zero();
  R0(MEAS_X, MEAS_X) = kDefaultGpsPositionVar;
  R0(MEAS_Y, MEAS_Y) = kDefaultGpsPositionVar;
  R0(MEAS_YAW, MEAS_YAW) = kDefaultGpsYawVar;
  R0(MEAS_VG, MEAS_VG) = kDefaultGpsSpeedVar;
  R0(MEAS_AY, MEAS_AY) = kDefaultAyVar;
  filter_.setMeasurementCovariance(R0);

  // Seed a zero input so correct() has a valid (non-empty) input to read
  // `yawRate` from even if called before any real predict() -- see the
  // class doc comment. dt=0 makes this a no-op on state/covariance beyond
  // adding Q0 once (negligible, and overwritten by any subsequent
  // setState()/setStateCovariance() call).
  filter_.predict(makeInput(0.0, 0.0, 0.0));
}

void FourStateEstimator::setState(double x, double y, double yaw,
                                   double vg) {
  StateVector state;
  state(STATE_X) = x;
  state(STATE_Y) = y;
  state(STATE_YAW) = yaw;
  state(STATE_VG) = vg;
  filter_.setState(state);
}

void FourStateEstimator::setStateCovariance(const StateCovariance &P) {
  filter_.setStateCovariance(P);
}

void FourStateEstimator::setProcessNoise(const StateCovariance &Q) {
  filter_.setProcessCovariance(Q);
}

void FourStateEstimator::setMeasurementNoise(const MeasurementCovariance &R) {
  filter_.setMeasurementCovariance(R);
}

void FourStateEstimator::predict(double ax, double yawRate, double dt) {
  filter_.predict(makeInput(ax, yawRate, dt));
}

void FourStateEstimator::correct(double x, double y, double yaw, double vg,
                                  double ay) {
  filter_.update(makeMeasurement(x, y, yaw, vg, ay));
}

void FourStateEstimator::smooth(StateVectorList &states,
                                 StateCovarianceList &covariances,
                                 const std::vector<Eigen::VectorXd> &inputs) {
  filter_.RTSSmoother(states, covariances, inputs);
}

std::pair<FourStateEstimator::StateVectorList,
          FourStateEstimator::StateCovarianceList>
FourStateEstimator::estimate(const std::vector<Eigen::VectorXd> &measurements,
                              const std::vector<Eigen::VectorXd> &inputs) {
  StateVectorList states;
  StateCovarianceList covariances;
  states.reserve(measurements.size());
  covariances.reserve(measurements.size());

  for (size_t i = 0; i < measurements.size(); ++i) {
    if (i > 0) {
      filter_.predict(inputs[i - 1]);
    }
    if (measurements[i].size() == FourStateEstimator::NUM_MEASUREMENTS) {
      filter_.update(measurements[i]);
    }
    states.push_back(filter_.getState());
    covariances.push_back(filter_.getCovariance());
  }
  return {states, covariances};
}

Eigen::VectorXd FourStateEstimator::makeInput(double ax, double yawRate,
                                               double dt) {
  Eigen::VectorXd input(NUM_INPUTS);
  input(INPUT_AX) = ax;
  input(INPUT_YAW_RATE) = yawRate;
  input(INPUT_DT) = dt;
  return input;
}

Eigen::VectorXd FourStateEstimator::makeMeasurement(double x, double y,
                                                     double yaw, double vg,
                                                     double ay) {
  Eigen::VectorXd measurement(NUM_MEASUREMENTS);
  measurement(MEAS_X) = x;
  measurement(MEAS_Y) = y;
  measurement(MEAS_YAW) = yaw;
  measurement(MEAS_VG) = vg;
  measurement(MEAS_AY) = ay;
  return measurement;
}

Eigen::VectorXd FourStateEstimator::stateFunction(const Eigen::VectorXd &state,
                                                   const Eigen::VectorXd &input,
                                                   void * /*userData*/) {
  const double yaw = state(STATE_YAW);
  const double vg = state(STATE_VG);

  const double ax = input(INPUT_AX);
  const double yawRate = input(INPUT_YAW_RATE);
  const double dt = input(INPUT_DT);

  const double cosYaw = std::cos(yaw);
  const double sinYaw = std::sin(yaw);

  Eigen::VectorXd next(NUM_STATES);
  next(STATE_X) = state(STATE_X) + vg * cosYaw * dt;
  next(STATE_Y) = state(STATE_Y) + vg * sinYaw * dt;
  next(STATE_YAW) = yaw + yawRate * dt;
  next(STATE_VG) = vg + ax * dt;
  return next;
}

Eigen::MatrixXd FourStateEstimator::stateJacobian(const Eigen::VectorXd &state,
                                                   const Eigen::VectorXd &input,
                                                   void * /*userData*/) {
  const double yaw = state(STATE_YAW);
  const double vg = state(STATE_VG);
  const double dt = input(INPUT_DT);

  const double cosYaw = std::cos(yaw);
  const double sinYaw = std::sin(yaw);

  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(NUM_STATES, NUM_STATES);
  F(STATE_X, STATE_YAW) = -vg * sinYaw * dt;
  F(STATE_X, STATE_VG) = cosYaw * dt;
  F(STATE_Y, STATE_YAW) = vg * cosYaw * dt;
  F(STATE_Y, STATE_VG) = sinYaw * dt;
  return F;
}

Eigen::VectorXd
FourStateEstimator::measurementFunction(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd &input,
                                         void * /*userData*/) {
  Eigen::VectorXd z(NUM_MEASUREMENTS);
  z(MEAS_X) = state(STATE_X);
  z(MEAS_Y) = state(STATE_Y);
  z(MEAS_YAW) = state(STATE_YAW);
  z(MEAS_VG) = state(STATE_VG);
  z(MEAS_AY) = state(STATE_VG) * input(INPUT_YAW_RATE);
  return z;
}

Eigen::MatrixXd
FourStateEstimator::measurementJacobian(const Eigen::VectorXd & /*state*/,
                                         const Eigen::VectorXd &input,
                                         void * /*userData*/) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(NUM_MEASUREMENTS, NUM_STATES);
  H(MEAS_X, STATE_X) = 1.0;
  H(MEAS_Y, STATE_Y) = 1.0;
  H(MEAS_YAW, STATE_YAW) = 1.0;
  H(MEAS_VG, STATE_VG) = 1.0;
  H(MEAS_AY, STATE_VG) = input(INPUT_YAW_RATE);
  return H;
}
