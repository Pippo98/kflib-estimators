#pragma once

#include <kflib/ekf.hpp>

#include <utility>
#include <vector>

/**
 * @brief Extended Kalman Filter estimating a vehicle's planar pose and
 * ground speed under a no-lateral-velocity assumption, from IMU
 * (prediction) and pose+speed+lateral-acceleration (correction)
 * measurements.
 *
 * State: [x, y, yaw, vg]
 *   x, y - position in the world frame.
 *   yaw  - heading angle (zero => vehicle points along the world +x axis).
 *   vg   - ground speed along the heading direction (lateral velocity is
 *          assumed zero, unlike FiveStateEstimator's separate u/v).
 *
 * Prediction input: IMU measurements [ax, yawRate] integrated over dt with
 * the kinematic model:
 *   x_dot   = vg * cos(yaw)
 *   y_dot   = vg * sin(yaw)
 *   yaw_dot = yawRate
 *   vg_dot  = ax
 *
 * Correction measurement: [x, y, yaw, vg, ay], where ay = vg * yawRate is
 * the lateral acceleration implied by the no-lateral-velocity assumption.
 * `yawRate` here is the one from the input used at the *preceding*
 * predict() call (kflib's EKF evaluates the measurement function with the
 * last input passed to predict()) — so, as usual, call predict() then
 * correct() in that order. If correct() is called with no prior predict()
 * at all (e.g. estimate()'s first measurement), the constructor seeds a
 * zero (ax=0, yawRate=0, dt=0) input so that step doesn't crash; the `ay`
 * component of that one correction simply carries no information then
 * (both its residual and Jacobian row are evaluated at yawRate=0), which
 * is the physically correct behavior since no yaw rate has been observed
 * yet.
 */
class FourStateEstimator {
public:
  enum State { STATE_X = 0, STATE_Y, STATE_YAW, STATE_VG, NUM_STATES };
  enum Input { INPUT_AX = 0, INPUT_YAW_RATE, INPUT_DT, NUM_INPUTS };
  enum Measurement {
    MEAS_X = 0,
    MEAS_Y,
    MEAS_YAW,
    MEAS_VG,
    MEAS_AY,
    NUM_MEASUREMENTS
  };

  using Filter = ExtendedKalmanFilter<NUM_STATES>;
  using StateVector = Filter::StateVector;
  using StateCovariance = Filter::StateCovariance;
  using StateVectorList = Filter::StateVectorList;
  using StateCovarianceList = Filter::StateCovarianceList;
  using MeasurementCovariance =
      Eigen::Matrix<double, NUM_MEASUREMENTS, NUM_MEASUREMENTS>;

  /**
   * @brief Construct with default state/process/measurement covariances
   * (see the .cpp for the assumed sensor characteristics) so the filter can
   * run before the caller supplies characterized values.
   */
  FourStateEstimator();

  /** @brief Set the state vector [x, y, yaw, vg]. */
  void setState(double x, double y, double yaw, double vg);
  /** @brief Set the state covariance matrix P (4x4). */
  void setStateCovariance(const StateCovariance &P);
  /** @brief Set the process noise covariance matrix Q (4x4). */
  void setProcessNoise(const StateCovariance &Q);
  /** @brief Set the measurement noise covariance matrix R (5x5). */
  void setMeasurementNoise(const MeasurementCovariance &R);

  /**
   * @brief Predict step: integrate IMU measurements over dt.
   * @param ax IMU longitudinal acceleration (positive increases vg).
   * @param yawRate IMU yaw rate (positive increases yaw).
   * @param dt Integration time step [s].
   */
  void predict(double ax, double yawRate, double dt);

  /**
   * @brief Correction step with an absolute pose + speed + lateral-
   * acceleration measurement.
   * @param x Measured x position.
   * @param y Measured y position.
   * @param yaw Measured heading.
   * @param vg Measured ground speed.
   * @param ay Measured lateral acceleration, expected to equal vg * yawRate
   *   (yawRate from the preceding predict() call; see the class doc).
   */
  void correct(double x, double y, double yaw, double vg, double ay);

  /**
   * @brief Rauch-Tung-Striebel smoother: given a chronological sequence of
   * filtered states and covariances (e.g. recorded across successive
   * predict()+correct() calls) and the input used at each predict() call
   * (see makeInput()), smooths them in place using future information.
   * @param states In/out state estimates, one per timestep.
   * @param covariances In/out state covariances, one per timestep.
   * @param inputs Input vector used to predict from timestep i to i+1;
   *   must have `states.size() - 1` entries.
   */
  void smooth(StateVectorList &states, StateCovarianceList &covariances,
              const std::vector<Eigen::VectorXd> &inputs);

  /**
   * @brief Run the filter over a whole recorded sequence: correct() with
   * measurements[0] first (no predict before it), then alternate
   * predict(inputs[i]) / correct(measurements[i + 1]) for the rest. An
   * entry of `measurements` that isn't a NUM_MEASUREMENTS-sized vector
   * (e.g. a default-constructed empty Eigen::VectorXd) is treated as "no
   * measurement at this timestep": that step still contributes its
   * predict(), but the correct() is skipped, and the current (prediction-
   * only) state/covariance is still recorded for that entry. Overwrites
   * the estimator's current state/covariance as it goes (set them
   * beforehand via setState()/setStateCovariance() to seed the prior used
   * for measurements[0]).
   * @param measurements Measurement vectors [x, y, yaw, vg, ay] (see
   *   makeMeasurement()), chronologically ordered; entries not of size
   *   NUM_MEASUREMENTS are skipped (see above).
   * @param inputs Input vectors [ax, yawRate, dt] (see makeInput()) used
   *   to predict from measurements[i] to measurements[i + 1]; must have
   *   `measurements.size() - 1` entries.
   * @return One filtered state/covariance per measurement, in the same
   *   layout smooth() expects for its `states`/`covariances` arguments
   *   (with these same `inputs`), so the two chain directly.
   */
  std::pair<StateVectorList, StateCovarianceList>
  estimate(const std::vector<Eigen::VectorXd> &measurements,
           const std::vector<Eigen::VectorXd> &inputs);

  /** @brief Build an EKF input vector [ax, yawRate, dt], as consumed by
   * predict() and by the `inputs` list passed to smooth()/estimate(). */
  static Eigen::VectorXd makeInput(double ax, double yawRate, double dt);

  /** @brief Build a measurement vector [x, y, yaw, vg, ay], as consumed by
   * correct() and by the `measurements` list passed to estimate(). */
  static Eigen::VectorXd makeMeasurement(double x, double y, double yaw,
                                          double vg, double ay);

  const StateVector &getState() const { return filter_.getState(); }
  const StateCovariance &getCovariance() const {
    return filter_.getCovariance();
  }

private:
  static Eigen::VectorXd stateFunction(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &input,
                                        void *userData);
  static Eigen::MatrixXd stateJacobian(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &input,
                                        void *userData);
  static Eigen::VectorXd measurementFunction(const Eigen::VectorXd &state,
                                              const Eigen::VectorXd &input,
                                              void *userData);
  static Eigen::MatrixXd measurementJacobian(const Eigen::VectorXd &state,
                                              const Eigen::VectorXd &input,
                                              void *userData);

  Filter filter_;
};
