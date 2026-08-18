#pragma once

#include <kflib/ekf.hpp>

/**
 * @brief Extended Kalman Filter estimating a vehicle's planar pose and
 * body-frame velocities from IMU (prediction) and pose+speed (correction)
 * measurements.
 *
 * State: [x, y, yaw, u, v]
 *   x, y - position in the world frame.
 *   yaw  - heading angle (zero => vehicle points along the world +x axis).
 *   u    - longitudinal velocity in the vehicle frame (positive forward).
 *   v    - lateral velocity in the vehicle frame (positive left).
 *
 * Prediction input: IMU measurements [ax, ay, yawRate] integrated over dt
 * with the kinematic model:
 *   x_dot   = u * cos(yaw) - v * sin(yaw)
 *   y_dot   = u * sin(yaw) + v * cos(yaw)
 *   yaw_dot = yawRate
 *   u_dot   = ax + yawRate * v
 *   v_dot   = ay - yawRate * u
 *
 * Correction measurement: [x, y, yaw, vg], where vg = hypot(u, v) is the
 * velocity magnitude at the vehicle's center of mass.
 */
class FiveStateEstimator {
public:
  enum State { STATE_X = 0, STATE_Y, STATE_YAW, STATE_U, STATE_V, NUM_STATES };
  enum Input { INPUT_AX = 0, INPUT_AY, INPUT_YAW_RATE, INPUT_DT, NUM_INPUTS };
  enum Measurement { MEAS_X = 0, MEAS_Y, MEAS_YAW, MEAS_VG, NUM_MEASUREMENTS };

  using Filter = ExtendedKalmanFilter<NUM_STATES>;
  using StateVector = Filter::StateVector;
  using StateCovariance = Filter::StateCovariance;

  FiveStateEstimator();

  /** @brief Set the state vector [x, y, yaw, u, v]. */
  void setState(double x, double y, double yaw, double u, double v);
  /** @brief Set the state covariance matrix P (5x5). */
  void setStateCovariance(const StateCovariance &P);
  /** @brief Set the process noise covariance matrix Q (5x5). */
  void setProcessNoise(const StateCovariance &Q);
  /** @brief Set the measurement noise covariance matrix R (4x4). */
  void setMeasurementNoise(const Eigen::Matrix4d &R);

  /**
   * @brief Predict step: integrate IMU measurements over dt.
   * @param ax IMU longitudinal acceleration (positive increases u).
   * @param ay IMU lateral acceleration (positive increases v).
   * @param yawRate IMU yaw rate (positive increases yaw).
   * @param dt Integration time step [s].
   */
  void predict(double ax, double ay, double yawRate, double dt);

  /**
   * @brief Correction step with an absolute pose + speed measurement.
   * @param x Measured x position.
   * @param y Measured y position.
   * @param yaw Measured heading.
   * @param vg Measured velocity magnitude, hypot(u, v).
   */
  void correct(double x, double y, double yaw, double vg);

  const StateVector &getState() const { return filter_.getState(); }
  const StateCovariance &getCovariance() const { return filter_.getCovariance(); }

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
