// Demo: 5-state vehicle estimator (x, y, yaw, u, v).
// Predict step integrates noisy IMU measurements (ax, ay, yawRate);
// correction step fuses a noisy [x, y, yaw, vg] measurement.

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <utility>

#include "5_state_estimator.hpp"

int main() {
  const double dt = 0.01;        // IMU rate: 100 Hz
  const int correctEvery = 10;   // pose+speed correction at 10 Hz

  FiveStateEstimator estimator;
  estimator.setState(0.0, 0.0, 0.0, 1.0, 0.0);
  estimator.setStateCovariance(FiveStateEstimator::StateCovariance::Identity() *
                                0.1);

  FiveStateEstimator::StateCovariance Q =
      FiveStateEstimator::StateCovariance::Identity() * 1e-4;
  estimator.setProcessNoise(Q);

  Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
  R(FiveStateEstimator::MEAS_X, FiveStateEstimator::MEAS_X) = 0.25;
  R(FiveStateEstimator::MEAS_Y, FiveStateEstimator::MEAS_Y) = 0.25;
  R(FiveStateEstimator::MEAS_YAW, FiveStateEstimator::MEAS_YAW) =
      (1.0 * std::numbers::pi / 180.0) * (1.0 * std::numbers::pi / 180.0);
  R(FiveStateEstimator::MEAS_VG, FiveStateEstimator::MEAS_VG) = 0.04;
  estimator.setMeasurementNoise(R);

  // True vehicle state, integrated with the same kinematic model.
  double x = 0.0, y = 0.0, yaw = 0.0, u = 1.0, v = 0.0;

  // History recorded for the RTS smoother: one filtered state/covariance
  // per timestep, and the input used to predict from timestep i to i+1.
  FiveStateEstimator::StateVectorList filteredStates = {estimator.getState()};
  FiveStateEstimator::StateCovarianceList filteredCovariances = {
      estimator.getCovariance()};
  std::vector<Eigen::VectorXd> inputsHistory;
  std::vector<std::pair<double, double>> truePositions = {{x, y}};

  std::mt19937 rng(12345);
  std::normal_distribution<double> imuAccNoise(0.0, 0.05);
  std::normal_distribution<double> imuGyroNoise(0.0, 0.01);
  std::normal_distribution<double> posNoise(0.0, std::sqrt(R(0, 0)));
  std::normal_distribution<double> yawNoise(0.0, std::sqrt(R(2, 2)));
  std::normal_distribution<double> vgNoise(0.0, std::sqrt(R(3, 3)));

  std::printf("%5s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s\n", "t",
              "true_x", "true_y", "true_yaw", "est_x", "est_y", "est_yaw",
              "est_u", "est_v");

  const int steps = 2000;
  for (int i = 0; i < steps; ++i) {
    const double t = i * dt;

    // True control inputs: mild acceleration and a slow S-curve turn.
    const double trueAx = 0.2 * std::sin(0.05 * t);
    const double trueAy = 0.0;
    const double trueYawRate = 0.3 * std::sin(0.1 * t);

    // Integrate ground truth with the same kinematic model.
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);
    const double xDot = u * cosYaw - v * sinYaw;
    const double yDot = u * sinYaw + v * cosYaw;
    x += xDot * dt;
    y += yDot * dt;
    yaw += trueYawRate * dt;
    u += (trueAx + trueYawRate * v) * dt;
    v += (trueAy - trueYawRate * u) * dt;

    // Noisy IMU measurement drives the predict step.
    const double measAx = trueAx + imuAccNoise(rng);
    const double measAy = trueAy + imuAccNoise(rng);
    const double measYawRate = trueYawRate + imuGyroNoise(rng);
    estimator.predict(measAx, measAy, measYawRate, dt);
    inputsHistory.push_back(
        FiveStateEstimator::makeInput(measAx, measAy, measYawRate, dt));

    if (i % correctEvery == correctEvery - 1) {
      const double measX = x + posNoise(rng);
      const double measY = y + posNoise(rng);
      const double measYaw = yaw + yawNoise(rng);
      const double measVg = std::hypot(u, v) + vgNoise(rng);
      estimator.correct(measX, measY, measYaw, measVg);
    }

    filteredStates.push_back(estimator.getState());
    filteredCovariances.push_back(estimator.getCovariance());
    truePositions.push_back({x, y});

    if (i % 50 == 0) {
      const auto &state = estimator.getState();
      std::printf(
          "%5.2f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, "
          "%10.4f\n",
          t, x, y, yaw, state(FiveStateEstimator::STATE_X),
          state(FiveStateEstimator::STATE_Y),
          state(FiveStateEstimator::STATE_YAW),
          state(FiveStateEstimator::STATE_U),
          state(FiveStateEstimator::STATE_V));
    }
  }

  auto positionRmse = [&](const FiveStateEstimator::StateVectorList &states) {
    double sumSq = 0.0;
    for (size_t i = 0; i < states.size(); ++i) {
      const double dx = states[i](FiveStateEstimator::STATE_X) - truePositions[i].first;
      const double dy = states[i](FiveStateEstimator::STATE_Y) - truePositions[i].second;
      sumSq += dx * dx + dy * dy;
    }
    return std::sqrt(sumSq / states.size());
  };
  std::printf("\nfiltered position RMSE: %.4f m\n", positionRmse(filteredStates));

  FiveStateEstimator::StateVectorList smoothedStates = filteredStates;
  FiveStateEstimator::StateCovarianceList smoothedCovariances = filteredCovariances;
  estimator.smooth(smoothedStates, smoothedCovariances, inputsHistory);
  std::printf("smoothed position RMSE: %.4f m\n", positionRmse(smoothedStates));

  return 0;
}
