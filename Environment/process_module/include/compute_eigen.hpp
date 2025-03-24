#include <Eigen/Eigen>
#include <filesystem>
#include <iostream>
#include <string>

#include <ceres/ceres.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>

#include <yaml-cpp/yaml.h>

#include "trajectory/trajectory_estimator.hpp"

struct TrajData {
  double timestamp;
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
};

template <int _N> class DataSufficiency {

public:
  DataSufficiency() = default;

  void loadTraj(const std::string &traj_file) {
    traj_data_.clear();

    if (!std::filesystem::exists(traj_file)) {
      std::cout << "trajectory file can not be found \"" << traj_file
                << "\", aborting...\"" << std::endl;
      throw std::runtime_error("File not found");
    }

    std::ifstream traj_file_stream{traj_file};
    std::string line;
    std::vector<double> timestamps, xyz, quaternion;
    while (std::getline(traj_file_stream, line)) {
      std::istringstream iss(line);
      TrajData data;
      iss >> data.timestamp >> data.position.x() >> data.position.y() >>
          data.position.z() >> data.orientation.x() >> data.orientation.y() >>
          data.orientation.z() >> data.orientation.w();
      traj_data_.emplace_back(data);
    }
  }
  void buildTrajectory() {
    trajectory_ = std::make_shared<basalt::Se3Spline<_N>>(
        traj_data_[1].timestamp - traj_data_[0].timestamp,
        traj_data_[0].timestamp);
    trajectory_estimator_ =
        std::make_unique<estimator::TrajectoryEstimator<_N>>(trajectory_);
    trajectory_->extendKnotsTo(traj_data_.back().timestamp,
                               Sophus::SE3<double>());
    std::cout << "initializing trajectory..." << std::endl;
    std::cout << "adding lidar-odometry measurements..." << std::endl;
    for (int i = 0; i < traj_data_.size(); i++) {
      PoseData pose_data;
      pose_data.timestamp = traj_data_[i].timestamp;
      pose_data.position = traj_data_[i].position;
      pose_data.orientation = SO3d(traj_data_[i].orientation);
      trajectory_estimator_->AddPoseMeasurement(pose_data, 0.5, 0.5);
    }
    std::cout << "Done , Solving Trajectory..." << std::endl;
    auto summary = trajectory_estimator_->Solve(100, false, 0);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "[Done Building Trajectory]" << std::endl;
  }

  void dump_traj() {
    std::filesystem::path path = std::filesystem::current_path();
    path += "/log/built_traj.txt";
    if (!std::filesystem::exists(path)) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream traj_file_stream(path);
    std::cout << "dump traj to " << path << std::endl;
    for (auto data : traj_data_) {
      auto pose = trajectory_->pose(data.timestamp).translation();
      auto orien =
          trajectory_->pose(data.timestamp).so3().unit_quaternion().coeffs();
      traj_file_stream << std::setprecision(12) << data.timestamp << " "
                       << pose.x() << " " << pose.y() << " " << pose.z() << " "
                       << orien.x() << " " << orien.y() << " " << orien.z()
                       << " " << orien.w() << std::endl;
    }
  }
  auto EigenDecomposition(double begin_time) {
    Eigen::Matrix<double, 2000, 3> jacobian;
    // dump lidar_omg
    // std::filesystem::path path = std::filesystem::current_path();
    // path += "/log/lidar_omg" + std::to_string(begin_time) + ".txt";
    // std::ofstream traj_file_stream(path);

    auto start_time = traj_data_[0].timestamp + begin_time;
    auto end_time = start_time + 15;
    auto jaco_index = 0;
    for (int i = 0; i < traj_data_.size(); i++) {
      if(traj_data_[i].timestamp < start_time || traj_data_[i].timestamp > end_time) continue;
      auto lidar_omg = trajectory_->rotVelBody(traj_data_[i].timestamp);
      // traj_file_stream << std::setprecision(12) << traj_data_[i].timestamp << " "
                      //  << lidar_omg.x() << " " << lidar_omg.y() << " " << lidar_omg.z() << std::endl;
      Eigen::Matrix3d lidar_omg_skew = Sophus::SO3d::hat(lidar_omg);
      jacobian.block<3, 3>(jaco_index++, 0) << lidar_omg_skew;
      if (jaco_index >= 1998) {
        break;
      }
    }
    Eigen::Matrix3d Hessian_rot = jacobian.transpose() * jacobian;
    Eigen::EigenSolver<Eigen::Matrix3d> es(Hessian_rot);
    auto EigenValue = es.eigenvalues().real();
    auto Scaled_Eigen = EigenValue / 100;
    Eigen::Vector3d Rot_percent(Scaled_Eigen[1] * Scaled_Eigen[2],
                                Scaled_Eigen[0] * Scaled_Eigen[2],
                                Scaled_Eigen[0] * Scaled_Eigen[1]);
    return Rot_percent.sum();
  }

private:
  std::deque<TrajData> traj_data_;
  std::unique_ptr<estimator::TrajectoryEstimator<_N>> trajectory_estimator_;
  std::shared_ptr<basalt::Se3Spline<_N>> trajectory_;
};
