#pragma once

#define FMT_HEADER_ONLY

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

static const double GRAVITY_NORM = -9.80;

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

struct IMUData {
  double timestamp;
  Eigen::Matrix<double, 3, 1> gyro;
  Eigen::Matrix<double, 3, 1> accel;
  SO3d orientation;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct IMUBias {
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d accel_bias;
};

struct IMUState {
  double timestamp;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
  Eigen::Quaterniond q;
  IMUBias bias;
  Eigen::Vector3d g;
  Eigen::Quaterniond gt_q;
};

struct PoseData {
  PoseData()
      : timestamp(0),
        position(Eigen::Vector3d(0, 0, 0)),
        orientation(SO3d(Eigen::Quaterniond::Identity())) {}

  double timestamp;
  Eigen::Vector3d position;
  SO3d orientation;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};