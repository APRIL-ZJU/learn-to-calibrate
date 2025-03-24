#pragma once

#include <basalt/spline/ceres_spline_helper.h>
#include <basalt/spline/ceres_spline_helper_jet.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/spline/spline_segment.h>
#include <ceres/ceres.h>
#include "utils/imu_data.h"
#include <sophus/so3.hpp>

#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

namespace estimator {

template <int _N>
class GyroCalibFactor  {
private: 
  std::shared_ptr<basalt::Se3Spline<_N>> traj_;
  IMUData imu_data_;

public:
  GyroCalibFactor(std::shared_ptr<basalt::Se3Spline<_N>> traj, const IMUData& imu_data) :
                  traj_(traj), imu_data_(imu_data){}
  template <class T>
  bool operator()(const T* const params, T* Residuals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    // Eigen::Map<const Vec3T> transVec(params);
    Eigen::Map<const Eigen::Quaternion<T>> rotQua(params);
    Vec3T rotVelSpline = traj_->rotVelBody(imu_data_.timestamp).template cast<T>();
    Vec3T gyro_bias(T(1.9393e-05),T(1.9393e-05),T(1.9393e-05));
    Vec3T rot = rotQua * rotVelSpline +  gyro_bias;
    Eigen::Map<Vec3T> error(Residuals);
    error = rot - imu_data_.gyro;
    return true;
  }
};

template <int _N>
class AccelCalibFactor  {
private: 
  std::shared_ptr<basalt::Se3Spline<_N>> traj_;
  IMUData imu_data_;

public:
  AccelCalibFactor(std::shared_ptr<basalt::Se3Spline<_N>> traj, const IMUData& imu_data) :
                  traj_(traj), imu_data_(imu_data){}
  template <class T>
  bool operator()(const T* const params, T* Residuals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    Eigen::Map<const Vec3T> transVec(params) ; // r_b
    Eigen::Map<const Eigen::Quaternion<T>> rotQua(params + 3); // C_i_b
    Vec3T transAccelSpline = traj_->pos_spline.acceleration(imu_data_.timestamp).template cast<T>(); // a_w
    auto orientation = traj_->pose(imu_data_.timestamp).so3().inverse(); // C_b_w
    Vec3T rotVelSpline = traj_->rotVelBody(imu_data_.timestamp).template cast<T>(); // w_b
    Vec3T rotAccSpline = traj_->so3_spline.accelerationBody(imu_data_.timestamp).template cast<T>(); // w_dot_b
    Vec3T g(T(0),T(0),T(9.81));
    Vec3T acc = rotQua * (orientation * (transAccelSpline - g) + \
                          rotAccSpline.cross(transVec) + rotVelSpline.cross(rotVelSpline.cross(transVec)));
    // Eigen::Map<Vec3T> error(Residuals);
    // error = acc - imu_data_.accel.template cast<T>();
    Residuals[0] = acc.x() - T(imu_data_.accel.x());
    Residuals[1] = acc.y() - T(imu_data_.accel.y());
    Residuals[2] = acc.z() - T(imu_data_.accel.z());
    return true;
  }

};

struct LidarCalibData {
  Eigen::Vector3d linear_acc_world;
  Eigen::Matrix3d R_L_L0;
  Eigen::Vector3d angle_vel_body;
  Eigen::Vector3d angle_acc_body;
  void print_values() {
    std::cout << "linear_acc_world: " << linear_acc_world.transpose() << std::endl;
    std::cout << "R_L_L0: \n" << R_L_L0 << std::endl;
    std::cout << "angle_vel_body: " << angle_vel_body.transpose() << std::endl;
    std::cout << "angle_acc_body: " << angle_acc_body.transpose() << std::endl;
  }
};

class GravCalibFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GravCalibFactor(const LidarCalibData& Lidar, const Eigen::Matrix3d& R_LI,
                  const Eigen::Vector3d& IMU_linear_acc, const Eigen::Vector3d& T_LI)
          : lidar_(Lidar), R_LI_(R_LI), IMU_linear_acc_(IMU_linear_acc), T_LI_(T_LI) {}

  template<typename T>
  bool operator()(const T *grav, T *residual) const {
      //Known parameters used for Residual Construction
      Eigen::Matrix<T, 3, 3> R_LL0_T = lidar_.R_L_L0.cast<T>();
      Eigen::Matrix<T, 3, 3> R_LI_T_transpose = R_LI_.transpose().cast<T>();
      Eigen::Matrix<T, 3, 1> IMU_linear_acc_T = IMU_linear_acc_.cast<T>();
      Eigen::Matrix<T, 3, 1> Lidar_linear_acc_T = lidar_.linear_acc_world.cast<T>();
      Eigen::Matrix<T, 3, 1> T_IL_;
      T_IL_ = R_LI_T_transpose * (-T_LI_.cast<T>());
      //Unknown Parameters, needed to be estimated
      // Eigen::Matrix<T, 3, 1> bias_aL{b_a[0], b_a[1], b_a[2]};    //Bias of Linear acceleration
      // Eigen::Matrix<T, 3, 1> T_IL{trans[0], trans[1], trans[2]}; //Translation of I-L (IMU wtr Lidar)
      // Eigen::Map<Eigen::Matrix<T, 3, 1>> Gravity(grav);
      Eigen::Matrix<T, 3, 1> Gravity{grav[0], grav[1], grav[2]};

      //Residual Construction
      Eigen::Matrix3d Lidar_omg_SKEW, Lidar_angacc_SKEW;
      Lidar_omg_SKEW << SKEW_SYM_MATRX(lidar_.angle_vel_body);
      Lidar_angacc_SKEW << SKEW_SYM_MATRX(lidar_.angle_acc_body);
      Eigen::Matrix3d Jacob_trans = Lidar_omg_SKEW * Lidar_omg_SKEW + Lidar_angacc_SKEW;
      Eigen::Matrix<T, 3, 3> Jacob_trans_T = Jacob_trans.cast<T>();


      Eigen::Matrix<T, 3, 1> resi = R_LL0_T * R_LI_T_transpose * IMU_linear_acc_T * 9.81
                                    + Gravity - Lidar_linear_acc_T - R_LL0_T * Jacob_trans_T * T_IL_;
      residual[0] = resi[0];
      residual[1] = resi[1];
      residual[2] = resi[2];
      return true;
  }

  LidarCalibData lidar_;
  Eigen::Matrix3d R_LI_;
  Eigen::Vector3d IMU_linear_acc_;
  Eigen::Vector3d T_LI_;
};

} // namespace estimator