#pragma once

#include <basalt/spline/spline_segment.h>
#include <basalt/spline/ceres_spline_helper.h>
#include <basalt/spline/ceres_spline_helper_jet.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <utils/imu_data.h>
#include <sophus/so3.hpp>

namespace estimator {

template <int _N>
class PoseFactor : public basalt::CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
template <typename pose_type>
  PoseFactor(const pose_type& pose_data, const basalt::SplineMeta<_N>& spline_meta,
                double pos_weight, double rot_weight)
      : pose_data_(pose_data),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight),
        rot_weight_(rot_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    double u;
    spline_meta_.ComputeSplineIndex(pose_data_.timestamp, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_IkToG;
    basalt::CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    Vec3T p_IkinG;
    basalt::CeresSplineHelper<_N>::template evaluate<T, 3, 0>(sKnots + P_offset, u,
                                                      inv_dt_, &p_IkinG);

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (R_IkToG * pose_data_.orientation.inverse()).log();

    residuals.template block<3, 1>(3, 0) =
        T(pos_weight_) * (p_IkinG - pose_data_.position);
    return true;
  }

 private:
  PoseData pose_data_;
  basalt::SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
};
}  // namespace estimator