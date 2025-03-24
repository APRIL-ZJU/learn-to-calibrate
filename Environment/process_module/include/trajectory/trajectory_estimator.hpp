#define FMT_HEADER_ONLY

#include "factor/mocap_pose_factor.hpp"
#include <basalt/spline/se3_spline.h>

#include <fstream>
#include <thread>

namespace estimator {

template <int _N>
class TrajectoryEstimator {
public:
  TrajectoryEstimator(std::shared_ptr<basalt::Se3Spline<_N>> oth)
      : trajectory_(oth) {
    problem_ = std::make_shared<ceres::Problem>(DefaultProblemOptions());
  }
  static ceres::Problem::Options DefaultProblemOptions() {
    ceres::Problem::Options options;
    options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    return options;
  }

  template <typename pose_type>
  void AddPoseMeasurement(const pose_type &pose_data, double rot_weight,
                          double pos_weight) {
    basalt::SplineMeta<_N> spline_meta;
    trajectory_->CalculateSplineMeta(
        {{pose_data.timestamp, pose_data.timestamp}}, spline_meta);

    using Functor = PoseFactor<_N>;
    Functor *functor =
        new Functor(pose_data, spline_meta, rot_weight, pos_weight);
    auto *cost_function =
        new ceres::DynamicAutoDiffCostFunction<Functor>(functor);

    /// add so3 knots
    for (int i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(4);
    }
    /// add vec3 knots
    for (int i = 0; i < spline_meta.NumParameters(); i++) {
      cost_function->AddParameterBlock(3);
    }

    cost_function->SetNumResiduals(6);

    std::vector<double *> vec;
    AddControlPoints(spline_meta, vec, false);
    AddControlPoints(spline_meta, vec, true);

    problem_->AddResidualBlock(cost_function, NULL, vec);
  }
  ceres::Solver::Summary Solve(int max_iterations, bool progress,
                               int num_threads) {
    ceres::Solver::Options options;

    options.minimizer_type = ceres::TRUST_REGION;
    //  options.gradient_tolerance = 0.01 *
    //  Sophus::Constants<double>::epsilon(); options.function_tolerance = 0.01
    //  * Sophus::Constants<double>::epsilon();
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //    options.trust_region_strategy_type = ceres::DOGLEG;
    //    options.dogleg_type = ceres::SUBSPACE_DOGLEG;

    //    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    options.minimizer_progress_to_stdout = progress;

    if (num_threads < 1) {
      num_threads = std::thread::hardware_concurrency();
    }
    options.num_threads = num_threads;
    options.max_num_iterations = max_iterations;

    if (callbacks_.size() > 0) {
      for (auto &cb : callbacks_) {
        options.callbacks.push_back(cb.get());
      }

      if (callback_needs_state_)
        options.update_state_every_iteration = true;
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem_.get(), &summary);

    // update state
    // calib_param_->UpdateExtrinicParam();
    // calib_param_->UpdateGravity();
    //  getCovariance();
    return summary;
  }
  void AddControlPoints(const basalt::SplineMeta<_N> &spline_meta,
                        std::vector<double *> &vec, bool addPosKont) {
    for (auto const &seg : spline_meta.segments) {
      size_t start_idx = trajectory_->computeTIndex(seg.t0 + 1e-9).second;
      for (size_t i = start_idx; i < (start_idx + seg.NumParameters()); ++i) {
        if (addPosKont) {
          vec.emplace_back(trajectory_->getKnotPos(i).data());
          problem_->AddParameterBlock(trajectory_->getKnotPos(i).data(), 3);
        } else {
          vec.emplace_back(trajectory_->getKnotSO3(i).data());
          problem_->AddParameterBlock(trajectory_->getKnotSO3(i).data(), 4);
        }
      }
    }
  }
  std::shared_ptr<basalt::Se3Spline<_N>> getTraj() { return trajectory_; }

private:
  std::shared_ptr<basalt::Se3Spline<_N>> trajectory_;
  std::shared_ptr<ceres::Problem> problem_;
  // ceres::Manifold* local_parameterization;

  bool traj_locked_;

  int fixed_control_point_index_ = -1;

  bool callback_needs_state_;
  std::vector<std::unique_ptr<ceres::IterationCallback>> callbacks_;
};
} // namespace estimator