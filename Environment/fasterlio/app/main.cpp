#include <omp.h>
#include <filesystem>
#include "fast_evo/evo_ape.hpp"
namespace py = pybind11;
namespace fs = std::filesystem;
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowVectorXf = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
class ParallelEnvironment {
   public:
    ParallelEnvironment(int num_envs, std::string config_file, std::string bag_dir, std::string ref_traj)
        : config_file_(config_file), config_(YAML::LoadFile(config_file)), bag_nums_(0) {
#ifndef _OPENMP
        std::cout << "Error finding _OPENMP macro, aborting..." << std::endl;
        throw std::runtime_error("Error finding _OPENMP macro");
#endif
        std::cout << "openmp enabled, set thread num: " << num_envs << std::endl;
        envs_.resize(num_envs);
        for (auto& env : envs_) {
            env.loadRefTraj(ref_traj);
        }
        std::cout << "Reference trajectory loaded" << std::endl;
        if (!fs::exists(bag_dir)) {
            std::cout << "Bag directory not found, aborting..." << std::endl;
            throw std::runtime_error("Bag directory not found");
        } else {
            for (const auto& entry : fs::directory_iterator(bag_dir)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    if (filename.find("chunk_") == 0 && filename.compare(filename.size() - 4, 4, ".bag") == 0) {
                        bag_names_.push_back(entry.path().string());
                    }
                }
            }
            init_bag_ = fs::path(bag_dir).append("init.bag").string();
            std::cout << "Found " << bag_names_.size() << " bag files" << std::endl;
        }
        std::cout << "Parallel Environment Built" << std::endl;
    }
    Eigen::VectorXf parallelRunOdomCalcErr(Eigen::Ref<RowMatrixXf> extr_matrix, bool terminate = false,
                                           float terminate_ratio = 0.2, bool use_imu = false) {
        Eigen::VectorXf total_errs(envs_.size());
#pragma omp parallel for
        for (int i = 0; i < envs_.size(); i++) {
            // std::cout << "thread num: " << omp_get_thread_num() << std::endl;
            Eigen::VectorXf extr = extr_matrix.row(i);
            auto bag_idx = (omp_get_thread_num() + bag_nums_ * envs_.size()) % bag_names_.size();
            auto bag_name = bag_names_[bag_idx];
            // total_errs(i) = bag_idx;
            envs_[i].runOdomCalcErr(config_file_, bag_name, extr, terminate, terminate_ratio, use_imu);
            total_errs(i) = std::get<2>(envs_[i].total_err_);
        }
        bag_nums_++;
        return total_errs;
    }
    double runOdomCalcErr(Eigen::Ref<Eigen::VectorXf> extr, std::string bag_file, bool terminate = false,
                          float terminate_ratio = 0.2, bool use_imu = false) {
        return envs_[0].runOdomCalcErr(config_file_, bag_file, extr, terminate, terminate_ratio, use_imu);
        // std::cout << extr << endl;
    }

   private:
    std::vector<Environment> envs_;
    std::string config_file_;
    // std::string bag_dir_;
    // std::string ref_traj_;
    std::vector<std::string> bag_names_;
    std::string init_bag_;
    YAML::Node config_;
    std::atomic<size_t> bag_nums_;
};

int main(int argc, char const *argv[]) {
    std::string config_file = argv[1];
    std::string bag_dir = argv[2];
    std::string ref_traj = argv[3];
    int num_envs = std::stoi(argv[4]);

    ParallelEnvironment pe(num_envs, config_file, bag_dir, ref_traj);
    Eigen::MatrixXf extr_matrix(4, 7);
    extr_matrix << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1;
    // pe.parallelRunOdomCalcErr(extr_matrix, false, 0.2, false);
}