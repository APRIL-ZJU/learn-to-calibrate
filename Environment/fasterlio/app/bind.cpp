#include <omp.h>
#include <filesystem>
#include "fast_evo/evo_ape.hpp"
// #include <pybind11/pybind11.h>
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
                                           float terminate_ratio = 0.2, bool use_imu = false, double max_diff = 0.01) {
        Eigen::VectorXf total_errs(envs_.size());
#pragma omp parallel for
        for (int i = 0; i < envs_.size(); i++) {
            Eigen::VectorXf extr = extr_matrix.row(i);
            auto bag_idx = (omp_get_thread_num() + bag_nums_ * envs_.size()) % bag_names_.size();
            auto bag_name = bag_names_[bag_idx];
            envs_[i].runOdomCalcErr(config_file_, bag_name, extr, terminate, terminate_ratio, use_imu, max_diff);
            total_errs(i) = std::get<2>(envs_[i].total_err_);
        }
        bag_nums_++;
        // std::cout << "extr:\n" << extr_matrix << std::endl;
        return total_errs;
    }
    double runOdomCalcErr(Eigen::Ref<Eigen::VectorXf> extr, std::string bag_file, bool terminate = false,
                          float terminate_ratio = 0.2, bool use_imu = false, double max_diff = 0.01) {
        auto error = envs_[0].runOdomCalcErr(config_file_, bag_file, extr, terminate, terminate_ratio, use_imu, max_diff);
        return error;
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

PYBIND11_MODULE(L2CE, m) {
    py::class_<Environment>(m, "Environment")
        .def(py::init<>())
        // .def("setExtr", &Environment::setExtr)
        // .def("printExtr", &Environment::printExtr)
        // .def("loadRefTraj", &Environment::loadRefTraj)
        // .def("printRefTraj", &Environment::printRefTraj)
        // .def("loadEstTraj", &Environment::loadEstTraj)
        .def("dumpEstTraj", &Environment::dumpEstTraj)
        // .def("ApplyUmeyamaCalAPE", &Environment::ApplyUmeyamaCalAPE)
        .def("runOdomCalcErr", &Environment::runOdomCalcErr)
        .def_readonly("total_err_", &Environment::total_err_);

    py::class_<ParallelEnvironment>(m, "ParallelEnvironment")
        .def(py::init<int, std::string, std::string, std::string>())
        .def("parallelRunOdomCalcErr", &ParallelEnvironment::parallelRunOdomCalcErr)
        .def("runOdomCalcErr", &ParallelEnvironment::runOdomCalcErr);
}