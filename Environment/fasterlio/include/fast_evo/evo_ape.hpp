#include <filesystem>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <yaml-cpp/yaml.h>

#include "laser_mapping.h"
#include "utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using std::cout, std::endl;


struct PoseStemp {
  double timestamp;
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
};

class Environment {
public:
    Environment() {
        // std::cout << "env build" << std::endl;
        total_err_ = std::make_tuple(0, 0, 0);
        est_len_ = 0;
        init_mode_ = false;
    }

    void setExtr(Eigen::Ref<Eigen::VectorXf> extr) {
        this->extr = extr;
    }
    void printExtr() {
        Eigen::Vector3d ext(extr.head<3>().cast<double>());
        Eigen::Quaterniond exr_qua(extr.tail<4>().cast<double>());
        cout << ext << endl << exr_qua.coeffs() << endl;
        auto Lidar_T_wrt_IMU = ext;
        auto Lidar_R_wrt_IMU = exr_qua.toRotationMatrix();
        cout << "current t: \n" << Lidar_T_wrt_IMU.transpose() << endl;
        cout << "current R: \n" << Lidar_R_wrt_IMU << endl;
    }

    void loadRefTraj(std::string traj_file) {
        refTimeStamps_.clear();
        refXYZ_.clear();

        if(!std::filesystem::exists(traj_file)){
            cout << "reference trajectory file can not be found \"" << traj_file << "\", aborting...\"" << endl;
            throw std::runtime_error("File not found");
        }
        
        std::ifstream traj_file_stream{traj_file};
        std::string line;
        std::vector<double> timestamps, xyz, quaternion;
        // std::vector<PoseStemp> buffer;
        while (std::getline(traj_file_stream, line)) {
        std::istringstream iss(line);
        PoseStemp data;
        iss >> data.timestamp >>
               data.position.x() >> data.position.y() >> data.position.z() >> 
               data.orientation.x() >> data.orientation.y() >> data.orientation.z() >> data.orientation.w();
        // buffer.emplace_back(data);
        refTimeStamps_.emplace_back(data.timestamp);
        refXYZ_.emplace_back(data.position.x());
        refXYZ_.emplace_back(data.position.y());
        refXYZ_.emplace_back(data.position.z());
        }
        ref_len_ = refTimeStamps_.size();

        // cout << "Loaded " << ref_len_ << " reference poses" << endl;
    }

    void loadEstTraj(std::string traj_file) {
        estTimeStamps_.clear();
        estXYZ_.clear();
        
        if(std::filesystem::exists(traj_file)){
            cout << "Loading reference trajectory from " << traj_file << endl;
        } else {
            cout << "File not found \"" << traj_file << '\"' << endl;
        }
        
        std::ifstream traj_file_stream{traj_file};
        std::string line;
        std::vector<double> timestamps, xyz, quaternion;
        std::vector<PoseStemp> buffer;
        while (std::getline(traj_file_stream, line)) {
        std::istringstream iss(line);
        PoseStemp data;
        iss >> data.timestamp >>
               data.position.x() >> data.position.y() >> data.position.z() >> 
               data.orientation.x() >> data.orientation.y() >> data.orientation.z() >> data.orientation.w();
        buffer.emplace_back(data);
        estTimeStamps_.emplace_back(data.timestamp);
        estXYZ_.emplace_back(data.position.x());
        estXYZ_.emplace_back(data.position.y());
        estXYZ_.emplace_back(data.position.z());
        }
        this->est_len_ = buffer.size();

        cout << "Loaded " << this->est_len_ << " estimate poses" << endl;
    }

    void printRefTraj() {
        for(int i = 0; i < refTimeStamps_.size(); i++) {
            cout << refTimeStamps_[i] << " " << refXYZ_[3*i] << " " << refXYZ_[3*i+1] << " " << refXYZ_[3*i+2] << endl;
        }
    }
    void dumpEstTraj() {
        auto path = std::filesystem::current_path().string() + "/log_est_traj.txt";
        std::cout << "dumping est traj to " << path << std::endl;
        std::ofstream traj_file_stream{path};
        for(int i = 0; i < estTimeStamps_.size(); i++) {
            traj_file_stream << std::setprecision(16) << estTimeStamps_[i] << " " << estXYZ_[3*i] << " " << estXYZ_[3*i+1] << " " << estXYZ_[3*i+2] << " 0 0 0 1" << endl;
        }
    } 

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> alignTimeIdx(double max_diff = 0.01) {
        Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>> refXYZ(refXYZ_.data(), 3, ref_len_);
        Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>> estXYZ(estXYZ_.data(), 3, est_len_);
        // std::cout << "ref_len_: " << ref_len_ << " est_len_: " << est_len_ << std::endl;
        Eigen::Map<Eigen::VectorXd> refTimeStamps(refTimeStamps_.data(), ref_len_);
        Eigen::Map<Eigen::VectorXd> estTimeStamps(estTimeStamps_.data(), est_len_);

        std::vector<Eigen::Vector3d> ref_aligned, est_aligned;
        ref_aligned.reserve(est_len_);
        est_aligned.reserve(est_len_);

        for (int i = 0; i < est_len_; ++i) {
            double est_time = estTimeStamps(i);
            auto it = std::lower_bound(refTimeStamps.data(), refTimeStamps.data() + ref_len_, est_time);
            int idx = std::distance(refTimeStamps.data(), it);

            // Check the closest timestamp in the neighborhood, assume that the timestamps are sorted and est_len < ref_len 
            double min_diff = std::numeric_limits<double>::max();
            int best_idx = -1;
            for (int j = std::max(0, idx - 1); j < std::min(static_cast<int>(ref_len_), idx + 2); ++j) {
                double diff = std::abs(refTimeStamps(j) - est_time);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_idx = j;
                }
            }

            if (best_idx != -1 && min_diff < max_diff) {
                ref_aligned.push_back(refXYZ.col(best_idx));
                est_aligned.push_back(estXYZ.col(i));
            }
        }

        if (ref_aligned.empty()) {
            cout << "reference file size: " << refTimeStamps.size() << ", estimate file size: " << estTimeStamps.size() << endl;
            dumpEstTraj();
            throw std::runtime_error("find no matching timestamps");
        }

        Eigen::Matrix<double, 3, Eigen::Dynamic> refXYZ_aligned(3, ref_aligned.size());
        Eigen::Matrix<double, 3, Eigen::Dynamic> estXYZ_aligned(3, est_aligned.size());

        for (size_t i = 0; i < ref_aligned.size(); ++i) {
            refXYZ_aligned.col(i) = ref_aligned[i];
            estXYZ_aligned.col(i) = est_aligned[i];
        }

        // std::cout << "Aligned " << ref_aligned.size() << " poses" << std::endl;
        // std::cout << "refXYZ_aligned: " << refXYZ_aligned.cols() << " estXYZ_aligned: " << estXYZ_aligned.cols() << std::endl;

        return std::make_pair(refXYZ_aligned, estXYZ_aligned);
    }
    // void dumpaligned(const Eigen::Matrix<double, 3, Eigen::Dynamic>&refXYZ_aligned, const Eigen::Matrix<double, 3, Eigen::Dynamic>&estXYZ_aligned) {
        // assert(estXYZ_aligned.size() == refXYZ_aligned.size());
        // auto path_est = std::filesystem::current_path().string() + "/log_aligned_traj_est.txt";
        // auto path_ref = std::filesystem::current_path().string() + "/log_aligned_traj_ref.txt";
        // std::cout << "dumping aligned est traj to " << path_est << std::endl;
        // std::cout << "dumping aligned ref traj to " << path_ref << std::endl;
        // std::ofstream est_traj_file_stream{path_est};
        // std::ofstream ref_traj_file_stream{path_ref};
        // est_traj_file_stream << estXYZ_aligned.transpose() << std::endl;
        // ref_traj_file_stream << refXYZ_aligned.transpose() << std::endl;
// 
    // }
    std::tuple<size_t, double, double> ApplyUmeyamaCalAPE(double max_diff) {
        auto [refXYZ_aligned, estXYZ_aligned] = alignTimeIdx(max_diff);
        // dumpaligned(refXYZ_aligned, estXYZ_aligned);
        Eigen::Matrix4d transformation = Eigen::umeyama(estXYZ_aligned, refXYZ_aligned, false);
        // std::cout << "transformation matrix: \n" << transformation << std::endl;
        Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
        Eigen::Vector3d t = transformation.block<3, 1>(0, 3);
        auto estXYZ_aligned_transform = R * estXYZ_aligned + t.replicate(1, estXYZ_aligned.cols());
        Eigen::Matrix<double, 1, Eigen::Dynamic> errors = (refXYZ_aligned - estXYZ_aligned_transform).colwise().norm();
        // 计算 MSE
        double mse = errors.array().square().mean();

        // 计算 RMSE
        double rmse = std::sqrt(mse);
        return std::make_tuple(estXYZ_aligned.cols(), mse, rmse);
    }
    void clear_est() {
        estTimeStamps_.clear();
        estXYZ_.clear();
        est_len_ = 0;
        total_err_ = std::make_tuple(0, 0., 0.);
        // bool init_mode_;
    }
    double runOdomCalcErr(const std::string& config_file, const std::string& bag_file, const Eigen::Ref<const Eigen::VectorXf>& extr, bool terminate = false, float terminate_ratio = 0.2, bool use_imu = false, double max_diff = 0.01) {
        clear_est();
        pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

        Eigen::Vector3d ext(extr.head<3>().cast<double>());
        Eigen::Quaterniond exr_qua(extr.tail<4>().cast<double>());

        std::vector<std::tuple<size_t, double, double>> vec_lenth_mse_rmse;

        auto laser_mapping = std::make_shared<faster_lio::LaserMapping>();
        if (!laser_mapping->InitWithoutROS(config_file)) {
            LOG(ERROR) << "LIO init failed.";
            throw std::runtime_error("LIO init failed. check your config file.");
        }
        laser_mapping->setExtr(ext, exr_qua);
        

        auto yaml = YAML::LoadFile(config_file);
        auto lidar_topic = yaml["common"]["lid_topic"].as<std::string>();
        auto imu_topic = yaml["common"]["imu_topic"].as<std::string>();
        rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
        auto view = rosbag::View(bag);
        auto now_time = view.getBeginTime();

        for (const rosbag::MessageInstance &m : view) {
            // if(terminate) {
            //     if(auto t = (m.getTime() - now_time).toSec(); t >= 1) {
            //         now_time = m.getTime();
            //         const auto&& [num_poses, mse, rmse] = ApplyUmeyamaCalAPE(max_diff);
            //         // std::cout << "calc rmse: " << rmse << std::endl;
            //         estTimeStamps_.clear();
            //         estXYZ_.clear();
            //         est_len_ = 0;
            //         if(std::get<0>(total_err_) == 0) {
            //             total_err_ = std::make_tuple(num_poses, mse, rmse);
            //         }
            //         else {
            //             auto&&[total_num_poses, total_mse, total_rmse] = total_err_;
            //             total_num_poses += num_poses;
            //             total_mse = (total_mse * (total_num_poses-num_poses) + mse * num_poses) / total_num_poses;
            //             total_rmse = std::sqrt(total_mse);
            //         }
            //         if(auto&& error = std::get<2>(total_err_); error > terminate_ratio) {
            //             error = 200;
            //             break;
            //         }
            //     }
            // }
            auto topic = m.getTopic();
            if(topic == lidar_topic) {
                auto type = m.getDataType();
                if(type == "livox_ros_driver/CustomMsg" || type == "livox_ros_driver2/CustomMsg") {
                    auto livox_msg = m.instantiate<livox_ros_driver::CustomMsg>();
                    // faster_lio::Timer::Evaluate(
                        // [&laser_mapping, &livox_msg, use_imu, this]() {
                                laser_mapping->LivoxPCLCallBack(livox_msg);
                                laser_mapping->Run(estTimeStamps_, estXYZ_, use_imu);
                            // },
                        // "Laser Mapping Single Run");
                    continue;
                } else if (type == "sensor_msgs/PointCloud2") {
                    auto std_msg = m.instantiate<sensor_msgs::PointCloud2>();
                    // faster_lio::Timer::Evaluate(
                        // [&laser_mapping, &std_msg, use_imu, this]() {
                            laser_mapping->StandardPCLCallBack(std_msg);
                            laser_mapping->Run(estTimeStamps_, estXYZ_, use_imu);
                            // },
                        // "Laser Mapping Single Run");
                    continue;
                }
            }
            else if (topic == imu_topic) {
                auto imu_msg = m.instantiate<sensor_msgs::Imu>();
                laser_mapping->IMUCallBack(imu_msg);
                continue;
            }
        }
        est_len_ = estXYZ_.size() / 3;
        // laser_mapping->Savetrajectory("/home/run/Projects/learning-to-calibration/Environment/Log/est_traj.txt");
        if (terminate) {
            // clear_est();
            return std::get<2>(total_err_);
        }
        else {
            // dumpEstTraj();
            const auto&& [num_poses, mse, rmse] = ApplyUmeyamaCalAPE(max_diff);
            clear_est();
            total_err_ = std::make_tuple(num_poses, mse, rmse);
            return std::get<2>(total_err_);
        }
    }
    auto getTotalErr() {
        return total_err_;
    }
private:
    Eigen::VectorXf extr;
    std::vector<double> refTimeStamps_;
    std::vector<double> estTimeStamps_;
    std::vector<double> refXYZ_;
    std::vector<double> estXYZ_;

    size_t ref_len_;
    size_t est_len_;
    bool init_mode_;
public:
    std::tuple<int, double, double> total_err_;
};