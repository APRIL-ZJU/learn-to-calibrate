//
// Created by xiang on 2021/10/9.
//
#include <yaml-cpp/yaml.h>
#include <gflags/gflags.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <unistd.h>
#include <csignal>

#include "laser_mapping.h"
#include "options.h"
#include "utils.h"

/// run faster-LIO in offline mode

DEFINE_string(config_file, "./config/avia.yaml", "path to config file");
DEFINE_string(bag_file, "/home/xiang/Data/dataset/fast_lio2/avia/2020-09-16-quick-shack.bag", "path to the ros bag");
DEFINE_string(time_log_file, "./Log/time.log", "path to time log file");
DEFINE_string(traj_log_file, "./Log/traj.txt", "path to traj log file");

// void SigHandle(int sig) {
    // faster_lio::options::FLAG_EXIT = true;
    // ROS_WARN("catch sig %d", sig);
// }

int main(int argc, char **argv) {
    auto t_start = std::chrono::high_resolution_clock::now();
    bool imu_en = false;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);

    const std::string bag_file = FLAGS_bag_file;
    const std::string config_file = FLAGS_config_file;
    YAML::Node config = YAML::LoadFile(config_file);
    const auto lidar_topic = config["common"]["lid_topic"].as<std::string>();
    const auto imu_topic = config["common"]["imu_topic"].as<std::string>();
    auto laser_mapping = std::make_shared<faster_lio::LaserMapping>();
    if (!laser_mapping->InitWithoutROS(FLAGS_config_file)) {
        LOG(ERROR) << "laser mapping init failed.";
        return -1;
    }

    /// handle ctrl-c
    // signal(SIGINT, SigHandle);

    // just read the bag and send the data
    LOG(INFO) << "Opening rosbag, be patient";
    rosbag::Bag bag(FLAGS_bag_file, rosbag::bagmode::Read);
    rosbag::View view_full(bag);
    rosbag::View view;
    view.addQuery(bag, rosbag::TopicQuery({lidar_topic, imu_topic}));


    for (const rosbag::MessageInstance &m : view) {

        const auto topic = m.getTopic();
        if(topic == lidar_topic) {
            if(auto type = m.getDataType(); 
            type == "livox_ros_driver/CustomMsg" || type == "livox_ros_driver2/CustomMsg") {
                auto livox_msg = m.instantiate<livox_ros_driver::CustomMsg>();
                faster_lio::Timer::Evaluate(
                    [&laser_mapping, &livox_msg]() {
                        laser_mapping->LivoxPCLCallBack(livox_msg);
                        laser_mapping->Run();
                    },
                    "Laser Mapping Single Run");
                continue;
            } else if(type == "sensor_msgs/PointCloud2") {
                auto point_cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
                faster_lio::Timer::Evaluate(
                    [&laser_mapping, &point_cloud_msg]() {
                        laser_mapping->StandardPCLCallBack(point_cloud_msg);
                        laser_mapping->Run();
                    },
                    "Laser Mapping Single Run");
                continue;
            }
        }
        else if(topic == imu_topic) {
            auto imu_msg = m.instantiate<sensor_msgs::Imu>();
            laser_mapping->IMUCallBack(imu_msg);
            continue;
        }

    }

    LOG(INFO) << "finishing mapping";
    // laser_mapping->Finish();

    /// print the fps
    auto t_end = std::chrono::high_resolution_clock::now();
    double fps_lidar = 1.0 / (faster_lio::Timer::GetMeanTime("Laser Mapping Single Run") / 1000.);
    int lidar_count = faster_lio::Timer::GetRecordSize("Laser Mapping Single Run");
    double fps_imu = 1.0 / (faster_lio::Timer::GetMeanTime("IMU Process") / 1000.);
    int imu_count = faster_lio::Timer::GetRecordSize("IMU Process");
    LOG(INFO) << "Faster lidar average FPS: " << fps_lidar;
    LOG(INFO) << "Faster imu average FPS: " << fps_imu;

    LOG(INFO) << "Total time: " << std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count() << "s";
    LOG(INFO) << "Total lidar time: " << 1.0 / fps_lidar * lidar_count << "s";
    LOG(INFO) << "Total imu time: " << 1.0 / fps_imu * imu_count << "s";
    LOG(INFO) << "save trajectory to: " << FLAGS_traj_log_file;
    laser_mapping->Savetrajectory(FLAGS_traj_log_file);

    // faster_lio::Timer::PrintAll();
    // faster_lio::Timer::DumpIntoFile(FLAGS_time_log_file);

    return 0;
}
