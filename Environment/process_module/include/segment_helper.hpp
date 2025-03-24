#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class SegmentHelper {

public:
  // Function to read IMU data from a ROS bag file
  void readIMUFromBag(const std::string &bag_file, const std::string &imu_topic,
                      std::vector<double> &timestamps,
                      std::vector<double> &gyro_norm,
                      std::vector<double> &accel_norm) {
    rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
    rosbag::View view;
    view.addQuery(bag, rosbag::TopicQuery(imu_topic));

    for (const rosbag::MessageInstance &m : view) {
      sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
      if (imu_msg->header.stamp.toSec() < view.getBeginTime().toSec())
        continue;
      if (imu_msg) {
        timestamps.push_back(imu_msg->header.stamp.toSec());
        gyro_norm.push_back(sqrt(pow(imu_msg->angular_velocity.x, 2) +
                                 pow(imu_msg->angular_velocity.y, 2) +
                                 pow(imu_msg->angular_velocity.z, 2)));
        accel_norm.push_back(sqrt(pow(imu_msg->linear_acceleration.x, 2) +
                                  pow(imu_msg->linear_acceleration.y, 2) +
                                  pow(imu_msg->linear_acceleration.z, 2)));
      }
    }
    bag.close();
  }

  // Find consecutive subarrays of indices
  std::vector<std::vector<int>>
  findConsecutiveSubarrays(const std::vector<int> &arr) {
    std::vector<std::vector<int>> result;
    if (arr.empty()) {
      return result;
    }

    std::vector<int> current_subarray = {arr[0]};
    for (size_t i = 1; i < arr.size(); ++i) {
      if (arr[i] == arr[i - 1] + 1) {
        current_subarray.push_back(arr[i]);
      } else {
        if (current_subarray.size() > 50) {
          result.push_back(current_subarray);
          current_subarray = {arr[i]};
        }
      }
    }
    if (current_subarray.size() > 30) {
      result.push_back(current_subarray);
    }

    return result;
  }

  // Crop timestamps with a minimum interval
  std::vector<double> cropTimestamps(const std::vector<double> &timestamps,
                                     double min_interval = 5.0) {
    std::vector<double> result = {timestamps[0]};
    for (size_t i = 1; i < timestamps.size(); ++i) {
      double diff = timestamps[i] - result.back();
      if (diff >= min_interval && diff <= 50) {
        result.push_back(timestamps[i]);
      }
    }
    return result;
  }
  void splitRosbag_once(const std::string &bag_file, double begin_time,
                        const std::string &output_file,
                        double segment_duration = 5.0) {
    if (fs::exists(output_file)) {
      return;
    }
    rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
    rosbag::View view;
    view.addQuery(bag);
    double start_time = view.getBeginTime().toSec();
    double segment_start = start_time + begin_time;
    double segment_end = segment_start + segment_duration;
    // std::string output_bag_file =
    // output_dir + "/chunk_" + std::to_string(chunk_idx) + ".bag";
    rosbag::Bag output_bag;
    output_bag.open(output_file, rosbag::bagmode::Write);
    std::cout << "Writing rosbag " << output_file << std::endl;
    for (const rosbag::MessageInstance &m : view) {
      ros::Time msg_time = m.getTime();
      if (msg_time.toSec() >= segment_start &&
          msg_time.toSec() <= segment_end) {
        output_bag.write(m.getTopic(), msg_time, m);
      }
    }
    output_bag.close();
  }
  // Split the ROS bag into smaller segments
  void splitRosbag(const std::string &bag_file,
                   const std::vector<double> &begin_times,
                   const std::string &output_dir, const std::string &run_mode,
                   double segment_duration = 5.0) {
    if (!fs::exists(output_dir)) {
      fs::create_directories(output_dir);
    }

    rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
    rosbag::View view;
    view.addQuery(bag);
    double start_time = view.getBeginTime().toSec();
    if (run_mode == "trajlo") {
      double segment_start = start_time + begin_times[0];
      double segment_end = view.getEndTime().toSec();
      std::string output_bag_file = output_dir + "/init.bag";
      rosbag::Bag output_bag;
      output_bag.open(output_bag_file, rosbag::bagmode::Write);
      for (const rosbag::MessageInstance &m : view) {
        ros::Time msg_time = m.getTime();
        if (msg_time.toSec() >= segment_start &&
            msg_time.toSec() <= segment_end) {
          output_bag.write(m.getTopic(), msg_time, m);
        }
      }
      output_bag.close();
      std::cout << std::setprecision(12)
                << "Created init bag: " << output_bag_file << " (from "
                << segment_start << " to " << segment_end << ")\n";
      bag.close();
      return;
    }
    for (size_t i = 0; i < begin_times.size(); ++i) {
      double segment_start = start_time + begin_times[i];
      double segment_end = segment_start + segment_duration;
      std::string output_bag_file =
          output_dir + "/chunk_" + std::to_string(i) + ".bag";
      rosbag::Bag output_bag;
      output_bag.open(output_bag_file, rosbag::bagmode::Write);

      for (const rosbag::MessageInstance &m : view) {
        ros::Time msg_time = m.getTime();
        if (msg_time.toSec() >= segment_start &&
            msg_time.toSec() <= segment_end) {
          output_bag.write(m.getTopic(), msg_time, m);
        }
      }
      output_bag.close();
      std::cout << "Created segment " << i + 1 << ": " << output_bag_file
                << " (from " << segment_start << " to " << segment_end << ")\n";
    }
    bag.close();
  }

  // Dump IMU norms into a file
  void dumpIMUNorms(const std::vector<double> &timestamps,
                    const std::vector<double> &gyro_norm,
                    const std::vector<double> &accel_norm,
                    const std::string &output_file) {
    std::ofstream fout(output_file);
    for (size_t i = 0; i < timestamps.size(); ++i) {
      fout << timestamps[i] << " " << gyro_norm[i] << " " << accel_norm[i] - 9.8
           << "\n";
    }
    fout.close();
    std::cout << "Dumped IMU norms to " << output_file << "\n";
  }
};
