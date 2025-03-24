#include "compute_eigen.hpp"
#include "segment_helper.hpp"

int main(int argc, char **argv) {
  // Command line arguments parsing
  if (argc < 4) {
    ROS_ERROR("Usage: ./process_bag <run-mode> <bag-file> <imu-topic> <lidar-type> <trajlo-bin>");
    return 1;
  }
  std::string run_mode = argv[1];
  std::string bag_dir = argv[2];
  std::string imu_topic = argv[3];
  std::string traj_lo_type = argv[4];
  std::string trajlo_bin = argv[5];
  SegmentHelper helper;
  std::vector<double> timestamps, gyro_norm, accel_norm;
  bool livox = imu_topic.find("livox") != std::string::npos;
  auto condition = [livox, &gyro_norm, &accel_norm]() {
    std::vector<bool> cond(gyro_norm.size(), false);
    for (size_t i = 0; i < gyro_norm.size(); ++i) {
      if (livox) {
        cond[i] = (gyro_norm[i] < 0.2 && gyro_norm[i] > 0.02 &&
                   std::abs(accel_norm[i] - 1) < 0.06);
      } else {
        cond[i] = (gyro_norm[i] < 0.2 && gyro_norm[i] > 0.02 &&
                   std::abs(accel_norm[i] - 9.8) < 0.5);
      }
    }
    return cond;
  };
  std::vector<std::string> bag_files;
  for (const auto &entry : fs::directory_iterator(bag_dir)) {
    if (entry.path().extension() == ".bag") {
      bag_files.push_back(entry.path().string());
    }
  }
  auto total_chunk_dir = fs::path(bag_dir).string() + "/total";
  auto chunk_idx = 0;
  std::tuple<double, std::string, double> max_rot_percent = {0, "", 0};
  if (!fs::exists(total_chunk_dir)) {
    fs::create_directories(total_chunk_dir);
  }
  std::vector<std::pair<double, std::string>> begin_times_init_bags;
  for (const std::string &bag_file : bag_files) {
    timestamps.clear();
    gyro_norm.clear();
    accel_norm.clear();
    std::cout << "Processing " << bag_file << "\n";
    helper.readIMUFromBag(bag_file, imu_topic, timestamps, gyro_norm,
                          accel_norm);

    std::vector<bool> cond = condition();
    std::vector<int> indices;
    for (size_t i = 0; i < cond.size(); ++i) {
      if (cond[i]) {
        indices.push_back(i);
      }
    }

    auto subarrays = helper.findConsecutiveSubarrays(indices);
    std::cout << "Found indices: " << indices.size() << "\n";
    std::cout << "Found subarrays: " << subarrays.size() << "\n";
    // // dump indices to a file
    // std::ofstream fout(bag_file + "_indices.txt");
    // for (const auto &idx : indices) {
    //   fout << idx << "\n";
    // }
    std::vector<double> indices_begin;
    for (const auto &subarray : subarrays) {
      if (subarray.size() >= 1) {
        indices_begin.push_back(timestamps[subarray[3]] - timestamps[0]);
      }
    }
    auto begin_times = helper.cropTimestamps(indices_begin, 15.0);
    std::cout << "Segmenting " << begin_times.size()
              << " bags (begin_time.size)\n";
    auto output_dir = total_chunk_dir;
    rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
    rosbag::View view;
    view.addQuery(bag);
    begin_times_init_bags.push_back(
        {begin_times[0] + view.getBeginTime().toSec(),
         output_dir + "/init.bag"});
    if (run_mode == "trajlo") {
      if (fs::exists(output_dir + "/init.tum")) continue;
      // helper.splitRosbag(bag_file, begin_times, output_dir, run_mode, 15);
      helper.splitRosbag(bag_file, begin_times,
                              output_dir, run_mode, 15);

      std::string trajlo_config = fs::path(__FILE__).parent_path().string() +
                                  "/../Traj-LO/data/config_" +
                                  traj_lo_type + ".yaml";
      auto command =
          trajlo_bin + ' ' + trajlo_config + ' ' + output_dir + "/init.bag";
      std::cout << command << std::endl;
      system(command.c_str());

    } else if (run_mode == "filtering") {
      auto sufficiency_judger = std::make_unique<DataSufficiency<4>>();
      sufficiency_judger->loadTraj(output_dir + "/init.tum");
      sufficiency_judger->buildTrajectory();
      sufficiency_judger->dump_traj();
      for (int i = 0; i < begin_times.size(); ++i) {
        if (auto rot_percent =
                sufficiency_judger->EigenDecomposition(begin_times[i]);
            rot_percent > 3e-3) {
          auto output_file = total_chunk_dir + "/chunk_" +
                             std::to_string(chunk_idx++) + ".bag";
          helper.splitRosbag_once(bag_file, begin_times[i], output_file, 15);
          if (rot_percent > std::get<0>(max_rot_percent)) {
            std::get<0>(max_rot_percent) = rot_percent;
            std::get<1>(max_rot_percent) = bag_file;
            std::get<2>(max_rot_percent) = begin_times[i];
          }
        }
      }
    }
  }
  if (run_mode == "trajlo") {
    // sort begin_times_init_bags
    std::for_each(begin_times_init_bags.begin(), begin_times_init_bags.end(),
                  [](auto &begin_time_init_bag) {
                    std::cout << begin_time_init_bag.first << " "
                              << begin_time_init_bag.second << std::endl;
                  });
    std::sort(
        begin_times_init_bags.begin(), begin_times_init_bags.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &begin_time_init_bag : begin_times_init_bags) {
      auto command = "cat " +
                     fs::path(begin_time_init_bag.second)
                         .replace_extension(".tum")
                         .string() +
                     " >> " + total_chunk_dir + "/lidar_traj.tum";
      std::cout << "running command: " << command << std::endl;
      system(command.c_str());
    }
    std::for_each(begin_times_init_bags.begin(), begin_times_init_bags.end(),
                  [](auto &begin_time_init_bag) {
                    std::cout << begin_time_init_bag.first << " "
                              << begin_time_init_bag.second << std::endl;
                  });
  } else if (run_mode == "filtering") {
    std::cout << "Creating init.bag with max rot percent: "
              << std::get<0>(max_rot_percent) << " at "
              << total_chunk_dir + "/init.bag from "
              << std::get<1>(max_rot_percent) << std::endl;
    helper.splitRosbag_once(std::get<1>(max_rot_percent),
                            std::get<2>(max_rot_percent),
                            total_chunk_dir + "/init.bag", 15);
  }
  return 0;
}
