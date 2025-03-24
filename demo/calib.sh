#!/bin/zsh

fastlio_config=(
"Environment/fasterlio/config/csc.yaml"
# "Environment/config/mcd_tuhh.yaml"
# "Environment/config/ntu-viral.yaml"
)
use_imu=False
trajlo_bin="Environment/Traj-LO/build/trajlo"

bag_dir="/home/run/csc-test/"
num_epochs=8000
num_collects=32

SO3_distribution=(
Bingham
# Gaussian
)

lidar_type=(
livox
# ouster
# velodyne
# hesai
)

imu_topic=(
# "/imu/imu"
"/imu/data"
# "/livox/imu"
)

rough_trans=0.1

for RUN_MODE in trajlo filtering; do
  ./Environment/process_module/build/process_bag ${RUN_MODE} ${bag_dir} ${imu_topic} ${lidar_type} ${trajlo_bin}
done

python train.py --lio-config ${fastlio_config} --bag-dir ${bag_dir} --alg ppo --SO3-distribution ${SO3_distribution} --num-epochs ${num_epochs} --min -${rough_trans} --max ${rough_trans}