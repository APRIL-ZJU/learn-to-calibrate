# L2Calib

Official code of $SE(3)$-Manifold Reinforcement Learning for Robust Extrinsic Calibration with Degenerate Motion Resilience

## Quick Start

### dependencies

- Ceres 1.14
- Sophus 1.22

1. Clone the Repository. 
```
cd
git clone https://github.com/APRIL-ZJU/learn-to-calibrate.git
git submodule update --init --recursive # for Traj-LO. Note that if you already have the reference trajectory, you don't need this component
```
2. Create Conda Environment
```
conda create -n L2Calib python=3.11
conda activate L2Calib
pip install numpy==1.26.4 torch==2.4.0 empy==3.3.4 pybind11==2.13.1
```
3. Build RL Environment
```
cd Environment
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)":$CMAKE_PREFIX_PATH
zsh build.sh
```
4. Before running, make sure the PATHs are specified

```
export PYTHONPATH=${HOME}/learn-to-calibrate/Environment/build/app:$PYTHONPATH
export PYTHONPATH=${HOME}/learn-to-calibrate/rl_solver:$PYTHONPATH
```

### test with our CSC example

1. Download our handheld rosbag [csc_01.bag](https://drive.google.com/file/d/1SMLkH2v-LdzJiV-x1GwTtzav8InIb5PN/view?usp=drive_link)

2. Replace the ```bag_dir``` variable in calib_csc.sh
3. Run ```zsh demo/calib_csc.sh```

### test with slam dataset

1. Download [NTU VIRAL](https://ntu-aris.github.io/ntu_viral_dataset/) or [MCD VIRAL](https://mcdviral.github.io/Download.html)
2. Replace the ```bag_dir``` variable in calib_ntu.sh / calib_mcd.sh
3. Run ```zsh demo/calib_ntu.sh``` or ```zsh demo/calib_mcd.sh```

### calibrate your own sensors

1. Put your rosbag in an empty folder 
2. Specify the ```fastlio_config```,```bag_dir```,```lidar_type```,```imu_topic``` and ```rough_trans``` in demo/calib.sh. 
- ```rough_trans``` represents the rough estimating of the translation between the sensors, where 0.1 means 10 cm.
- Note that if you already have the reference IMU trajectory, e.g., the IMU ground truth trajectory obtained from Mocap/RTK systems, you can set ```use_imu``` to ```True```. And ignore the steps below, just run 
```
python train.py --lio-config {FASTLIO CONFIG} --bag-dir {BAG DIR} --alg ppo --SO3-distribution Bingham --num-epochs 8000 --min -{ROUGH} --max 0.1
```
3. Specify the Traj-LO configuration under ```Environment/Traj-LO/data/```. Only the ```topic``` needs to be specified.
4. Run ```zsh demo/calib.sh```

## Acknowledegment

Thanks for these awesome works  [Traj-LO](https://github.com/kevin2431/Traj-LO)(Lidar-only odometry) [Faster-lio](https://github.com/gaoxiang12/faster-lio) (Tightly-coupled LIO) [BPP](https://github.com/stepjam/BPP) (Bingham policy parameterization for RL)

The parallel environment is adapted from [Fast-Evo](https://github.com/midiexiangxxx/fastevo) (Faster evo-ape/traj implemented in C++20)

Thanks [Chengrui Zhu](https://github.com/chengruiz) for implementing the PPO RL algorithm.