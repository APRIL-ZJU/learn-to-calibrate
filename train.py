import numpy as np
import torch
import multiprocessing as mp
import yaml
import os
import glob
from rl_solver import Environment, VectorizedEnvironment, Solver
import argparse

import L2CE

class LIOVectorizedEnvironment(VectorizedEnvironment):
    def __init__(self, args: argparse.Namespace):
        action_dim = 19 if args.SO3_distribution == 'Bingham' else 7
        num_envs = args.num_envs
        super().__init__(action_dim, num_envs)
        self.use_imu = args.use_imu
        self.config_file = args.lio_config
        self.bag_dir = args.bag_dir
        self.bag_file = self.bag_dir + 'total/init.bag'
        self.ref_file = self.bag_dir + 'total/imu_traj.tum' if args.use_imu else self.bag_dir + 'total/lidar_traj.tum'
        self.bag_num = len(glob.glob(os.path.join(self.bag_dir, 'total/chunk_*.bag')))
        self.envs = L2CE.ParallelEnvironment(self.num_envs, self.config_file, self.bag_dir+'total/', self.ref_file)
        print(f'bag_dir: {self.bag_dir}')
        print(f'bag_num: {self.bag_num}')
        print(f'load ref traj from {self.ref_file}')
    def step(self, action, init_mode = False):
        error = self.evaluate(action, init_mode)
        error.clip(0, 300, out=error)
        return np.exp(-error / 2)

    def evaluate(self, action: np.ndarray, init_mode = True) -> np.ndarray:
        if len(action.shape) == 1: # inference
            init_file = self.bag_dir + "init.bag"
            bag_file = init_file if os.path.exists(init_file) else self.bag_file
            result = self.envs.runOdomCalcErr(action, bag_file, False, 3, self.use_imu, 0.01)
            return np.array(result)
        elif action.shape[0] == self.num_envs: # training
            residuals = self.envs.parallelRunOdomCalcErr(action, False, 3, self.use_imu, 0.01)
            return np.array(residuals).reshape(self.num_envs, 1)
        

if __name__ == '__main__':
    print("Start")
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-dir', type=str, default='')
    parser.add_argument('--lio-config', type=str, default='')
    parser.add_argument('--alg', type=str, default='sac')
    parser.add_argument('--use-imu', type=bool, default=False)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)
    parser.add_argument('--run-name', type=str)
    parser.add_argument('--run-proj', type=str)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--num-collects', type=int, default=32)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--SO3-distribution', type=str, default='Gaussian')
    parser.add_argument('--min', type=float, default=-1.)
    parser.add_argument('--max', type=float, default=1.)
    args = parser.parse_args()
    print(args)
    np.random.seed(42)
    torch.manual_seed(42)
    Solver(
        LIOVectorizedEnvironment(args), 
        args
    ).run()
