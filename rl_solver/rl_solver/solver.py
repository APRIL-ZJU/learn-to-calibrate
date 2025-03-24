import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import numpy as np
import torch
import wandb

from rl_solver import PPO
from rl_solver.common import Algorithm, Actor, CustomActor_B, QNet, VectorModule, ValueNet
from bingham import utils
__all__ = ['Environment', 'VectorizedEnvironment', 'Solver']


class Environment:
    action_dim: int

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def step(self, action: np.ndarray) -> float:
        raise NotImplementedError

    def evaluate(self, action: np.ndarray) -> float:
        raise NotImplementedError


class VectorizedEnvironment:
    action_dim: int
    num_envs: int

    def __init__(self, action_dim: int, num_envs: int):
        self.action_dim = action_dim
        self.num_envs = num_envs

    def step(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class VectorizedEnvironmentWrapper(VectorizedEnvironment):
    action_dim: int
    num_envs: int

    def __init__(self, env: Environment):
        self.env = env
        super().__init__(env.action_dim, 1)

    def step(self, action: np.ndarray, episode_idx) -> np.ndarray:
        return np.array([self.env.step(action.squeeze(axis=0))])

    def evaluate(self, action: np.ndarray) -> np.ndarray:
        return np.array([self.env.evaluate(action)])


class Solver:
    @dataclass(slots=True)
    class Arguments:
        alg: str = 'sac'
        wandb: bool = None
        run_name: str = None
        run_proj: str = None
        num_epochs: int = 1000
        num_collects: int = 32
        SO3_distribution: str = 'Gaussian'

    @classmethod
    def parse_args(cls) -> Arguments:
        parser = argparse.ArgumentParser()
        parser.add_argument('--bag-file', type=str, default='')
        parser.add_argument('--lio-config', type=str, default='')
        parser.add_argument('--alg', type=str, default='sac')
        parser.add_argument('--use-imu', type=bool, default=False)
        parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)
        parser.add_argument('--run-name', type=str)
        parser.add_argument('--run-proj', type=str)
        parser.add_argument('--num-epochs', type=int, default=1000)
        parser.add_argument('--num-collects', type=int, default=32)
        parser.add_argument('--SO3-distribution', type=str, default='Gaussian')
        parser.add_argument('--min', type=float, default=1.)
        parser.add_argument('--max', type=float, default=1.)
        args = parser.parse_args()
        print(args)
        return args

    def __init__(self, env: Environment | VectorizedEnvironment, args: Arguments = None):
        self.args = args or self.parse_args()
        self.trans_min = self.args.min
        self.trans_max = self.args.max
        self.actor: Actor = None
        self.agent: Algorithm = None
        assert isinstance(env, Environment) or isinstance(env, VectorizedEnvironment)
        self.env = env
        if isinstance(self.env, Environment):
            self.env = VectorizedEnvironmentWrapper(env)
        self.log_dir = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if self.args.run_name is not None:
            self.log_dir += f"-{self.args.run_name}"
        self.log_dir = f'logs/{self.log_dir}'
        os.makedirs(self.log_dir)
        self.log_file = open(f'{self.log_dir}/log.txt', 'w')
        self.result_file = open(f'{self.log_dir}/result.yml', 'w')

        wandb.require("core")
        wandb.init(
            name=self.args.run_name,
            project=self.args.run_proj,
            mode=None if self.args.wandb else 'disabled',
        )

        if self.args.num_collects % self.env.num_envs != 0:
            raise ValueError("num_collects must be divisible by num_envs")
        if self.args.alg == 'ppo':
            self.setup_ppo(num_collects=self.args.num_collects // self.env.num_envs)
        elif self.args.alg == 'sac':
            self.setup_sac(num_collects=self.args.num_collects // self.env.num_envs)
        else:
            raise NotImplementedError(f"Algorithm {self.args.alg} is not implemented")

    def __del__(self):
        self.log_file.close()
        self.result_file.close()
    def process_action(self, action):
        def generate_orthonormal_matrix_last_column(u):
            orthogonal_vectors = []
            for _ in range(3):
                v = np.random.rand(4)
                for w in orthogonal_vectors:
                    projection = (np.dot(v, w) / np.dot(w, w)) * w
                    v -= projection
                v -= (np.dot(v, u) / np.dot(u, u)) * u
                v = v / np.linalg.norm(v)
                orthogonal_vectors.append(v)
            orthogonal_vectors.append(u)
            orthonormal_matrix = np.array(orthogonal_vectors).T
            return orthonormal_matrix
        fixed_dims = action[:3]
        mode = action[3:]
        M = generate_orthonormal_matrix_last_column(mode.numpy())
        
        flattened_M = M.flatten()
        result = np.concatenate([fixed_dims, flattened_M])
        
        return result
    def init_ppo(self, init_iters = 128):
        # UNIFORM RANDOM ROTATIONS K. Shoemake 
        def sample_SO3(n):
            u1 = torch.rand(n)
            u2 = torch.rand(n)
            u3 = torch.rand(n)

            q0 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
            q1 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
            q2 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
            q3 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

            q = torch.stack([q0, q1, q2, q3], dim=1)

            return q

        p_actions = torch.FloatTensor(init_iters,3).uniform_(self.trans_min, self.trans_max)
        r_actions = sample_SO3(init_iters)
        actions = torch.cat([p_actions, r_actions], dim=1)
        max_reward = -np.inf

        for i in range(init_iters):
            avg_reward = 0
            action = actions[i]
            for _ in range(self.args.num_collects // self.env.num_envs):
                # action = actions[i*self.env.num_envs:(i+1)*self.env.num_envs]
                rewards = self.env.step(action.tile((self.env.num_envs,1)).numpy(), init_mode=True)
                # max_idx = np.argmax(rewards)
                # reward = rewards[max_idx]
                # if reward > max_reward:
                    # max_reward = reward
                    # best_action = action[max_idx]
                    # if reward > 0.05:
                        # break
                avg_reward += rewards.mean()
            avg_reward /= (self.args.num_collects // self.env.num_envs)
            print(avg_reward)
            if avg_reward > max_reward:
                max_reward = avg_reward
                best_action = action
                if avg_reward > 1e-3:
                    break
        print("init action: ", best_action, "init reward: ", max_reward)

        if self.args.SO3_distribution == 'Bingham':
            best_action = self.process_action(best_action)
        return best_action
    
    def setup_ppo(
        self,
        num_collects: int,
        num_subepochs: int = 8,
        num_mini_batches: int = 1,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        normalize_adv: bool = True,
        clip_value_loss: bool = True,
    ):
        init_action = self.init_ppo(init_iters=8192)
        # init_action = self.process_action(torch.tensor([0.6,  -0.35, -0.03, -0.004773663356900215, -0.7079726457595825, 0.002554073231294751, -0.7062190175056458]))
        # init_action = np.array([0.6,  -0.35, -0.03,  0.26833337,  0.01073508,  0.96307361,
#  -0.01926613,  0.64233445, -0.37628932, -0.18759307, -0.64079762,  0.57704496,
#  -0.24702301, -0.14271491,  0.76526541,  0.42710978,  0.89290018, -0.13011734,
#  -0.05810314])
        print("init_action: ", init_action)
        p_std_clamp_ratio = self.args.max/2
        if(self.args.SO3_distribution == 'Gaussian'):
            self.actor = Actor(VectorModule(self.env.action_dim))
            # self.actor.state_vector.set(torch.tensor( [-0.029665499925613403, -0.020126797258853912, -0.0018311796011403203, 0.00020893810142297298, -0.09414270520210266, -0.08577223122119904, -0.02246987819671631]))
            self.actor.state_vector.set(torch.tensor(init_action))
            self.actor.distribution.set_std(torch.tensor([p_std_clamp_ratio,p_std_clamp_ratio,p_std_clamp_ratio, 0.1, 0.1, 0.1, 0.1]))
        elif(self.args.SO3_distribution == 'Bingham'):
            self.actor = CustomActor_B(VectorModule(self.env.action_dim))
            self.actor.state_vector.set(torch.tensor(init_action))
            
            self.actor.p_distribution.set_std(torch.tensor([p_std_clamp_ratio,p_std_clamp_ratio,p_std_clamp_ratio]))
            self.actor.r_distribution.set_Z(torch.tensor([-20,-20,-20]))
        critic = ValueNet()
        critic.value.set(0.)
        self.agent = PPO(
            self.actor, critic,
            num_collects=num_collects,
            num_subepochs=num_subepochs,
            trans_min = self.trans_min, 
            trans_max = self.trans_max,
            num_mini_batches=num_mini_batches,
            clip_ratio=clip_ratio,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            normalize_adv=normalize_adv,
            clip_value_loss=clip_value_loss,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

    def run(self):
        min_error = np.inf
        print(self.agent.learning_rate)
        for episode in range(1, self.args.num_epochs + 1):
            avg_reward = 0
            avg_error = 0
            for i in range(self.agent.num_collects):
                action = self.agent.act(self.env.num_envs)
                error = self.env.evaluate(action)
                reward = np.exp(-error / 2)
                self.agent.step(reward)
                avg_reward += reward.mean()
                avg_error += error.mean()
            stats = self.agent.update(warmup=episode < 50, entropy_large=episode > 300)
            avg_reward /= self.agent.num_collects
            avg_error /= self.agent.num_collects
            # with torch.inference_mode():
                # if(self.args.SO3_distribution == 'Gaussian'):
                    # action = self.actor()
                    # error = self.env.evaluate(action.cpu().numpy()).item()
                    # action[3:] /= (torch.norm(action[3:]) + 1e-6)
                # elif(self.args.SO3_distribution == 'Bingham'):
                    # action = self.actor()
                    # xyz = action[:3]
                    # rotation_matrix = utils.gram_schmidt(action[3:].reshape(1, 4, 4))
                    # quaternion = rotation_matrix[0,:,-1]
                    # action = torch.cat([xyz, quaternion],0)
                    # error = self.env.evaluate(action.cpu().numpy()).item()
            wandb.log(stats | {'reward': avg_reward, 'error': avg_error})
            print(f"Episode {episode}: Error: {avg_error:.4f}, Reward: {avg_reward:.4f}")
            if(self.args.SO3_distribution == 'Gaussian'):
                action = self.actor()
                action = action.detach()
                action[3:] /= (torch.norm(action[3:]) + 1e-6)
            elif(self.args.SO3_distribution == 'Bingham'):
                action = self.actor()
                xyz = action[:3]
                rotation_matrix = utils.gram_schmidt(action[3:].reshape(1, 4, 4))
                quaternion = rotation_matrix[0,:,-1]
                action = torch.cat([xyz, quaternion],0)
            self.log_file.write(
                f"Episode {episode}: Error {avg_error}, Reward: {avg_reward}, "
                f"Result: {action.detach().cpu().squeeze().numpy().tolist()}\n"
            )
            self.log_file.flush()
            if avg_error < min_error:
                min_error = avg_error
                self.result_file.write(
                    f"- episode: {episode}\n"
                    f"  error: {avg_error}\n"
                    f"  result: {action.detach().cpu().squeeze().numpy().tolist()}\n"
                )
                self.result_file.flush()
