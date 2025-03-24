import numpy as np
import torch

from rl_solver import Environment, VectorizedEnvironment, Solver


class ToyEnvironment(Environment):
    def __init__(self):
        super().__init__(action_dim=19)
        self.dst = np.random.randn(self.action_dim)

    def step(self, action: np.ndarray) -> float:
        error = self.evaluate(action)
        return -error + np.random.randn(1) * 0.01

    def evaluate(self, action: np.ndarray) -> float:
        return np.linalg.norm(self.dst - action, axis=-1, keepdims=True)


class ToyVectorizedEnvironment(VectorizedEnvironment):
    def __init__(self):
        super().__init__(action_dim=19, num_envs=8)
        # self.dst = np.random.randn(7)
        self.dst = np.array([1.5,0.74,0.09,  0.653, -0.271,  0.271,  0.653])
        self.dst[3:] /= np.linalg.norm(self.dst[3:])

    def step(self, action: np.ndarray, episode_idx) -> np.ndarray:
        error = self.evaluate(action)
        return -error + np.random.randn(*error.shape) * 0.01

    def quaternion_distance(self, q1, q2):
        # Compute the quaternion difference between two quaternions
        return np.arccos(2 * np.abs(np.sum(q1 * q2, axis=-1))**2 - 1)

    def evaluate(self, action: np.ndarray, init = False) -> np.ndarray:
        p_error = np.linalg.norm(self.dst[:3]-action[...,:3], axis=-1, keepdims=True)
        r_error = self.quaternion_distance(self.dst[3:], action[...,3:])
        return p_error / 2 + r_error / torch.pi
        # return np.linalg.norm(self.dst - action, axis=-1, keepdims=True)


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    Solver(
        # ToyEnvironment()
        # Alternatively:
        ToyVectorizedEnvironment()
    ).run()
