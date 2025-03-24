import math
from collections import defaultdict
from typing import Union

import numpy as np
import torch
from torch import nn

# for bingham distribution
from bingham.torch_bingham import BinghamDistribution
from bingham import utils

class VectorModule(nn.Module):
    def __init__(self, dim, requires_grad=True):
        super().__init__()
        self.param = nn.Parameter(torch.randn(dim), requires_grad=requires_grad)

    @property
    def size(self):
        return self.param.size()[0]

    def forward(self, x=None):
        if x is None:
            return self.param
        # return self.param.repeat(*x.shape[:-1], 1)
        return self.param.repeat(x.shape[0], 1)

    def clamp(self, lb=None, ub=None, indices=None):
        self.param.data[indices].clamp_(min=lb, max=ub)

    def set(self, value):
        self.param.data[:] = value

    def set(self, value:torch.Tensor):
        self.param.data[:] = value


class RLaplace(nn.Module):
    def __init__(self, action_dim, init_std=1.0, requires_grad=True):
        super().__init__()
        

class Bingham(nn.Module):
    def __init__(self, action_dim, init_std = -50, requires_grad=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distribution = BinghamDistribution(device)
        self._Z = VectorModule(action_dim, requires_grad)
        self._Z.set(math.log(-init_std))

    def forward(self, M):
        Z = -self._Z(M[:,:,-1]).exp()
        zeros = torch.zeros(Z.shape[0], 1, device=Z.device)
        return M, torch.cat((Z, zeros),1)
    
    def sample_log_prob(self, M):
        Z = -self._Z(M).exp()
        s = self.sample(M)
        num_parallels, _, _ = M.shape
        zeros = torch.zeros(Z.shape[0], 1, device=Z.device)
        return M, torch.cat((Z, zeros),1), s, self.calc_log_prob(M, Z.reshape(num_parallels, 3), s)

    def calc_log_prob(self, M: torch.Tensor, Z: torch.Tensor, actions):
        log_prob = self.distribution.log_probs(actions, M, Z)
        return log_prob

    def calc_entropy(self, Z) -> torch.Tensor:
        e = self.distribution.entropy(Z)
        return e

    def sample(self, M: VectorModule) -> torch.Tensor:
        Z = -self._Z(M).exp()
        num_parallels, _, _ = M.shape
        s = self.distribution.rsample(M, Z.reshape(num_parallels, 3))
        return s
    
    def clamp_std(self, lb=None, ub=None, indices=None):
        self._Z.clamp(lb=math.log(-lb), ub=math.log(-ub), indices=indices)

    def set_Z(self, value):
        self._Z.set(torch.log(-value))


LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))
ENTROPY_BIAS = 0.5 + 0.5 * math.log(2 * math.pi)


class Gaussian(nn.Module):
    def __init__(self, action_dim, init_std=1.0, requires_grad=True):
        super().__init__()
        self.std = VectorModule(action_dim, requires_grad)
        self.std.set(math.log(init_std))

    def forward(self, action_mean):
        action_std = self.std(action_mean).exp()
        return action_mean, action_std

    def sample_log_prob(self, action_mean):
        action_std = self.std(action_mean).exp()
        sample = torch.distributions.Normal(action_mean, action_std).rsample()
        log_prob = self.calc_log_prob(action_mean, action_std, sample)
        return action_mean, action_std, sample, log_prob

    @staticmethod
    def calc_log_prob(mean, std, sample):
        return torch.sum(
            -((sample - mean) ** 2) / (2 * std ** 2) - std.log() - LOG_SQRT_2PI,
            dim=-1, keepdim=True
        )

    @staticmethod
    def calc_entropy(std):
        return torch.sum(
            ENTROPY_BIAS + std.log(),
            dim=-1, keepdim=True
        )

    def set_std(self, std):
        self.std.set(torch.log(std))

    def clamp_std(self, lb=None, ub=None, indices=None):
        self.std.clamp(lb=math.log(lb), ub=math.log(ub), indices=indices)


class CustomActor_B(nn.Module):
    def __init__(self, state_vector: VectorModule):
        super().__init__()
        self.state_vector = state_vector
        self.p_distribution = Gaussian(3)
        Z_init = torch.tensor([-50., -50., -50.])
        self.r_distribution = Bingham(3)

    @property
    def action_dim(self):
        return self.state_vector.size

    def explore(self, num_parallels=1, sample=True):
        action_mean = self.state_vector().repeat(num_parallels, 1)
        position_action_mean = action_mean[:,:3]
        rotation_action_mode = utils.gram_schmidt(
                            action_mean[:,3:].reshape(num_parallels, 4, 4)
                            )
        if not sample:
            p_action, p_std = self.p_distribution(position_action_mean)
            r_action, r_Z = self.r_distribution(rotation_action_mode)
            r_action = r_action.reshape(r_action.shape[0],-1)
            return torch.cat((p_action, r_action),1), torch.cat((p_std, r_Z),1)
        p_action_mean, p_action_std, p_sample, p_log_prob = self.p_distribution.sample_log_prob(position_action_mean)
        r_action_mode, r_action_std, r_sample, r_log_prob = self.r_distribution.sample_log_prob(rotation_action_mode)
        r_action_mode = r_action_mode.reshape(r_action_mode.shape[0], -1)
        return torch.cat((p_action_mean, r_action_mode),1), torch.cat((p_action_std, r_action_std),1), \
                torch.cat((p_sample, r_sample),1), p_log_prob+r_log_prob

    def act_log_prob(self, num_parallels=1):
        _, _, position_action, position_log_prob, _, _, rotation_action, rotation_log_prob = self.explore(num_parallels, sample=True)
        return torch.cat((position_action, rotation_action), 1), position_log_prob + rotation_log_prob

    def act_stochastic(self, num_parallels=1):
        _, _, p_action, _, _, _, r_action, _ = self.explore(num_parallels, sample=True)
        return torch.cat((p_action, r_action), 1)

    def calc_log_prob_entropy(self, mean, std, sample):
        p_mean = mean[:,:3]
        p_std = std[:,:3]
        M = mean[:,3:].reshape(mean.shape[0],4,4).repeat(sample.shape[0],1,1)
        Z = std[:,3:6].repeat(sample.shape[0],1)
        return self.p_distribution.calc_log_prob(p_mean, p_std, sample[:,:3]) + self.r_distribution.calc_log_prob(M, Z, sample[:,3:]) \
                , self.p_distribution.calc_entropy(p_std) + self.r_distribution.calc_entropy(Z)

    def forward(self):
        return self.state_vector()
        # action_mean = self.state_vector()
        # position_action_mean = action_mean[:3]
        # rotation_action_mode = utils.gram_schmidt(
                            # action_mean[3:].reshape(1, 4, 4)
                            # )[0,:,-1]
        # return torch.cat([position_action_mean, rotation_action_mode], 0)

class Actor(nn.Module):
    def __init__(self, state_vector: VectorModule):
        super().__init__()
        self.state_vector = state_vector
        self.distribution = Gaussian(state_vector.size)

    @property
    def action_dim(self):
        return self.state_vector.size

    def explore(self, num_parallels=1, sample=True):
        action_mean = self.state_vector().repeat(num_parallels, 1)
        if not sample:
            return self.distribution(action_mean)
        return self.distribution.sample_log_prob(action_mean)

    def act_log_prob(self, num_parallels=1):
        _, _, action, log_prob = self.explore(num_parallels, sample=True)
        return action, log_prob

    def act_stochastic(self, num_parallels=1):
        _, _, action, _ = self.explore(num_parallels, sample=True)
        return action

    def calc_log_prob_entropy(self, mean, std, sample):
        return self.distribution.calc_log_prob(mean, std, sample), self.distribution.calc_entropy(std)

    def forward(self):
        return self.state_vector()


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = VectorModule(1, requires_grad=True)

    def forward(self, num_parallels=1):
        return self.value().repeat(num_parallels, 1)


class MLP(nn.Module):
    def __init__(
        self, shape, input_dim, output_dim, activation_fn=nn.ReLU,
    ):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn

        modules = []
        shape = [input_dim] + list(shape)
        for idx in range(len(shape) - 1):
            modules.append(nn.Linear(shape[idx], shape[idx + 1]))
            modules.append(self.activation_fn())
        modules.append(nn.Linear(shape[-1], output_dim))
        self.network = nn.Sequential(*modules)

        self.shape = shape
        self.input_dim = input_dim
        self.output_shape = [output_dim]

    def forward(self, x):
        return self.network(x)


class QNet(nn.Module):
    def __init__(self, action_dim, shape=(128, 64), num_q=2):
        super().__init__()
        self.networks = nn.ModuleList([MLP(shape, action_dim, 1) for _ in range(num_q)])

    def forward(self, act):
        return torch.cat([m(act) for m in self.networks], dim=-1)

    def evaluate(self, act, take_min=False):
        q_value = self(act)
        if take_min:
            # Min over all critic networks
            q_value, _ = torch.min(q_value, dim=-1, keepdim=True)
        return q_value


class Statistics:
    def __init__(self):
        self.stats = defaultdict(lambda: 0.)
        self.count = defaultdict(lambda: 0)

    def add(self, **stats):
        for name, stat in stats.items():
            if stat is None:
                continue
            self.stats[name] += self._to_float(stat)
            self.count[name] += 1

    def clear(self):
        self.stats.clear()
        self.count.clear()

    @classmethod
    def _to_float(cls, stat):
        if isinstance(stat, float):
            return stat
        if isinstance(stat, torch.Tensor):
            if stat.dim() == 0:
                return stat.detach().cpu().item()
            with torch.inference_mode():
                return stat.detach().mean().cpu().item()
        if isinstance(stat, np.ndarray):
            return stat.mean().item()
        raise ValueError(f'Unsupported type {type(stat)}')

    def get(self, prefix='', clear=False):
        ret = {f'{prefix}{name}': stat / self.count[name] for name, stat in self.stats.items()}
        if clear:
            self.clear()
        return ret

    def __getitem__(self, item):
        return self.stats[item] / self.count[item]


class Algorithm:
    MODULES: list[str] = []
    OPTIMIZERS: list[str] = []
    OPTIONALS: list[str] = []

    def __init__(
        self,
        action_dim: int,
        num_collects: int,
        device: Union[torch.device, str] = None,
    ):
        self.action_dim = action_dim
        self.num_collects = num_collects
        self.device = torch.device(device)
        self.stats = Statistics()

    def as_th(self, tensor, **kwargs):
        return torch.as_tensor(tensor, **kwargs, device=self.device)

    def act(self, num_parallels=1) -> np.ndarray:
        pass

    def step(self, reward):
        pass

    def update(self, warmup=False) -> dict:
        pass

    def state_dict(self):
        state_dict = {}
        for name in self.MODULES + self.OPTIMIZERS:
            state_dict[name] = getattr(self, name).state_dict()
        for name in self.OPTIONALS:
            component = getattr(self, name, None)
            if component is None:
                continue
            if hasattr(component, "state_dict"):
                state_dict[name] = component.state_dict()
            elif isinstance(component, torch.Tensor):
                state_dict[name] = component
            else:
                raise RuntimeError(f"Unsupported component `{name}`")
        return state_dict

    def load_state_dict(self, state_dict):
        for name in self.MODULES:
            getattr(self, name).load_state_dict(state_dict[name])
        for name in self.OPTIMIZERS:
            getattr(self, name).load_state_dict(state_dict[name])
        for name in self.OPTIONALS:
            component = getattr(self, name, None)
            if component is not None:
                if name not in state_dict:
                    raise RuntimeError(
                        f"Component `{name}` exists in the algorithm "
                        "but does not exists in the state_dict"
                    )
                component.load_state_dict(state_dict[name])
            elif name in state_dict:
                raise RuntimeError(
                    f"Component `{name}` exists in the state_dict "
                    "but does not exists in the algorithm"
                )

    @staticmethod
    def optim_step(optimizer: torch.optim.Optimizer, loss: torch.Tensor, max_grad_norm=None):
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            for param in optimizer.param_groups:
                nn.utils.clip_grad_norm_(param['params'], max_grad_norm)
        optimizer.step()
