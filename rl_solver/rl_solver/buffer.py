from dataclasses import dataclass
from typing import Any

import torch

__all__ = ['Buffer', 'ReplayBuffer', 'RolloutBuffer']


@dataclass(slots=True)
class Transition:
    action: torch.Tensor = None
    reward: torch.Tensor = None


@dataclass(slots=True)
class Batch:
    action: torch.Tensor = None
    reward: torch.Tensor = None


class Buffer(Transition):
    Transition = Transition
    Batch = Batch

    def __init__(self, length, device):
        super().__init__()
        self.device = device
        self.length = length
        self._initialized = False
        self.step = 0

    def zeros_th(self, *shape, **kwargs):
        return torch.zeros(shape, **kwargs, device=self.device)

    def as_th(self, data):
        return torch.as_tensor(data, device=self.device)

    def add_transition(self, t: Transition):
        if not self._initialized:
            self._lazy_init(t)
            self._initialized = True
        if self._is_overflow():
            raise RuntimeError("Buffer overflow")
        self._add(t)
        self._inc_step()

    def _lazy_init(self, t: Transition):
        self.reward = self._init_buf_of(t.reward)
        self.action = self._init_buf_of(t.action)

    def _add(self, t: Transition):
        self.action[self.step] = self.as_th(t.action)
        self.reward[self.step] = self.as_th(t.reward)

    def clear(self):
        self.step = 0

    def _is_overflow(self):
        return self.step >= self.length

    def _inc_step(self):
        self.step += 1

    def _init_buf_of(self, example: torch.Tensor, **kwargs):
        return self.zeros_th(self.length, *example.shape, **kwargs)


class ReplayBuffer(Buffer):
    def __init__(self, buffer_size, device):
        super().__init__(buffer_size, device)

        self.is_full = False

    def _lazy_init(self, t: Transition):
        # actual length
        self.length = int(self.length / t.reward.shape[0])
        super()._lazy_init(t)

    def add_transition(self, t: Transition):
        super().add_transition(t)
        if self.step >= self.length:
            self.step -= self.length
            self.is_full = True

    def sampler(self, batch_size, num_batches):
        for _ in range(num_batches):
            indices = torch.randint(self.num_samples, (batch_size,), device=self.device)
            batch = self.Batch(self.action.flatten(0, 1)[indices],
                               self.reward.flatten(0, 1)[indices])
            yield batch

    def clear(self):
        super().clear()
        self.is_full = False

    @property
    def num_samples(self):
        if self.is_full:
            return self.length * self.action.shape[1]
        if self.step == 0:
            return 0
        return self.step * self.action.shape[1]


class RolloutBuffer(Buffer):
    @dataclass(slots=True)
    class Transition(Buffer.Transition):
        value: Any = None
        action_mean: Any = None
        action_std: Any = None
        action_log_prob: Any = None

    @dataclass(slots=True)
    class Batch:
        action: torch.Tensor = None
        action_mean: torch.Tensor = None
        action_std: torch.Tensor = None
        reward: torch.Tensor = None
        value: torch.Tensor = None
        advantage: torch.Tensor = None
        action_log_prob: torch.Tensor = None
        # assigned every ppo update
        curr_action_mean: torch.Tensor = None
        curr_action_std: torch.Tensor = None
        curr_value: torch.Tensor = None

    def _lazy_init(self, t: Transition):
        super()._lazy_init(t)
        self.value = torch.zeros_like(self.reward)
        self.advantage = torch.zeros_like(self.reward)
        # self.action_mean = torch.zeros_like(self.action)
        self.action_mean = torch.zeros(self.action.shape[0],*t.action_mean.shape, device=self.action.device)
        self.action_std = torch.zeros_like(self.action)
        self.action_log_prob = self._init_buf_of(t.action_log_prob)

    def _add(self, t: Transition):
        super()._add(t)
        self.value[self.step] = self.as_th(t.value)
        self.action_mean[self.step] = self.as_th(t.action_mean)
        self.action_std[self.step] = self.as_th(t.action_std)
        self.action_log_prob[self.step] = self.as_th(t.action_log_prob)

    @torch.no_grad()
    def compute_advantage(self):
        self.advantage[:] = self.reward - self.value
        std, mean = torch.std_mean(self.advantage)
        self.advantage[:] = (self.advantage - mean) / (std + 1e-8)

    def sampler(self, num_mini_batches, num_repetitions, batch: Batch = None):
        self.compute_advantage()
        mini_batch_size = self.action.shape[0] * self.action.shape[1] // num_mini_batches
        for _ in range(num_repetitions):
            rand_indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
            for i in range(num_mini_batches):
                indices = rand_indices[i * mini_batch_size: (i + 1) * mini_batch_size]
                yield self._minibatch_sampler(indices, batch)

    def _minibatch_sampler(self, indices, batch=None):
        if batch is None:
            batch = self.Batch()
        batch.action = self.action.flatten(0, 1)[indices]
        batch.action_mean = self.action_mean.flatten(0, 1)[indices]
        batch.action_std = self.action_std.flatten(0, 1)[indices]
        batch.reward = self.reward.flatten(0, 1)[indices]
        batch.value = self.value.flatten(0, 1)[indices]
        batch.advantage = self.advantage.flatten(0, 1)[indices]
        batch.action_log_prob = self.action_log_prob.flatten(0, 1)[indices]
        return batch
