from typing import Union

import torch

from rl_solver.buffer import RolloutBuffer
from rl_solver.common import Actor, CustomActor_B, ValueNet, Algorithm


class PPO(Algorithm):
    MODULES = ['actor', 'critic']
    OPTIMIZERS = ['optimizer']

    def __init__(
        self,
        actor: Actor,
        critic: ValueNet,
        num_collects: int,
        num_subepochs: int,
        trans_min: float,
        trans_max: float,
        num_mini_batches: int = 1,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        normalize_adv=True,
        clip_value_loss=True,
        device: Union[torch.device, str] = None,
    ):
        # PPO components
        super().__init__(actor.action_dim, num_collects, device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # PPO parameters
        self.clip_ratio = clip_ratio
        self.num_subepochs = num_subepochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_adv = normalize_adv
        self.clip_value_loss = clip_value_loss

        # ADAM
        self.learning_rate = learning_rate
        self.optimizer = self._make_optimizer()

        # learning_rate
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=(1/pow(10,1/10)))

        # storage
        self.storage: RolloutBuffer = self.init_storage()
        self.transition = self.storage.Transition()

        self.trans_min = trans_min
        self.trans_max = trans_max
    def _make_optimizer(self):
        params = [{'params': self.actor.parameters(), 'lr': self.learning_rate},
                  {'params': self.critic.parameters(), 'lr': self.learning_rate}]
        return torch.optim.Adam(params, lr=self.learning_rate)

    def init_storage(self):
        return RolloutBuffer(self.num_collects, self.device)

    @torch.no_grad()
    def act(self, num_parallels=1):
        t = self.transition
        t.action_mean, t.action_std, t.action, t.action_log_prob = self.actor.explore(num_parallels=num_parallels)
        t.value = self.critic(num_parallels=num_parallels)
        return t.action.cpu().numpy()

    @torch.no_grad()
    def step(self, reward):
        self.transition.reward = reward
        self.storage.add_transition(self.transition)
        self.transition.__init__()

    def update(self, warmup=False, entropy_large=False):
        summary = self._warmup_step() if warmup else self._train_step(entropy_large)
        self.storage.clear()
        self.scheduler.step()
        summary.update({
            'PPO/learning_rate': self.optimizer.param_groups[0]['lr'],
            'PPO/entropy': self.entropy_coef,
        })
        return summary

    def _warmup_step(self):
        for batch in self.storage.sampler(self.num_mini_batches, self.num_subepochs):
            value_loss = self._calc_value_loss(batch)
            self.optim_step(self.optimizer, value_loss, self.max_grad_norm)

            self.stats.add(
                value=batch.curr_value,
                value_loss=value_loss,
                surrogate=0.,
                ratio=0.
            )

        return self.stats.get(prefix='PPO/', clear=True)

    def _train_step(self, entropy_large=False):
        if entropy_large:
            entropy_coef = self.entropy_coef/2
        else: 
            entropy_coef = self.entropy_coef
        for idx, batch in enumerate(self.storage.sampler(self.num_mini_batches, self.num_subepochs)):
            # Surrogate loss
            surrogate_loss, curr_entropy, ratio = self._calc_surrogate_loss(batch)

            value_loss = self._calc_value_loss(batch)
            loss = surrogate_loss + value_loss - entropy_coef * curr_entropy.mean()
            if not isinstance(self.actor, CustomActor_B):
                loss += 0.1 * ((batch.curr_action_mean[..., 3:].norm(dim=-1) - 1) ** 2).mean()

            # Gradient step
            self.optim_step(self.optimizer, loss, self.max_grad_norm)
            if isinstance(self.actor, CustomActor_B):
                self.actor.p_distribution.clamp_std(lb=1e-3, ub=self.trans_max/2)
                self.actor.state_vector.clamp(self.trans_min, self.trans_max)
                # self.actor.r_distribution.clamp_std(lb=-torch.inf, ub=-50)
                bingham_M, bingham_Z = self.actor.r_distribution(self.actor()[3:].reshape(-1, 4, 4))
                bingham_x = bingham_M[..., -1]
                action_std = batch.curr_action_std[..., :3]
                self.stats.add(
                    value=batch.curr_value,
                    value_loss=value_loss,
                    surrogate=surrogate_loss,
                    ratio=torch.abs(ratio - 1.0),
                    action_std=action_std,
                    M = bingham_M,
                    Z = bingham_Z,
                    bh_X = bingham_x
                )
            elif isinstance(self.actor, Actor):
                self.actor.distribution.clamp_std(lb=1e-3, ub=self.trans_max/2)
                self.actor.state_vector.clamp(self.trans_min, self.trans_max, slice(0,3))

                self.stats.add(
                    value=batch.curr_value,
                    value_loss=value_loss,
                    surrogate=surrogate_loss,
                    ratio=torch.abs(ratio - 1.0),
                    action_std=batch.curr_action_std,
                )
        return self.stats.get(prefix='PPO/', clear=True)

    def _calc_surrogate_loss(self, batch: RolloutBuffer.Batch):
        batch.curr_action_mean, batch.curr_action_std = self.actor.explore(sample=False)
        curr_action_log_prob, curr_entropy = self.actor.calc_log_prob_entropy(
            batch.curr_action_mean, batch.curr_action_std, batch.action)

        ratio = torch.exp(curr_action_log_prob - batch.action_log_prob)
        surrogate_loss = -torch.min(
            batch.advantage * ratio,
            batch.advantage * ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio),
        ).mean()
        return surrogate_loss, curr_entropy, ratio

    def _calc_value_loss(self, batch: RolloutBuffer.Batch):
        batch.curr_value = self.critic()

        if self.clip_value_loss:
            value_diff = batch.curr_value - batch.value
            value_clipped = batch.value + value_diff.clamp(-self.clip_ratio, self.clip_ratio)
            value_loss = (batch.curr_value - batch.reward).pow(2)
            value_loss_clipped = (value_clipped - batch.reward).pow(2)
            value_loss = torch.max(value_loss, value_loss_clipped).mean()
        else:
            value_loss = (batch.reward - batch.curr_value).pow(2).mean()
        return value_loss * self.value_loss_coef
