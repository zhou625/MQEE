import copy
import numpy as np
import torch
import torch.nn.functional as F

from utils  import Actor, Critic
from models import BEE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
TAU   = 0.005
LR    = 3e-4

class TD3:
    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        max_action: float,
        use_bee:    bool = True,
        bee_kwargs: dict = None,
    ):
        self.actor           = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target    = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)

        self.critic           = Critic(state_dim, action_dim).to(device)
        self.critic_target    = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR)

        self.max_action = max_action
        self.total_it   = 0

        self.use_bee = use_bee
        if use_bee:
            self.bee = BEE(**(bee_kwargs or {}))

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(256)
        current_Q1, current_Q2 = self.critic(state, action)

        with torch.no_grad():
            if self.use_bee:
                target_Q = self.bee.compute_target(
                    rewards       = reward,
                    next_state    = next_state,
                    not_done      = not_done,
                    actor_target  = self.actor_target,
                    critic_target = self.critic_target,
                    max_action    = self.max_action,
                )
            else:
                noise       = (torch.randn_like(action) * 0.2 * self.max_action
                               ).clamp(-0.5 * self.max_action, 0.5 * self.max_action)
                next_action = (self.actor_target(next_state) + noise
                               ).clamp(-self.max_action, self.max_action)
                Q1, Q2      = self.critic_target(next_state, next_action)
                target_Q    = reward + not_done * GAMMA * torch.min(Q1, Q2)

        critic_loss = (F.mse_loss(current_Q1, target_Q)
                     + F.mse_loss(current_Q2, target_Q))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % 2 == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor,  self.actor_target)

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module):
        for param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(
                TAU * param.data + (1.0 - TAU) * target_param.data
            )