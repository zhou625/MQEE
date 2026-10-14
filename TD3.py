import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Actor, Critic
from models import RewardCentering, BEE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, use_bee=True, bee_kwargs=None):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.total_it = 0
        self.rc = RewardCentering(gamma=0.99, eta=0.01, step_size=0.1, avg_reward_init=0.0)
        self.use_bee = use_bee

        if use_bee:
            kwargs = bee_kwargs or {}
            self.bee = BEE(**kwargs)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(256)

        current_Q1, current_Q2 = self.critic(state, action)

        with torch.no_grad():
            if self.use_bee:
                next_q_mixed = self.bee.compute_mixed_next_value(
                    not_done, next_state,
                    self.actor_target, self.critic_target, self.max_action
                )
            else:
                noise = (torch.randn_like(action) * 0.2 * self.max_action).clamp(
                    -0.5 * self.max_action, 0.5 * self.max_action)
                next_action = (self.actor_target(next_state) + noise).clamp(
                    -self.max_action, self.max_action)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                next_q_mixed = not_done * torch.min(target_Q1, target_Q2)

        adjusted_target = self.rc.update(
            rewards=reward,
            next_q=next_q_mixed,
            pred_q=current_Q1.detach()
        )

        critic_loss = (F.mse_loss(current_Q1, adjusted_target) + F.mse_loss(current_Q2, adjusted_target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % 2 == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)