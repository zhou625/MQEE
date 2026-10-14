import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DC:
    def __init__(self, gamma=0.99, eta=0.01, step_size=1.0, avg_reward_init=0.0):
        self.device = device
        self.gamma = torch.tensor(gamma).float().to(device)
        self.eta = torch.tensor(eta).float().to(device)
        self.step_size = step_size
        self.avg_reward_init = avg_reward_init
        self.avg_reward = torch.tensor(avg_reward_init, requires_grad=False,
                                       dtype=torch.float, device=device)
        self.beta = self.eta * self.step_size

    def update(self, rewards, next_q, pred_q):
        rewards = rewards.to(self.device)
        next_q = next_q.to(self.device)
        pred_q = pred_q.to(self.device)
        with torch.no_grad():
            target_return = rewards - self.avg_reward + self.gamma * next_q
            delta = target_return - pred_q
            self.avg_reward += self.beta * delta.mean()
        return target_return.detach()


class DQ:
    def __init__(self, quantile=0.7, lambda_method="fixed_0.7", gamma=0.99, num_samples=10):
        self.quantile = quantile
        self.lambda_method = lambda_method
        self.gamma = gamma
        self.num_samples = num_samples

    def compute_mixed_next_value(self, not_done, next_state, actor_target, critic_target, max_action):
        batch_size = next_state.shape[0]

        with torch.no_grad():
            next_state_rep = next_state.repeat_interleave(self.num_samples, dim=0)
            base_next_action = actor_target(next_state_rep)
            noise = (torch.randn_like(base_next_action) * 0.2 * max_action).clamp(
                -0.5 * max_action, 0.5 * max_action)
            next_action_rep = (base_next_action + noise).clamp(-max_action, max_action)

            target_Q1, target_Q2 = critic_target(next_state_rep, next_action_rep)
            target_Q = torch.min(target_Q1, target_Q2).view(batch_size, self.num_samples)
            next_v = torch.quantile(target_Q, self.quantile, dim=1, keepdim=True)  # 分位数
            next_q_e = target_Q.mean(dim=1, keepdim=True)                          # 均值

            if self.lambda_method.startswith("fixed_"):
                lam = float(self.lambda_method[6:])
            else:
                raise NotImplementedError(f"未实现的 lambda 方法: {self.lambda_method}")

            mixed_next_v = lam * next_v + (1.0 - lam) * next_q_e
            return not_done * mixed_next_v