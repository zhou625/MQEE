import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BEE:
    def __init__(
        self,
        quantile:      float = 0.7,
        lambda_method: str   = "fixed_0.5",
        gamma:         float = 0.99,
        num_samples:   int   = 10,
    ):
        self.quantile      = quantile
        self.lambda_method = lambda_method
        self.gamma         = gamma
        self.num_samples   = num_samples

    def _parse_lambda(self) -> float:
        if self.lambda_method.startswith("fixed_"):
            return float(self.lambda_method[6:])
        raise NotImplementedError(f"未实现的 lambda 方法: {self.lambda_method}")

    def compute_target(
        self,
        rewards:        torch.Tensor,   # [B×1]  当前步奖励
        next_state:     torch.Tensor,   # [B×S]  下一状态
        not_done:       torch.Tensor,   # [B×1]  终止掩码
        actor_target,                   # Target Actor 网络
        critic_target,                  # Target Critic 网络
        max_action:     float,
    ) -> torch.Tensor:

        batch_size = next_state.shape[0]
        lam        = self._parse_lambda()

        with torch.no_grad():

            # ── Step 1  扩展状态，为每个 next_state 准备 N 份副本 ──────
            next_state_rep = next_state.repeat_interleave(
                self.num_samples, dim=0
            )                                               # [B·N × S]

            # ── Step 2  Target Policy Smoothing ────────────────────────
            base_action = actor_target(next_state_rep)      # [B·N × A]
            noise       = (
                torch.randn_like(base_action) * 0.2 * max_action
            ).clamp(-0.5 * max_action, 0.5 * max_action)
            next_action = (base_action + noise).clamp(-max_action, max_action)

            # ── Step 3  Clipped Double Q → Q 值矩阵 ────────────────────
            Q1, Q2  = critic_target(next_state_rep, next_action)
            Q_mat   = torch.min(Q1, Q2).view(batch_size, self.num_samples)  # [B×N]

            # ── Step 4  双统计量提取 ────────────────────────────────────
            V_tau = torch.quantile(Q_mat, self.quantile, dim=1, keepdim=True)  # [B×1] 保守分位
            Q_bar = Q_mat.mean(dim=1, keepdim=True)                            # [B×1] 无偏均值

            # ── Step 5  批次中心化（每 batch 即时计算，无跨步状态）──────
            V_tau_centered = V_tau - V_tau.mean()   # 消除批次内绝对量级偏移
            Q_bar_centered = Q_bar - Q_bar.mean()

            # ── Step 6  λ 加权混合 ─────────────────────────────────────
            V_mix = lam * V_tau_centered + (1.0 - lam) * Q_bar_centered  # [B×1]

            # ── 标准折扣 Bellman 目标（无奖励中心化）──────────────────
            next_value    = not_done * V_mix                              # 终止状态归零
            target_return = rewards + self.gamma * next_value             # y = r + γ·V_mix

        return target_return.detach()