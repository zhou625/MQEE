import os
import time
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import TD3
import utils

os.makedirs("./results", exist_ok=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)

TOTAL_STEPS    = int(1e6)
WARMUP_STEPS   = int(25e3)
EVAL_INTERVAL  = int(5e3)      # 每隔多少步做一次评估
PRINT_INTERVAL = int(1e4)      # 每隔多少步打印一次进度
LOG_INTERVAL   = int(1e5)      # 每隔多少步打印耗时

NOISE_INIT     = 0.3
NOISE_MIN      = 0.05
NOISE_DECAY    = 0.9999995     # 每步衰减系数


def train(Policy, Env, Seed, fig_idx, lambda_val):

    env = gym.make(Env)
    env.action_space.seed(Seed)
    torch.manual_seed(Seed)
    np.random.seed(Seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3.TD3(
        state_dim, action_dim, max_action,
        use_bee    = True,
        bee_kwargs = {
            "quantile":      0.7,
            "lambda_method": f"fixed_{lambda_val}",
            "gamma":         0.99,
        }
    )
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    file_name   = f"{Policy}_{Env[:-3]}_lam{lambda_val}_seed{Seed}"
    evaluations = [utils.eval_policy(policy, Env, Seed)]
    R           = evaluations[0]          # 避免首次打印时 UnboundLocalError

    start_time = time.time()

    state, _          = env.reset(seed=Seed)
    episode_reward    = 0.0
    episode_timesteps = 0
    episode_num       = 0

    for t in range(TOTAL_STEPS + 1):
        episode_timesteps += 1

        # 动作选取：预热期随机探索，之后带衰减噪声的策略动作
        if t < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            decay_steps = t - WARMUP_STEPS
            noise_std   = max(NOISE_MIN, NOISE_INIT * (NOISE_DECAY ** decay_steps))
            action      = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * noise_std, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done      = terminated or truncated
        done_bool = float(terminated)

        replay_buffer.add(state, action, next_state, reward, done_bool)
        state          = next_state
        episode_reward += reward

        if t >= WARMUP_STEPS:
            policy.train(replay_buffer)

        if done:
            state, _          = env.reset()
            episode_reward    = 0.0
            episode_timesteps = 0
            episode_num       += 1

        if t % EVAL_INTERVAL == 0 and t > 0:
            R = utils.eval_policy(policy, Env, Seed)
            evaluations.append(R)

        if t % PRINT_INTERVAL == 0:
            elapsed_min = round((time.time() - start_time) / 60.0, 1)
            print(f"Step: {t:>8d}  Reward: {R:>8.1f}  Elapsed: {elapsed_min} min")

        if t % LOG_INTERVAL == 0 and t > 0:
            print(f"{'─'*50}")

        if t == TOTAL_STEPS:
            np.save(f"./results/{file_name}", evaluations)
            plt.figure(fig_idx)
            plt.plot(evaluations, 'b')
            plt.savefig(
                f"./results/{file_name}-{round(np.max(evaluations))}.png"
            )
            plt.close(fig_idx)

if __name__ == "__main__":
    Policy      = "TD3"
    Seeds       = np.array([0, 1, 2, 3])
    EnvA        = ["HalfCheetah-v4", "Hopper-v4"]
    Lambda_List = [0.3, 0.7]

    fig_idx = 0
    for Env in EnvA:
        for lam in Lambda_List:
            for Seed in Seeds:
                fig_idx += 1
                print(f"\n>>> 实验进度: {fig_idx}")
                print(f">>> 环境: {Env} | Lambda: {lam} | Seed: {int(Seed)}")
                try:
                    train(Policy, Env, int(Seed), fig_idx, lambda_val=lam)
                except Exception as e:
                    print(f"!!! 实验 {fig_idx} 失败: {e}")