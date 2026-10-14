import os
import time
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import shimmy
import torch

import TD3
import utils

os.makedirs("./results", exist_ok=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)

START_TIMESTEPS = 25e3      # 随机探索步数,之后开始训练
MAX_TIMESTEPS = int(1e6)    # 总训练步数
EVAL_FREQ = 5e3             # 评估间隔
LOG_FREQ = 1e4              # 打印间隔

NOISE_STD = 0.3             # 探索噪声初始标准差(相对 max_action)
NOISE_STD_MIN = 0.05        # 探索噪声下限
NOISE_DECAY = 0.9999995     # 每步衰减系数


def train(policy_name, env_name, seed, fig_id, lambda_val):
    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    last_print_time = time.time()  # 用于计算平均单步耗时

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3.TD3(
        state_dim, action_dim, max_action,
        use_dq=True,
        dq_kwargs={
            "quantile": 0.7,
            "lambda_method": f"fixed_{lambda_val}",
            "gamma": 0.99,
        },
    )

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    evaluations = [utils.eval_policy(policy, env_name, seed)]
    state, _ = env.reset(seed=seed)

    safe_env_name = env_name.replace("/", "_")
    file_name = f"{policy_name}_{safe_env_name[0:-3]}_lam{lambda_val}_seed{seed}"

    for t in range(MAX_TIMESTEPS + 1):
        # 选择动作:前期随机探索,之后策略 + 衰减高斯噪声
        if t < START_TIMESTEPS:
            action = env.action_space.sample()
        else:
            current_noise = max(NOISE_STD_MIN, NOISE_STD * (NOISE_DECAY ** (t - START_TIMESTEPS)))
            action = (policy.select_action(np.array(state))
                      + np.random.normal(0, max_action * current_noise, size=action_dim)
                      ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_bool = float(terminated)
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state

        if t >= START_TIMESTEPS:
            policy.train(replay_buffer)

        if done:
            state, _ = env.reset()

        # 评估与日志
        if t % EVAL_FREQ == 0 and t > 0:
            R = utils.eval_policy(policy, env_name, seed)
            evaluations.append(R)

            if t % LOG_FREQ == 0:
                current_time = time.time()
                avg_step_time = (current_time - last_print_time) / 1e4
                last_print_time = current_time

                print(f"Step: {t}  Reward: {R:.1f}  Avg Step Time: {avg_step_time:.4f}s")

                if t % 1e5 == 0:
                    print(f"------------  {round((time.time() - start_time) / 60., 1)} min  ------------")

                if t == MAX_TIMESTEPS:
                    np.save(f"./results/{file_name}", evaluations)
                    plt.figure(fig_id)
                    plt.plot(evaluations, 'b')
                    plt.savefig("./results/" + file_name + "-" + str(round(np.max(evaluations))) + '.png')
                    plt.close(fig_id)


if __name__ == "__main__":
    policy_name = "TD3"
    seeds = [1, 2, 3]

    # "Walker2d-v4", "Humanoid-v4", "HalfCheetah-v4"
    env_list = ["Ant-v4"]
    lambda_list = [0, 1]

    exp_id = 0
    for env_name in env_list:
        for lam in lambda_list:
            for seed in seeds:
                exp_id += 1
                print(f"\n>>> 实验进度: {exp_id}")
                print(f">>> 环境: {env_name} | Lambda: {lam} | Seed: {seed}")

                try:
                    train(policy_name, env_name, seed, exp_id, lambda_val=lam)
                except Exception as e:
                    print(f"!!! 实验 {exp_id} 失败: {e}")
                    continue