import torch, utils, TD3, time
import matplotlib.pyplot as plt
import shimmy
import gymnasium as gym
import numpy as np
import warnings
import os

os.makedirs("./results", exist_ok=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train(Policy, Env, Seed, j, lambda_val):
    noise_std = 0.3
    noise_std_min = 0.05
    noise_decay = 0.9999995  # 每步衰减系数
    env = gym.make(Env)
    env = gym.wrappers.FlattenObservation(env)
    env.action_space.seed(Seed)
    torch.manual_seed(Seed)
    np.random.seed(Seed)
    start_time = time.time()

    # 记录上一次打印的时间，用于计算平均单步耗时
    last_print_time = time.time()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy = TD3.TD3(state_dim, action_dim, max_action, use_bee=True,
                     bee_kwargs={
                         "quantile": 0.7,
                         "lambda_method": f"fixed_{lambda_val}",  # 动态设置
                         "gamma": 0.99
                     })

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    evaluations = [utils.eval_policy(policy, Env, Seed)]
    state, _ = env.reset(seed=Seed)
    episode_reward, episode_timesteps, episode_num = 0, 0, 0

    # 因为 DMControl 环境名带斜杠(如 dm_control/humanoid-stand-v0)，这里替换成下划线避免路径报错
    safe_env_name = Env.replace("/", "_")
    file_name = f"{Policy}_{safe_env_name[0:-3]}_lam{lambda_val}_seed{Seed}"

    for t in range(int(1e6 + 1)):
        episode_timesteps += 1

        if t < 25e3:
            action = env.action_space.sample()
        else:
            current_noise = max(noise_std_min, noise_std * (noise_decay ** (t - 25e3)))
            action = (policy.select_action(np.array(state))
                      + np.random.normal(0, max_action * current_noise, size=action_dim)
                      ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_bool = float(terminated)
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state  # 更新状态
        episode_reward += reward

        if t >= 25e3:
            policy.train(replay_buffer)

        if done:
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if t % 5e3 == 0 and t > 0:
            R = utils.eval_policy(policy, Env, Seed)
            evaluations.append(R)
            if t % 1e4 == 0:
                # 计算过去 10000 步的平均单步耗时
                current_time = time.time()
                avg_step_time = (current_time - last_print_time) / 1e4
                last_print_time = current_time  # 重置计时器

                print(f"Step: {t}  Reward: {R:.1f}  Avg Step Time: {avg_step_time:.4f}s")

                if t % 1e5 == 0:
                    print(f"------------  {round((time.time() - start_time) / 60., 1)} min  ------------")
                if t == int(1e6):
                    np.save(f"./results/{file_name}", evaluations)
                    plt.figure(j)
                    plt.plot(evaluations, 'b')
                    plt.savefig("./results/" + file_name + "-" + str(round(np.max(evaluations))) + '.png')
                    plt.close(j)


if __name__ == "__main__":
    Policy = "TD3"
    Seeds = np.array([1, 2, 3])

    # 在列表中增加了 DMControl 的 Humanoid 实验
    # "Walker2d-v4", "Humanoid-v4", "HalfCheetah-v4"
    EnvA = [
         "Ant-v4"
    ]

    Lambda_List = [0,1]
    j = 0

    for en in range(len(EnvA)):
        Env = EnvA[en]
        for lam in Lambda_List:  # 第一层参数循环
            for se in range(Seeds.shape[0]):  # 第二层种子循环
                j += 1
                Seedi = int(Seeds[se])
                print(f"\n>>> 实验进度: {j}")
                print(f">>> 环境: {Env} | Lambda: {lam} | Seed: {Seedi}")

                try:
                    train(Policy, Env, Seedi, j, lambda_val=lam)
                except Exception as e:
                    print(f"!!! 实验 {j} 失败: {e}")
                    continue  # 修复了你代码里缺少的