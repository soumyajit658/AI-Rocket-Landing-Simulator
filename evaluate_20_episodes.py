
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np


def evaluate(model_path, seeds):
    model = PPO.load(model_path)
    env = gym.make("LunarLander-v3")
    rewards = []

    try:
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            total_reward = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs, deterministic=True
                )
                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )
                total_reward += reward

            rewards.append(total_reward)
    finally:
        env.close()

    return np.array(rewards)


seeds = list(range(100, 120))

original = evaluate("ppo_lunarlander", seeds)
shaped = evaluate("ppo_lunarlander_shaped", seeds)


def report(name, rewards):
    print(f"\n{name}")
    print(f"Average reward: {rewards.mean():.2f}")
    print(f"Median reward:  {np.median(rewards):.2f}")
    print(f"Best reward:    {rewards.max():.2f}")
    print(f"Worst reward:   {rewards.min():.2f}")
    print(f"Scores >= 200:  {np.sum(rewards >= 200)}/20")


report("Original PPO", original)
report("Reward-shaped PPO", shaped)

print("\nAverage reward difference (shaped - original):")
print(f"{shaped.mean() - original.mean():.2f}")
