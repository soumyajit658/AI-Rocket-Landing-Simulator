
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


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
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

            rewards.append(total_reward)
    finally:
        env.close()

    return np.array(rewards)


seeds = list(range(100, 120))

models = {
    "Original PPO": "ppo_lunarlander",
    "Reward-shaped PPO": "ppo_lunarlander_shaped",
    "Extended PPO": "ppo_lunarlander_extended",
}

results = {}

for name, path in models.items():
    scores = evaluate(path, seeds)
    results[name] = scores

    print(f"\n{name}")
    print(f"Average reward: {scores.mean():.2f}")
    print(f"Median reward:  {np.median(scores):.2f}")
    print(f"Best reward:    {scores.max():.2f}")
    print(f"Scores >= 200:  {np.sum(scores >= 200)}/20")

print("\nAverage reward differences:")
print(
    "Extended PPO - Reward-shaped PPO:",
    f"{results['Extended PPO'].mean() - results['Reward-shaped PPO'].mean():.2f}"
)
