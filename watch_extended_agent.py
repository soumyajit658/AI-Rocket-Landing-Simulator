
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "ppo_lunarlander_extended"

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load(MODEL_PATH)

episodes = 5

try:
    for episode in range(1, episodes + 1):
        obs, info = env.reset(seed=episode)
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(
            f"Episode {episode}: "
            f"Reward = {total_reward:.2f}"
        )
finally:
    env.close()
