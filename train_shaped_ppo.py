
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium import RewardWrapper


class LandingRewardWrapper(RewardWrapper):
    """Add small rewards for stable, controlled landing progress."""

    def reward(self, reward):
        # The base LunarLander reward is preserved.
        return reward


class ShapedLunarLander(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_distance = np.linalg.norm(
            observation[:2] - np.array([0.0, 0.0])
        )
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # Encourage moving toward the landing area.
        distance = np.linalg.norm(observation[:2])

        if self.previous_distance is not None:
            progress = self.previous_distance - distance
            reward += 0.5 * progress

        self.previous_distance = distance

        # Small penalty for excessive movement near the ground.
        if observation[1] < 0.3:
            reward -= 0.1 * (
                abs(observation[2]) + abs(observation[3])
            )

        return observation, reward, terminated, truncated, info


env = ShapedLunarLander(gym.make("LunarLander-v3"))
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
)

print("Training reward-shaped PPO agent...")
model.learn(total_timesteps=200_000)

model.save("ppo_lunarlander_shaped")
env.close()

print("Training complete! Saved ppo_lunarlander_shaped.zip")
