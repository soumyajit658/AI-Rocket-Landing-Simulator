
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("LunarLander-v3"))

# Continue learning from the original PPO model
model = PPO.load("ppo_lunarlander", env=env)

print("Continuing PPO training for 500,000 additional timesteps...")

model.learn(
    total_timesteps=500_000,
    reset_num_timesteps=False,
)

model.save("ppo_lunarlander_extended")
env.close()

print("Training complete!")
print("Saved model: ppo_lunarlander_extended.zip")

