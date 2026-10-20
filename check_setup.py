
import gymnasium as gym
import stable_baselines3
import torch

print("All libraries imported successfully!")
print("Gymnasium:", gym.__version__)
print("Stable-Baselines3:", stable_baselines3.__version__)
print("PyTorch:", torch.__version__)

env = gym.make("LunarLander-v3")
observation, info = env.reset(seed=42)

print("Observation:", observation)
print("Observation shape:", observation.shape)
print("Action space:", env.action_space)

env.close()
print("Setup successful!")
