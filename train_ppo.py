
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Create the LunarLander environment
env = gym.make("LunarLander-v3")
env = Monitor(env)

# Create the PPO reinforcement learning agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
)

print("Training started!")

# Train the agent
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_lunarlander")

env.close()
print("Training complete! Model saved as ppo_lunarlander.zip")
