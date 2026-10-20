
import gymnasium as gym
from stable_baselines3 import PPO

# Load the trained AI model
model = PPO.load("ppo_lunarlander")

# Create a visible LunarLander environment
env = gym.make("LunarLander-v3", render_mode="human")

num_episodes = 5
episode_rewards = []

print("Evaluating the trained PPO agent...")

try:
    for episode in range(1, num_episodes + 1):
        observation, info = env.reset(seed=episode)
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Let the trained agent choose an action
            action, _states = model.predict(
                observation, deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {episode}: Reward = {total_reward:.2f}")

    average_reward = sum(episode_rewards) / len(episode_rewards)
    print("-" * 35)
    print(f"Average reward over {num_episodes} episodes: {average_reward:.2f}")

finally:
    env.close()

print("Evaluation complete!")
