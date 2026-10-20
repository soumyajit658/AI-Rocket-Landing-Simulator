
import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_lunarlander_shaped")
env = gym.make("LunarLander-v3", render_mode="human")

num_episodes = 5
rewards = []

print("Evaluating reward-shaped PPO agent...")

try:
    for episode in range(1, num_episodes + 1):
        observation, info = env.reset(seed=episode)
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(
                observation, deterministic=True
            )
            observation, reward, terminated, truncated, info = (
                env.step(action)
            )
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode}: Environment reward = {total_reward:.2f}")

    print("-" * 40)
    print(f"Average environment reward: {sum(rewards) / len(rewards):.2f}")

finally:
    env.close()

print("Evaluation complete!")
