
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)
total_reward = 0
episode = 1

print("Starting LunarLander simulation...")

try:
    while episode <= 3:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode {episode} finished!")
            print(f"Total reward: {total_reward:.2f}")
            print("-" * 30)

            episode += 1
            total_reward = 0

            if episode <= 3:
                observation, info = env.reset()

finally:
    env.close()
    print("Simulation finished.")
