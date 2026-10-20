
import gymnasium as gym
from stable_baselines3 import PPO


def evaluate_model(model_path, seeds):
    model = PPO.load(model_path)
    env = gym.make("LunarLander-v3")
    rewards = []

    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)
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
    finally:
        env.close()

    return rewards


seeds = [1, 2, 3, 4, 5]

original = evaluate_model("ppo_lunarlander", seeds)
shaped = evaluate_model("ppo_lunarlander_shaped", seeds)

print("\nOriginal PPO rewards:", [round(r, 2) for r in original])
print("Shaped PPO rewards:  ", [round(r, 2) for r in shaped])

print(f"\nOriginal PPO average: {sum(original) / len(original):.2f}")
print(f"Shaped PPO average:   {sum(shaped) / len(shaped):.2f}")

