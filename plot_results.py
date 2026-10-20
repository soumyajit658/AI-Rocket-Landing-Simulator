
import matplotlib.pyplot as plt

models = ["Reward-shaped PPO", "Extended PPO"]
average_rewards = [-79.93, 194.49]
median_rewards = [-75.73, 253.45]
success_episodes = [0, 15]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(models, average_rewards)
axes[0].set_title("Average Episode Reward")
axes[0].set_ylabel("Reward")
axes[0].axhline(0, linewidth=0.8)
axes[0].tick_params(axis="x", rotation=12)

for i, value in enumerate(average_rewards):
    axes[0].text(i, value, f"{value:.2f}", ha="center",
                 va="bottom" if value >= 0 else "top")

axes[1].bar(models, success_episodes)
axes[1].set_title("Episodes Scoring 200 or Higher")
axes[1].set_ylabel("Episodes out of 20")
axes[1].set_ylim(0, 20)
axes[1].tick_params(axis="x", rotation=12)

for i, value in enumerate(success_episodes):
    axes[1].text(i, value + 0.5, f"{value}/20", ha="center")

plt.suptitle("LunarLander PPO Model Comparison")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
