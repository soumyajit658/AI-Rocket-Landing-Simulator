
# 🚀 AI Rocket Landing with Reward Shaping

An experimental Reinforcement Learning project that trains an AI agent to perform lunar landings using Proximal Policy Optimization (PPO).

The project explores baseline PPO training, reward shaping, and extended training to improve landing performance in the Gymnasium LunarLander-v3 environment. It also includes a custom cinematic lunar landing visualization.

## 🌙 Project Overview

Landing a spacecraft safely requires controlling its position, velocity, and orientation. This project uses reinforcement learning to train an AI agent to make landing decisions through repeated interaction with a simulated environment.

The project compares different training approaches and evaluates their performance across multiple episodes.

## ✨ Key Features

- 🧠 **Reinforcement Learning:** PPO agent using Stable-Baselines3.
- 🚀 **Baseline Training:** Train an agent in the LunarLander-v3 environment.
- 🎯 **Reward Shaping Experiment:** Explore custom reward modifications to influence agent behaviour.
- 📈 **Extended Training:** Continue training a PPO model and evaluate its performance.
- 🧪 **Model Evaluation:** Compare rewards across multiple episodes.
- 📊 **Performance Visualization:** Generate model comparison plots.
- 🌕 **Custom Lunar Scene:** Display the trained agent in a cinematic moon-landing visualization using Pygame.

## 🛠️ Tech Stack

- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- Box2D
- Pygame
- Pillow
- Matplotlib

## 📂 Project Structure

```text
AI-Rocket-Landing-Reward-Shaping/
│
├── check_setup.py
├── train_ppo.py
├── train_shaped_ppo.py
├── continue_training.py
│
├── evaluate_ppo.py
├── evaluate_shaped_ppo.py
├── evaluate_20_episodes.py
│
├── compare_agents.py
├── compare_three_models.py
├── plot_results.py
│
├── watch_random_agent.py
├── watch_extended_agent.py
├── watch_moon_landing.py
│
├── create_moon_background.py
├── moon_background.png
├── moon_scene.png
├── model_comparison.png
│
├── ppo_lunarlander.zip
├── ppo_lunarlander_shaped.zip
├── ppo_lunarlander_extended.zip
│
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/soumyajit658/AI-Rocket-Landing-Reward-Shaping.git
cd AI-Rocket-Landing-Reward-Shaping
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install gymnasium stable-baselines3 pygame-ce pillow matplotlib swig
python -m pip install box2d-py
```

If Box2D installation fails, consult the Gymnasium Box2D installation instructions for your Python and operating-system version.

## ▶️ Run the Lunar Landing Visualization

Make sure these files are available in the project directory:

- `watch_moon_landing.py`
- `moon_scene.png`
- `ppo_lunarlander_extended.zip`

Run:

```powershell
python watch_moon_landing.py
```

The visualization displays the trained agent with a custom lunar background and a flight-style interface.

## 🧠 Training Models

### Baseline PPO

```powershell
python train_ppo.py
```

### Reward-Shaped PPO

```powershell
python train_shaped_ppo.py
```

### Evaluate the Models

```powershell
python evaluate_ppo.py
python evaluate_shaped_ppo.py
python evaluate_20_episodes.py
```

Training scripts may overwrite existing model files, depending on their implementation. Back up important trained models before retraining.

## 📊 Experimental Results

In one evaluation of the extended PPO model across 20 episodes:

| Metric | Result |
|---|---:|
| Average reward | 194.49 |
| Median reward | 253.45 |
| Best reward | 277.88 |
| Episodes scoring at least 200 | 15 / 20 |

These results are from a particular evaluation run. Performance can vary depending on random seeds, environment settings, and training configuration.

The reward-shaped experiment and extended PPO model are separate approaches; reward shaping did not automatically produce better results in the experiments conducted.

## 🔬 Learning Outcomes

Through this project, I explored:

- Reinforcement learning fundamentals
- PPO-based policy training
- Reward shaping and its limitations
- Model evaluation and reward analysis
- Comparing training approaches
- Visualizing an AI agent using Pygame
- Building and documenting an end-to-end machine learning experiment

## ⚠️ Important Technical Note

The AI policy is trained in the standard Gymnasium LunarLander-v3 environment. The custom lunar scene is a visualization layer; it does not replace the underlying physics with a realistic 3D lunar simulation.

## 🚀 Future Improvements

- Experiment with more effective reward functions.
- Evaluate agents across additional random seeds.
- Track training performance with TensorBoard.
- Improve landing-position and touchdown analysis.
- Add more interactive visualization features.
- Explore more realistic physics and simulation environments.

## 👨‍💻 Author

**Soumyajit Bhowmik**

GitHub: [@soumyajit658](https://github.com/soumyajit658)

---

⭐ If you find this project interesting, consider giving the repository a star!
