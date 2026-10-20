🚀 AI Rocket Landing Simulator
An experimental reinforcement learning project that trains and evaluates an AI agent to land a lunar lander using Proximal Policy Optimization (PPO). The project also includes reward-shaping experiments, model comparisons, and a Pygame visualization with both AI and manual flight modes.
🌕 Project Overview
The goal is to train an agent to control a lander in Gymnasium's LunarLander-v3 environment. The project compares baseline PPO training, a reward-shaped experiment, and extended PPO training, then visualizes the trained policy.

The cinematic moon scene is a visual layer. The flight physics and rewards come from Gymnasium's LunarLander environment.
✨ Features
- PPO reinforcement learning using Stable-Baselines3.
- Baseline and extended training experiments.
- Reward shaping experiment to explore how reward design affects learning.
- Evaluation and comparison scripts for measuring model performance.
- Pygame visualization with a custom lunar background.
- Manual flight mode with keyboard controls in play_moon_game.py.
- AI flight mode using the trained discrete-action PPO model.
- Result plots for comparing model performance.
🧰 Tech Stack
- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- Box2D
- Pygame / pygame-ce
- Pillow
- NumPy
- Matplotlib
📂 Project Structure
AI-Rocket-Landing-Simulator/
├── check_setup.py
├── train_ppo.py
├── train_shaped_ppo.py
├── continue_training.py
├── evaluate_ppo.py
├── evaluate_shaped_ppo.py
├── evaluate_20_episodes.py
├── compare_agents.py
├── compare_three_models.py
├── plot_results.py
├── watch_random_agent.py
├── watch_extended_agent.py
├── watch_moon_landing.py
├── play_moon_game.py
├── create_moon_background.py
├── moon_background.png
├── moon_scene.png
├── model_comparison.png
├── ppo_lunarlander.zip
├── ppo_lunarlander_shaped.zip
├── ppo_lunarlander_extended.zip
├── requirements.txt
├── .gitignore
└── README.md
The model files shown above are expected to be present if you want to run the pre-trained agents. If a model file is not included in a future version of the repository, train or obtain the corresponding model before running scripts that load it.
⚙️ Installation
1. Clone the repository
git clone https://github.com/soumyajit658/AI-Rocket-Landing-Simulator.git
cd AI-Rocket-Landing-Simulator
2. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation, you can use the environment's Python directly without activating it:
.\.venv\Scripts\python.exe -m pip install --upgrade pip
3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
If Box2D installation fails on your machine, check the Gymnasium Box2D installation guidance for your Python version and operating system.
🎮 Run the Game
Keep moon_scene.png and ppo_lunarlander_extended.zip in the same folder as the game scripts.
Play manually or watch the AI
python play_moon_game.py
- Press 1 for Manual Mode.
- Press 2 for AI PPO Mode.
- In Manual Mode: ↑ applies main thrust; ← and → steer.
- Press R to restart a flight.
- Press Esc to return to the menu.
Manual mode uses continuous controls; AI mode uses the discrete-action PPO environment expected by the trained model.
Watch the trained AI land
python watch_moon_landing.py
- Space: pause/resume.
- R: restart the episode.
- Esc: quit.
Other visualization scripts
python watch_extended_agent.py
python watch_random_agent.py
🧠 Train and Evaluate
Baseline PPO training
python train_ppo.py
Reward-shaped PPO experiment
python train_shaped_ppo.py
Continue training
python continue_training.py
Review each script's configuration before running it. Training may overwrite model files depending on the script, so back up any model you want to preserve.

Evaluate models
python evaluate_ppo.py
python evaluate_shaped_ppo.py
python evaluate_20_episodes.py
Compare models and generate plots
python compare_agents.py
python compare_three_models.py
python plot_results.py
📊 Experimental Results
In one evaluation of the extended PPO model over 20 episodes:
Metric	Result
Average reward	194.49
Median reward	253.45
Best reward	277.88
Episodes scoring at least 200	15 / 20


These values describe one evaluation run, not a guaranteed result. Scores can vary with random seeds, environment versions, and training settings.

In the experiments conducted, extended PPO performed better than the reward-shaped model. Reward shaping is an experiment in this project, not a claim that shaping always improves performance.
⚠️ Technical Notes
- The PPO policy is trained for Gymnasium's discrete-action LunarLander-v3 environment.
- Manual controls in play_moon_game.py use LunarLanderContinuous-v3; the trained discrete PPO model is not used to control that continuous environment.
- The moon background and landing-pad artwork are visual elements. They do not replace the underlying environment physics with a realistic lunar or 3D simulation.
- For reliable comparisons, evaluate models using the same episode seeds and environment configuration.
🚀 Future Improvements
- Improve reward functions and test across more random seeds.
- Add training curves and TensorBoard logging.
- Add a short gameplay demo or GIF.
- Improve flight HUD and landing analysis.
- Package the project for easier installation and execution.
- Explore more realistic physics or simulation environments.
👨‍💻 Author
Soumyajit Bhowmik

GitHub: @soumyajit658
If you find this project interesting, consider starring the repository.
