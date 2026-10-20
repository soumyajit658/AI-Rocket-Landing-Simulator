
import math
from pathlib import Path

import gymnasium as gym
import pygame
from PIL import Image
from stable_baselines3 import PPO


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

WIDTH = 1200
HEIGHT = 800
FPS = 10

PROJECT_DIR = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_DIR / "moon_scene.png"
MODEL_PATH = PROJECT_DIR / "ppo_lunarlander_extended.zip"

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Moon Landing Challenge")
clock = pygame.time.Clock()

font = pygame.font.SysFont("consolas", 22)
small_font = pygame.font.SysFont("consolas", 16)
title_font = pygame.font.SysFont("consolas", 38, bold=True)
big_font = pygame.font.SysFont("consolas", 48, bold=True)


# --------------------------------------------------
# LOAD THE BACKGROUND
# --------------------------------------------------

if not SCENE_PATH.exists():
    pygame.quit()
    raise SystemExit(
        "moon_scene.png is missing. Put it beside play_moon_game.py."
    )

image = Image.open(SCENE_PATH).convert("RGB")
image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

background = pygame.image.frombytes(
    image.tobytes(), image.size, "RGB"
)


# --------------------------------------------------
# LOAD THE TRAINED AI
# --------------------------------------------------

try:
    ai_model = PPO.load(str(MODEL_PATH))
except Exception as error:
    pygame.quit()
    raise SystemExit(
        f"Could not load {MODEL_PATH.name}: {error}"
    )


# Keep the original LunarLander physics.
env = gym.make("LunarLander-v3")


# --------------------------------------------------
# GAME STATE
# --------------------------------------------------

mode = None
obs = None
episode_reward = 0.0
game_score = 0
fuel = 100.0
done = True
episode_number = 1
last_action = 0
message = "Choose your game mode"
message_color = (255, 255, 255)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def draw_text(text, x, y, color=(255, 255, 255), selected_font=font):
    surface = selected_font.render(str(text), True, color)
    screen.blit(surface, (x, y))


def draw_panel(x, y, width, height):
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((2, 12, 28, 220))
    screen.blit(panel, (x, y))
    pygame.draw.rect(
        screen, (38, 190, 240),
        (x, y, width, height), 2, border_radius=12
    )


def start_game(selected_mode):
    global mode, obs, episode_reward, game_score
    global fuel, done, episode_number, last_action
    global message, message_color

    mode = selected_mode
    obs, _ = env.reset()
    episode_reward = 0.0
    game_score = 0
    fuel = 100.0
    done = False
    episode_number += 1
    last_action = 0
    message = "Mission in progress"
    message_color = (80, 255, 180)


def return_to_menu():
    global mode, obs, done, message
    mode = None
    obs = None
    done = True
    message = "Choose your game mode"


def finish_game():
    global done, game_score, message, message_color

    done = True

    # Gymnasium observation:
    # [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
    x, y, vx, vy, angle = obs[:5]
    left_leg = obs[6] > 0.5
    right_leg = obs[7] > 0.5

    safe_landing = (
        left_leg
        and right_leg
        and abs(vx) < 0.25
        and abs(vy) < 0.35
        and abs(angle) < 0.20
    )

    if safe_landing:
        game_score = max(
            100,
            int(1000 - abs(vx) * 500 - abs(vy) * 500
                - abs(angle) * 300)
        )
        message = "MISSION SUCCESS!"
        message_color = (50, 255, 130)
    else:
        game_score = 0
        message = "MISSION FAILED!"
        message_color = (255, 80, 80)


def draw_lander(x, y, angle, action):
    """Draw a stylized rocket that follows the simulated lander."""

    # Build the rocket on its own transparent surface.
    ship = pygame.Surface((100, 140), pygame.SRCALPHA)

    # Legs
    pygame.draw.line(ship, (180, 195, 205), (28, 100), (10, 126), 5)
    pygame.draw.line(ship, (180, 195, 205), (72, 100), (90, 126), 5)
    pygame.draw.line(ship, (180, 195, 205), (10, 126), (2, 130), 4)
    pygame.draw.line(ship, (180, 195, 205), (90, 126), (98, 130), 4)

    # Rocket body
    pygame.draw.rect(
        ship, (220, 229, 237), (29, 32, 42, 69),
        border_radius=13
    )
    pygame.draw.ellipse(ship, (248, 250, 255), (29, 16, 42, 42))

    # Dark engine section
    pygame.draw.rect(ship, (48, 61, 78), (29, 88, 42, 16))

    # Cabin window
    pygame.draw.circle(ship, (10, 35, 66), (50, 54), 10)
    pygame.draw.circle(ship, (80, 190, 255), (50, 54), 6)

    # Metallic details
    pygame.draw.line(ship, (100, 120, 140), (35, 74), (65, 74), 2)

    # Main engine flame when action 2 is active
    if action == 2:
        pygame.draw.polygon(
            ship, (255, 150, 35),
            [(40, 102), (60, 102), (50, 137)]
        )
        pygame.draw.polygon(
            ship, (255, 245, 160),
            [(45, 102), (55, 102), (50, 126)]
        )

    # Side thruster flames
    if action == 1:
        pygame.draw.polygon(
            ship, (80, 190, 255),
            [(27, 65), (10, 72), (28, 76)]
        )
    elif action == 3:
        pygame.draw.polygon(
            ship, (80, 190, 255),
            [(73, 65), (90, 72), (72, 76)]
        )

    # Gym angle is in radians. Rotate the visual rocket.
    rotated = pygame.transform.rotate(ship, -math.degrees(angle))
    rect = rotated.get_rect(center=(int(x), int(y)))
    screen.blit(rotated, rect)


def draw_hud():
    draw_panel(20, 20, 340, 225)
    draw_text("MOON LANDING", 40, 35, (70, 220, 255), title_font)
    draw_text(
        f"Mode: {'MANUAL' if mode == 'manual' else 'AI AUTOPILOT'}",
        40, 88
    )
    draw_text(f"Score: {game_score}", 40, 120, (70, 255, 150))
    draw_text(f"Reward: {episode_reward:.1f}", 40, 150)
    draw_text(f"Altitude: {max(0, obs[1] * 100):.1f} m", 40, 180)
    draw_text(f"Vertical speed: {obs[3]:.2f}", 40, 210)

    # Fuel meter: game fuel is for manual mode.
    draw_panel(WIDTH - 300, 25, 270, 105)
    draw_text("FUEL", WIDTH - 275, 38, (70, 220, 255))
    draw_text(
        f"{fuel:.0f}%" if mode == "manual" else "AI SIMULATION",
        WIDTH - 150, 38
    )

    pygame.draw.rect(
        screen, (50, 70, 85),
        (WIDTH - 275, 75, 220, 18), border_radius=5
    )
    if mode == "manual":
        pygame.draw.rect(
            screen,
            (50, 240, 130) if fuel > 25 else (255, 75, 70),
            (WIDTH - 275, 75, int(220 * fuel / 100), 18),
            border_radius=5
        )
    else:
        pygame.draw.rect(
            screen, (50, 190, 255),
            (WIDTH - 275, 75, 220, 18), border_radius=5
        )

    draw_text(
        "UP: THRUST | LEFT/RIGHT: ROTATE",
        30, HEIGHT - 45, (255, 255, 255), small_font
    )


def draw_menu():
    screen.blit(background, (0, 0))

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 115))
    screen.blit(overlay, (0, 0))

    draw_panel(250, 115, 700, 540)

    title = title_font.render(
        "AI MOON LANDING CHALLENGE", True, (70, 220, 255)
    )
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 180)))

    draw_text(
        "Choose how you want to land on the Moon",
        330, 235, (255, 255, 255)
    )

    draw_panel(325, 300, 550, 105)
    draw_text("1  -  MANUAL PILOT", 365, 315, (80, 255, 160), title_font)
    draw_text(
        "Control the rocket yourself using the keyboard.",
        365, 365, (220, 235, 245), small_font
    )

    draw_panel(325, 430, 550, 105)
    draw_text("2  -  AI AUTOPILOT", 365, 445, (70, 210, 255), title_font)
    draw_text(
        "Watch your trained PPO model attempt the landing.",
        365, 495, (220, 235, 245), small_font
    )

    draw_text(
        "Land gently and upright to earn points.",
        365, 570, (255, 220, 100)
    )
    draw_text(
        "ESC: Quit",
        520, 615, (220, 235, 245), small_font
    )


def draw_end_screen():
    draw_panel(340, 270, 520, 230)

    draw_text(
        message,
        390, 300, message_color, title_font
    )
    draw_text(
        f"Final score: {game_score}",
        390, 365, (255, 255, 255), font
    )
    draw_text(
        f"Episode reward: {episode_reward:.1f}",
        390, 400, (220, 235, 245), font
    )
    draw_text(
        "R: Play again     M: Main menu",
        390, 450, (70, 220, 255), small_font
    )


# --------------------------------------------------
# MAIN GAME LOOP
# --------------------------------------------------

running = True

try:
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif mode is None:
                    if event.key == pygame.K_1:
                        start_game("manual")
                    elif event.key == pygame.K_2:
                        start_game("ai")

                elif done:
                    if event.key == pygame.K_r:
                        start_game(mode)
                    elif event.key == pygame.K_m:
                        return_to_menu()

        if mode is None:
            draw_menu()
            pygame.display.flip()
            continue

        if not done:
            if mode == "manual":
                keys = pygame.key.get_pressed()

                # Gymnasium LunarLander discrete actions:
                # 0 = no thrust, 1 = left engine,
                # 2 = main engine, 3 = right engine.
                action = 0

                if fuel > 0:
                    if keys[pygame.K_UP]:
                        action = 2
                    elif keys[pygame.K_LEFT]:
                        action = 1
                    elif keys[pygame.K_RIGHT]:
                        action = 3

                # A simple arcade fuel limit for manual play.
                if action != 0:
                    fuel = max(0, fuel - (0.12 if action == 2 else 0.06))

            else:
                predicted_action, _ = ai_model.predict(
                    obs, deterministic=True
                )
                action = int(predicted_action)

            last_action = action

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)

            if terminated or truncated:
                finish_game()

        # Map simulated coordinates to the screen.
        # The mapping is for display, not real-world scale.
        lander_x = WIDTH / 2 + float(obs[0]) * 310
        lander_y = HEIGHT * 0.52 - float(obs[1]) * 300

        lander_x = max(70, min(WIDTH - 70, lander_x))
        lander_y = max(100, min(HEIGHT - 90, lander_y))

        screen.blit(background, (0, 0))
        draw_lander(
            lander_x, lander_y, float(obs[4]), last_action
        )
        draw_hud()

        if done:
            draw_end_screen()

        pygame.display.flip()

finally:
    env.close()
    pygame.quit()
