from pathlib import Path
import math

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image
from stable_baselines3 import PPO


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
WIDTH = 1200
HEIGHT = 800
FPS = 50

BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_PATH = BASE_DIR / "moon_scene.png"
MODEL_PATH = BASE_DIR / "ppo_lunarlander_extended.zip"

# LunarLander-v3 is a discrete-action environment, matching the trained PPO model.
env = gym.make("LunarLander-v3")

# --------------------------------------------------
# PYGAME SETUP
# --------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Watch Moon Landing — PPO AI")
clock = pygame.time.Clock()

FONT = pygame.font.SysFont("arial", 24)
SMALL_FONT = pygame.font.SysFont("arial", 18)
TITLE_FONT = pygame.font.SysFont("arial", 42)

WHITE = (245, 245, 245)
CYAN = (90, 220, 255)
GREEN = (95, 240, 150)
ORANGE = (255, 170, 65)
RED = (255, 95, 95)


# --------------------------------------------------
# LOAD BACKGROUND AND MODEL
# --------------------------------------------------
if not BACKGROUND_PATH.exists():
    env.close()
    pygame.quit()
    raise FileNotFoundError(
        f"Missing {BACKGROUND_PATH.name}. Place it beside watch_moon_landing.py."
    )

if not MODEL_PATH.exists():
    env.close()
    pygame.quit()
    raise FileNotFoundError(
        f"Missing {MODEL_PATH.name}. Place it beside watch_moon_landing.py."
    )

background_image = Image.open(BACKGROUND_PATH).convert("RGB")
background_image = background_image.resize(
    (WIDTH, HEIGHT), Image.Resampling.LANCZOS
)
background = pygame.image.fromstring(
    background_image.tobytes(), background_image.size, background_image.mode
).convert()

model = PPO.load(str(MODEL_PATH))


# --------------------------------------------------
# DRAWING HELPERS
# --------------------------------------------------
def draw_text(text, font, color, x, y, center=False):
    surface = font.render(text, True, color)
    rect = surface.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    screen.blit(surface, rect)


def draw_panel(rect, alpha=185, border_color=None):
    panel = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(
        panel, (5, 10, 24, alpha), panel.get_rect(), border_radius=15
    )
    if border_color:
        pygame.draw.rect(
            panel,
            (*border_color, 235),
            panel.get_rect(),
            width=2,
            border_radius=15,
        )
    screen.blit(panel, (rect[0], rect[1]))


def observation_to_screen(obs):
    # LunarLander observation values:
    # x, y, horizontal velocity, vertical velocity, angle,
    # angular velocity, left-leg contact, right-leg contact.
    lander_x = float(obs[0])
    lander_y = float(obs[1])

    x = int(WIDTH * 0.5 + lander_x * 180)
    y = int(HEIGHT * 0.72 - lander_y * 180)

    x = max(45, min(WIDTH - 45, x))
    y = max(100, min(HEIGHT - 150, y))
    return x, y


def draw_lander(x, y, angle, action):
    action = int(action)

    # Action meanings for LunarLander-v3:
    # 0 = do nothing, 1 = fire left engine,
    # 2 = fire main engine, 3 = fire right engine.
    main_engine = action == 2
    side_engine = action in (1, 3)

    # Main engine flame
    if main_engine:
        flame_length = np.random.randint(24, 38)
        pygame.draw.polygon(
            screen,
            ORANGE,
            [
                (x - 9, y + 19),
                (x + 9, y + 19),
                (x, y + flame_length + 20),
            ],
        )
        pygame.draw.polygon(
            screen,
            (255, 235, 140),
            [
                (x - 4, y + 20),
                (x + 4, y + 20),
                (x, y + flame_length),
            ],
        )

    # Side-engine bursts
    if side_engine:
        if action == 1:
            points = [(x + 14, y + 5), (x + 30, y + 10), (x + 15, y + 14)]
        else:
            points = [(x - 14, y + 5), (x - 30, y + 10), (x - 15, y + 14)]
        pygame.draw.polygon(screen, ORANGE, points)

    # Rocket body, rotated to reflect the lander's angle
    body_points = [
        (0, -28),
        (18, 15),
        (11, 23),
        (-11, 23),
        (-18, 15),
    ]
    rotated_points = []
    radians = -float(angle)

    for px, py in body_points:
        rx = px * math.cos(radians) - py * math.sin(radians)
        ry = px * math.sin(radians) + py * math.cos(radians)
        rotated_points.append((int(x + rx), int(y + ry)))

    pygame.draw.polygon(screen, (225, 235, 245), rotated_points)
    pygame.draw.polygon(screen, CYAN, rotated_points, 2)

    # Cockpit
    pygame.draw.circle(screen, (65, 190, 255), (x, y - 5), 6)
    pygame.draw.circle(screen, (225, 250, 255), (x - 2, y - 7), 2)

    # Landing legs
    pygame.draw.line(screen, WHITE, (x - 10, y + 17), (x - 22, y + 29), 3)
    pygame.draw.line(screen, WHITE, (x + 10, y + 17), (x + 22, y + 29), 3)
    pygame.draw.line(screen, WHITE, (x - 27, y + 30), (x - 17, y + 30), 3)
    pygame.draw.line(screen, WHITE, (x + 17, y + 30), (x + 27, y + 30), 3)


def draw_landing_pad():
    # Decorative landing pad; the real physics are simulated by Gymnasium.
    pad_y = HEIGHT - 105
    pygame.draw.line(
        screen, (160, 180, 200),
        (WIDTH // 2 - 110, pad_y),
        (WIDTH // 2 + 110, pad_y), 5
    )
    pygame.draw.line(
        screen, CYAN,
        (WIDTH // 2 - 75, pad_y + 7),
        (WIDTH // 2 + 75, pad_y + 7), 2
    )
    draw_text(
        "LANDING ZONE", SMALL_FONT, WHITE,
        WIDTH // 2, pad_y + 18, center=True
    )


def draw_hud(episode_reward, obs, action, episode_number, paused):
    draw_panel((20, 20, 365, 190), alpha=200, border_color=GREEN)
    draw_text("PPO AI FLIGHT", FONT, GREEN, 40, 35)
    draw_text(f"Episode: {episode_number}", SMALL_FONT, WHITE, 40, 72)
    draw_text(f"Reward: {episode_reward:,.1f}", SMALL_FONT, WHITE, 40, 100)
    draw_text(f"Action: {int(action)}", SMALL_FONT, WHITE, 40, 128)
    draw_text(
        f"Vertical speed: {float(obs[3]):.2f}",
        SMALL_FONT, WHITE, 40, 156
    )

    if paused:
        draw_panel((WIDTH // 2 - 130, 30, 260, 52), alpha=220,
                   border_color=ORANGE)
        draw_text("PAUSED", FONT, ORANGE, WIDTH // 2, 56, center=True)

    draw_panel((20, HEIGHT - 78, 480, 55), alpha=180)
    draw_text(
        "SPACE: pause   R: restart episode   ESC: quit",
        SMALL_FONT, WHITE, 36, HEIGHT - 60
    )


def draw_result_overlay(message, color, episode_reward):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((2, 6, 18, 155))
    screen.blit(overlay, (0, 0))

    draw_panel(
        (WIDTH // 2 - 300, HEIGHT // 2 - 110, 600, 220),
        alpha=235, border_color=color
    )
    draw_text(message, TITLE_FONT, color, WIDTH // 2, HEIGHT // 2 - 42,
              center=True)
    draw_text(
        f"Episode reward: {episode_reward:,.2f}",
        FONT, WHITE, WIDTH // 2, HEIGHT // 2 + 18, center=True
    )
    draw_text(
        "Press R to watch another landing",
        SMALL_FONT, CYAN, WIDTH // 2, HEIGHT // 2 + 62, center=True
    )
    draw_text(
        "Press ESC to quit",
        SMALL_FONT, WHITE, WIDTH // 2, HEIGHT // 2 + 88, center=True
    )


# --------------------------------------------------
# EPISODE HELPERS
# --------------------------------------------------
def reset_episode(episode_number):
    observation, _ = env.reset()
    return observation, 0.0, False, 0, episode_number


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
obs, episode_reward, done, action, episode_number = reset_episode(1)
paused = False
result_message = ""
result_color = WHITE
running = True

try:
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE and not done:
                    paused = not paused

                elif event.key == pygame.K_r:
                    episode_number += 1
                    obs, episode_reward, done, action, episode_number = reset_episode(
                        episode_number
                    )
                    paused = False
                    result_message = ""

        if not paused and not done:
            predicted_action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(predicted_action).item())

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)

            if terminated or truncated:
                done = True
                left_contact = bool(obs[6] > 0.5)
                right_contact = bool(obs[7] > 0.5)
                vertical_speed = abs(float(obs[3]))
                horizontal_speed = abs(float(obs[2]))
                angle = abs(float(obs[4]))

                safe_landing = (
                    left_contact
                    and right_contact
                    and vertical_speed < 0.45
                    and horizontal_speed < 0.5
                    and angle < 0.25
                )

                if safe_landing:
                    result_message = "SUCCESSFUL LANDING!"
                    result_color = GREEN
                elif left_contact or right_contact:
                    result_message = "ROUGH LANDING"
                    result_color = ORANGE
                else:
                    result_message = "MISSION ENDED"
                    result_color = RED

        screen.blit(background, (0, 0))
        draw_landing_pad()

        x, y = observation_to_screen(obs)
        draw_lander(x, y, float(obs[4]), action)
        draw_hud(episode_reward, obs, action, episode_number, paused)

        if done:
            draw_result_overlay(result_message, result_color, episode_reward)

        pygame.display.flip()

finally:
    env.close()
    pygame.quit()
