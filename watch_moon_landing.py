
import math
import random
from pathlib import Path

import gymnasium as gym
import pygame
from PIL import Image
from stable_baselines3 import PPO

# ============================================================
# CONFIGURATION
# ============================================================

WIDTH, HEIGHT = 1200, 800
FPS = 30

MODEL_PATH = "ppo_lunarlander_extended"
SCENE_PATH = "moon_scene.png"

# ============================================================
# INITIALIZATION
# ============================================================

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Moon Landing | PPO Mission Control")
clock = pygame.time.Clock()

font = pygame.font.SysFont("consolas", 19)
small_font = pygame.font.SysFont("consolas", 15)
title_font = pygame.font.SysFont("consolas", 23, bold=True)

rng = random.Random()

PROJECT_DIR = Path(__file__).resolve().parent
scene_path = PROJECT_DIR / SCENE_PATH
model_path = PROJECT_DIR / MODEL_PATH

# ============================================================
# LOAD BACKGROUND IMAGE
# ============================================================

try:
    scene = Image.open(scene_path).convert("RGB")
except FileNotFoundError:
    pygame.quit()
    raise SystemExit(
        f"Cannot find {SCENE_PATH}. "
        "Place the image in the same folder as this script."
    )

# Preserve aspect ratio while fitting the image into the window
scene.thumbnail((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

# Convert Pillow image into a Pygame Surface
background_image = pygame.image.frombytes(
    scene.tobytes(), scene.size, "RGB"
)

background_x = (WIDTH - background_image.get_width()) // 2
background_y = (HEIGHT - background_image.get_height()) // 2

background = pygame.Surface((WIDTH, HEIGHT))
background.fill((3, 5, 15))
background.blit(background_image, (background_x, background_y))

# ============================================================
# LOAD TRAINED PPO AGENT AND ENVIRONMENT
# ============================================================

try:
    model = PPO.load(str(model_path))
except Exception as error:
    pygame.quit()
    raise SystemExit(f"Could not load PPO model: {error}")

env = gym.make("LunarLander-v3")

episode = 1
obs, info = env.reset(seed=episode)

total_reward = 0.0
done = False
last_action = 0
running = True

dust_particles = []


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def draw_text(text, x, y, color=(220, 235, 255), font_obj=small_font):
    surface = font_obj.render(str(text), True, color)
    screen.blit(surface, (x, y))


def draw_panel(rect, alpha=220):
    panel_surface = pygame.Surface(
        (rect[2], rect[3]), pygame.SRCALPHA
    )
    panel_surface.fill((4, 13, 26, alpha))
    screen.blit(panel_surface, (rect[0], rect[1]))

    pygame.draw.rect(
        screen, (50, 155, 205), rect, 1, border_radius=9
    )


def draw_lander(x, y, angle, engine_on, side_action):
    """Draw a stylized lunar lander with engine effects."""

    body = pygame.Surface((120, 140), pygame.SRCALPHA)

    # Main engine flame
    if engine_on:
        flame_length = rng.randint(35, 65)

        pygame.draw.polygon(
            body,
            (255, 105, 20, 220),
            [(43, 88), (77, 88), (60, 88 + flame_length)],
        )

        pygame.draw.polygon(
            body,
            (255, 220, 120, 235),
            [
                (51, 88),
                (69, 88),
                (60, 88 + flame_length // 2),
            ],
        )

    # Side thruster flames
    if side_action in (1, 3):
        flame_x = 17 if side_action == 1 else 103
        direction = -1 if side_action == 1 else 1

        pygame.draw.polygon(
            body,
            (255, 145, 40, 210),
            [
                (flame_x, 48),
                (flame_x, 65),
                (flame_x + direction * 25, 56),
            ],
        )

    # Landing legs
    pygame.draw.line(body, (185, 195, 205), (43, 76), (23, 108), 5)
    pygame.draw.line(body, (185, 195, 205), (77, 76), (97, 108), 5)

    # Landing feet
    pygame.draw.line(body, (235, 195, 100), (16, 109), (32, 109), 4)
    pygame.draw.line(body, (235, 195, 100), (88, 109), (104, 109), 4)

    # Lower stage
    pygame.draw.polygon(
        body,
        (190, 137, 53),
        [(35, 54), (85, 54), (78, 86), (42, 86)],
    )

    pygame.draw.rect(body, (230, 195, 120), (44, 60, 32, 19))
    pygame.draw.line(body, (120, 85, 35), (60, 55), (60, 85), 2)

    # Upper cabin
    pygame.draw.polygon(
        body,
        (210, 220, 230),
        [(43, 22), (77, 22), (86, 52), (34, 52)],
    )

    # Cabin window
    pygame.draw.polygon(
        body,
        (45, 125, 165),
        [(47, 28), (73, 28), (78, 44), (42, 44)],
    )

    pygame.draw.line(body, (160, 225, 245), (46, 29), (42, 43), 2)

    # Antenna
    pygame.draw.line(body, (220, 225, 230), (60, 22), (60, 10), 2)
    pygame.draw.circle(body, (245, 245, 250), (60, 9), 4)

    # Rotate and draw the spacecraft
    rotated = pygame.transform.rotate(body, -math.degrees(angle))
    rect = rotated.get_rect(center=(int(x), int(y)))
    screen.blit(rotated, rect)


def reset_episode():
    global obs, info, total_reward, done, episode, last_action

    episode += 1
    obs, info = env.reset(seed=episode)

    total_reward = 0.0
    done = False
    last_action = 0
    dust_particles.clear()


def update_dust(dt, x, y, create_dust):
    if create_dust:
        for _ in range(4):
            dust_particles.append(
                [
                    x + rng.randint(-22, 22),
                    y + 48,
                    rng.uniform(-65, 65),
                    rng.uniform(10, 55),
                    rng.randint(15, 32),
                ]
            )

    for particle in dust_particles[:]:
        particle[0] += particle[2] * dt
        particle[1] += particle[3] * dt
        particle[4] -= 1

        radius = max(1, int(particle[4] // 5))

        pygame.draw.circle(
            screen,
            (155, 151, 145),
            (int(particle[0]), int(particle[1])),
            radius,
        )

        if particle[4] <= 0:
            dust_particles.remove(particle)


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

try:
    while running:
        dt = clock.tick(FPS) / 1000.0

        # Keyboard events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r and done:
                    reset_episode()

        # AI controls the lander
        if not done:
            action, _ = model.predict(obs, deterministic=True)
            last_action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            done = terminated or truncated

        # Approximate mapping from environment observations to pixels
        x = WIDTH * 0.5 + float(obs[0]) * 310
        y = 390 + (0.75 - float(obs[1])) * 360

        x = max(100, min(WIDTH - 100, x))
        y = max(175, min(HEIGHT - 125, y))

        angle = float(obs[4])
        vertical_speed = float(obs[3])
        horizontal_speed = float(obs[2])

        # Background
        screen.blit(background, (0, 0))

        # Dust and engine effects
        engine_on = last_action == 2 and not done

        update_dust(
            dt,
            x,
            y,
            create_dust=engine_on and y > 420,
        )

        draw_lander(
            x,
            y,
            angle,
            engine_on,
            last_action,
        )

        # Left telemetry panel
        draw_panel((18, 18, 350, 230))

        draw_text(
            "AI MOON LANDING",
            34,
            30,
            (85, 210, 255),
            title_font,
        )

        draw_text(
            "PPO AGENT | REWARD SHAPING",
            34,
            61,
            (80, 205, 240),
        )

        pygame.draw.line(
            screen, (40, 100, 135), (34, 86), (350, 86), 1
        )

        draw_text(f"Episode        {episode}", 34, 100)
        draw_text(
            f"Total reward  {total_reward:8.2f}",
            34,
            128,
            (120, 255, 160),
        )
        draw_text(f"Altitude       {obs[1]:8.3f}", 34, 156)
        draw_text(f"Vertical speed {vertical_speed:8.3f}", 34, 184)
        draw_text(f"Horizontal v   {horizontal_speed:8.3f}", 34, 212)

        # Right flight systems panel
        draw_panel((WIDTH - 290, 18, 272, 190))

        draw_text(
            "FLIGHT SYSTEMS",
            WIDTH - 270,
            32,
            (85, 210, 255),
            title_font,
        )

        draw_text(
            f"Angle       {math.degrees(angle):7.2f} deg",
            WIDTH - 270,
            73,
        )

        draw_text(
            f"Main engine  {'ON' if engine_on else 'OFF'}",
            WIDTH - 270,
            105,
            (255, 185, 100) if engine_on else (175, 195, 215),
        )

        draw_text(f"Action       {last_action}", WIDTH - 270, 137)

        draw_text(
            "PPO AUTOPILOT ACTIVE",
            WIDTH - 270,
            171,
            (110, 255, 160),
        )

        # Bottom status panel
        draw_panel((WIDTH // 2 - 255, HEIGHT - 66, 510, 46))

        if done:
            status = "EPISODE ENDED | PRESS R TO RETRY"
            status_color = (255, 220, 120)
        else:
            status = "AI AUTOPILOT ACTIVE | ESC TO EXIT"
            status_color = (130, 220, 255)

        status_surface = font.render(status, True, status_color)
        status_rect = status_surface.get_rect(
            center=(WIDTH // 2, HEIGHT - 43)
        )
        screen.blit(status_surface, status_rect)

        pygame.display.flip()

finally:
    env.close()
    pygame.quit()
