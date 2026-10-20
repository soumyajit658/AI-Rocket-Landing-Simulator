
from PIL import Image, ImageDraw
import random
import math

random.seed(42)

WIDTH, HEIGHT = 1000, 700
image = Image.new("RGB", (WIDTH, HEIGHT), (8, 10, 24))
draw = ImageDraw.Draw(image)

# Starry space
for _ in range(220):
    x = random.randint(0, WIDTH - 1)
    y = random.randint(0, 430)
    brightness = random.randint(130, 255)
    draw.ellipse((x, y, x + 2, y + 2),
                 fill=(brightness, brightness, brightness))

# Distant lunar mountains
draw.polygon(
    [(0, 440), (120, 310), (230, 420), (370, 280),
     (510, 430), (660, 325), (810, 425), (920, 300),
     (1000, 400), (1000, 700), (0, 700)],
    fill=(65, 68, 82)
)

# Moon surface
draw.polygon(
    [(0, 500), (150, 470), (320, 510), (500, 485),
     (680, 510), (830, 475), (1000, 505),
     (1000, 700), (0, 700)],
    fill=(105, 106, 112)
)

# Surface craters
for _ in range(18):
    x = random.randint(20, WIDTH - 50)
    y = random.randint(520, 680)
    radius = random.randint(12, 38)
    draw.ellipse(
        (x - radius, y - radius // 3,
         x + radius, y + radius // 3),
        fill=(75, 77, 84),
        outline=(140, 141, 147),
        width=2
    )

# Landing pad
pad_x1, pad_x2 = 390, 610
pad_y = 510
draw.line((pad_x1, pad_y, pad_x2, pad_y),
          fill=(255, 205, 70), width=5)
draw.line((pad_x1 + 25, pad_y, pad_x1 + 25, pad_y + 12),
          fill=(255, 205, 70), width=3)
draw.line((pad_x2 - 25, pad_y, pad_x2 - 25, pad_y + 12),
          fill=(255, 205, 70), width=3)

# Label
draw.text((25, 25), "LUNAR MISSION", fill=(230, 235, 245))
draw.text((25, 50), "MOON LANDING ZONE", fill=(190, 195, 210))
draw.text((410, 525), "LANDING PAD", fill=(245, 245, 245))

image.save("moon_background.png")
print("Moon background created: moon_background.png")

