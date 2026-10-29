"""Generate packaging/app.ico — a simple waveform mark used by the exe and setup.

Run once from the repository root (Pillow is a dependency of CustomTkinter):

    python packaging/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
BACKGROUND = (22, 24, 29)
ACCENT = (79, 140, 255)
BARS = (24, 52, 92, 132, 160, 132, 96, 56, 28)


def render() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, SIZE - 8, SIZE - 8), radius=56, fill=BACKGROUND)

    bar_width = 16
    gap = 8
    total = len(BARS) * bar_width + (len(BARS) - 1) * gap
    left = (SIZE - total) // 2
    for index, height in enumerate(BARS):
        x = left + index * (bar_width + gap)
        top = SIZE // 2 - height
        bottom = SIZE // 2 + height
        draw.rounded_rectangle((x, top, x + bar_width, bottom), radius=bar_width // 2, fill=ACCENT)
    return image


def main() -> None:
    target = Path(__file__).with_name("app.ico")
    render().save(target, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
