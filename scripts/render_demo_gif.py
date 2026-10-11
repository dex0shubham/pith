"""Render `pith demo`'s transcript into docs/demo.gif. Needs Pillow, which is not a pith dependency.

Run: .venv/bin/python scripts/render_demo_gif.py
"""
import io
import math
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow is needed to render the GIF: pip install pillow")

from pith.demo import run_demo

W, H, PAD, SIZE = 960, 540, 18, 14
BG, FG, PROMPT, ACCENT = (24, 26, 33), (220, 223, 228), (120, 200, 140), (240, 200, 110)
MAX_FRAMES, STEP_MS, HOLD_MS = 30, 350, 3000
OUT = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"


def font():
    for name in ("/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, SIZE)
        except OSError:
            pass
    return ImageFont.load_default()


def frame(lines, fnt, line_h):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    visible = lines[-((H - 2 * PAD) // line_h):]  # scroll like a terminal once the screen fills
    for i, line in enumerate(visible):
        color = PROMPT if line.startswith("$") else ACCENT if line.startswith("pinned ") else FG
        d.text((PAD, PAD + i * line_h), line, font=fnt, fill=color)
    return img


def main() -> int:
    buf = io.StringIO()
    rc = run_demo(out=buf)
    lines = ["$ pith demo"] + buf.getvalue().rstrip("\n").splitlines()
    fnt = font()
    line_h = SIZE + 6
    per = max(1, math.ceil(len(lines) / (MAX_FRAMES - 1)))
    frames = [frame(lines[:n], fnt, line_h) for n in range(1, len(lines), per)] + [frame(lines, fnt, line_h)]
    durations = [STEP_MS] * (len(frames) - 1) + [HOLD_MS]
    frames = [f.quantize(colors=16) for f in frames]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{OUT}: {len(frames)} frames, {OUT.stat().st_size / 1024:.0f} KB")
    return rc


if __name__ == "__main__":
    sys.exit(main())
