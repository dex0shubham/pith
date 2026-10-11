"""Render `pith demo`'s transcript into docs/demo.gif. Needs Pillow, which is not a pith dependency.

Run: .venv/bin/python scripts/render_demo_gif.py
The transcript is deterministic, but the GIF bytes depend on the font and the Pillow version: regenerating on another
OS or Pillow release produces a different binary for the same frames.
"""
import io
import math
import sys
import textwrap
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
    try:
        return ImageFont.load_default(SIZE)  # Pillow >= 10.1 with FreeType: a scalable built-in font
    except (TypeError, ImportError):
        sys.exit("no monospace TTF found (Menlo, DejaVuSansMono) and this Pillow has no sized default font; "
                 "install fonts-dejavu or Pillow >= 10.1")


def frame(lines, fnt, line_h):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    cols = int((W - 2 * PAD) // fnt.getlength("M"))
    rows = [(part, PROMPT if line.startswith("$") else ACCENT if line.startswith("pinned ") else FG)
            for line in lines for part in textwrap.wrap(line, cols, subsequent_indent="    ", drop_whitespace=False)]
    for i, (text, color) in enumerate(rows[-((H - 2 * PAD) // line_h):]):  # scroll like a terminal once full
        d.text((PAD, PAD + i * line_h), text, font=fnt, fill=color)
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
