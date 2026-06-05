"""
Grid 1: Symmetrization timing experiments.

Tests two hypotheses for why symmetrization kills large-scale features:
  A. Symmetrizing initial noise locks the seed into a symmetric subspace too early
  B. Symmetrizing during the first few steps (when global structure forms) disrupts it

Rows = variants, Cols = step snapshots every 5 steps.
"""

import os, sys, torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from symmetric_diffusion import SymmetricDiffusionPipeline

PROMPT = "red and blue fish, tightly intertwined, 3d rendering, detailed"
NEGATIVE = "blurry, ugly, watermark"
SEED = 42
STEPS = 30
CFG = 7.5
WIDTH = 512
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUT_DIR = "experiments/timing_grid"

SNAP_EVERY = 5
SNAP_STEPS = list(range(SNAP_EVERY - 1, STEPS, SNAP_EVERY))

# Each entry: (label, group, pipeline kwargs)
VARIANTS = [
    ("p1  baseline",
     "p1", dict(symmetrize_every_n=1, symmetric_noise=True,
                symmetrize_start_fraction=0.0)),

    ("pg  sym every step\n(current default)",
     "pg", dict(symmetrize_every_n=1, symmetric_noise=True,
                symmetrize_start_fraction=0.0)),

    ("pg  sym every 3 steps",
     "pg", dict(symmetrize_every_n=3, symmetric_noise=True,
                symmetrize_start_fraction=0.0)),

    ("pg  skip first 10%\nthen sym every step",
     "pg", dict(symmetrize_every_n=1, symmetric_noise=True,
                symmetrize_start_fraction=0.10)),

    ("pg  skip first 10%\nthen sym every 3 steps",
     "pg", dict(symmetrize_every_n=3, symmetric_noise=True,
                symmetrize_start_fraction=0.10)),

    ("pg  NO symmetric noise\nsym every step",
     "pg", dict(symmetrize_every_n=1, symmetric_noise=False,
                symmetrize_start_fraction=0.0)),

    ("pg  NO symmetric noise\nskip first 10% + every 3",
     "pg", dict(symmetrize_every_n=3, symmetric_noise=False,
                symmetrize_start_fraction=0.10)),
]


def load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def make_grid(rows, thumb_w=180):
    label_col_w = 240
    step_hdr_h = 24
    row_label_h = 20
    cell_h = thumb_w
    n_cols = len(rows[0][1])
    n_rows = len(rows)
    total_w = label_col_w + n_cols * thumb_w
    total_h = step_hdr_h + n_rows * (cell_h + row_label_h)

    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    fsm = load_font(13)
    flbl = load_font(12)

    for ci, (sn, _) in enumerate(rows[0][1]):
        x = label_col_w + ci * thumb_w + thumb_w // 2
        txt = "final" if sn == STEPS - 1 else f"step {sn+1}"
        draw.text((x, step_hdr_h // 2), txt, fill=(180, 180, 180),
                  font=fsm, anchor="mm")

    for ri, (lbl, step_imgs) in enumerate(rows):
        y0 = step_hdr_h + ri * (cell_h + row_label_h)
        # row label on left
        for li, line in enumerate(lbl.split("\n")):
            draw.text((label_col_w // 2, y0 + 20 + li * 16),
                      line, fill=(255, 215, 80), font=flbl, anchor="mm")
        # thumbnails
        for ci, (_, img) in enumerate(step_imgs):
            thumb = img.resize((thumb_w, thumb_w), Image.LANCZOS)
            canvas.paste(thumb, (label_col_w + ci * thumb_w, y0))

    return canvas


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}")

    rows = []
    for label, group_id, kwargs in VARIANTS:
        print(f"\n── {label.replace(chr(10), ' | ')} ──")
        pipe = SymmetricDiffusionPipeline.from_pretrained(
            MODEL, wallpaper_group=group_id, torch_dtype=dtype)
        pipe.to(device)

        result = pipe(
            prompt=PROMPT, negative_prompt=NEGATIVE,
            num_inference_steps=STEPS, guidance_scale=CFG,
            width=WIDTH, seed=SEED,
            save_intermediate_steps=SNAP_STEPS,
            **kwargs,
        )
        final_img, intermediates = result if isinstance(result, tuple) else (result, {})

        step_imgs = [(s, intermediates.get(s, final_img)) for s in SNAP_STEPS]
        slug = label.replace("\n", "_").replace(" ", "_").replace("/", "-")[:40]
        for s, img in step_imgs:
            img.save(os.path.join(OUT_DIR, f"{slug}_step{s+1:03d}.png"))
        final_img.save(os.path.join(OUT_DIR, f"{slug}_final.png"))
        pipe.make_tiled_preview(final_img).save(
            os.path.join(OUT_DIR, f"{slug}_tiled.png"))
        rows.append((label, step_imgs))
        print(f"  done — {len(step_imgs)} snapshots")

        del pipe
        if device == "mps": torch.mps.empty_cache()
        elif device == "cuda": torch.cuda.empty_cache()

    grid = make_grid(rows)
    path = os.path.join(OUT_DIR, "timing_grid.png")
    grid.save(path)
    print(f"\nSaved grid: {path}")


if __name__ == "__main__":
    run()
