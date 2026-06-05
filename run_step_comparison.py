"""
Step-by-step comparison: p1 vs pg (every step) vs pg (every 3 steps) vs pg (pixel-space).

Saves a grid image showing the denoising progression for each variant.
Each row = one variant, each column = one step snapshot.

Usage:
    python3 run_step_comparison.py
"""

import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from symmetric_diffusion import SymmetricDiffusionPipeline

# ── Config ────────────────────────────────────────────────────────────────────
PROMPT = "red and blue fish, tightly intertwined, 3d rendering, detailed"
NEGATIVE = "blurry, ugly, watermark"
SEED = 42
STEPS = 30
CFG = 7.5
WIDTH = 512
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUT_DIR = "experiments/step_comparison"

SNAP_EVERY = 5  # save a snapshot every N denoising steps
SNAP_STEPS = list(range(SNAP_EVERY - 1, STEPS, SNAP_EVERY))  # [4,9,14,...,29]

VARIANTS = [
    dict(
        label="p1 (no symmetry)",
        group="p1",
        symmetrize_every_n=1,
        pixel_space_symmetrize=False,
    ),
    dict(
        label="pg – latent sym every step",
        group="pg",
        symmetrize_every_n=1,
        pixel_space_symmetrize=False,
    ),
    dict(
        label="pg – latent sym every 3 steps",
        group="pg",
        symmetrize_every_n=3,
        pixel_space_symmetrize=False,
    ),
    dict(
        label="pg – pixel-space sym every step",
        group="pg",
        symmetrize_every_n=1,
        pixel_space_symmetrize=True,
        pixel_symmetrize_every_n=1,
    ),
    dict(
        label="pg – pixel-space sym every 3 steps",
        group="pg",
        symmetrize_every_n=1,
        pixel_space_symmetrize=True,
        pixel_symmetrize_every_n=3,
    ),
]
# ──────────────────────────────────────────────────────────────────────────────


def load_font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def make_comparison_strip(
    rows: list[tuple[str, list[tuple[int, Image.Image]]]],
    thumb_w: int = 200,
    label_h: int = 28,
    step_label_h: int = 22,
) -> Image.Image:
    """Build a grid: rows = variants, cols = step snapshots + final."""
    n_cols = len(rows[0][1])
    n_rows = len(rows)
    label_col_w = 260

    cell_h = thumb_w  # square thumbnails
    total_w = label_col_w + n_cols * thumb_w
    total_h = step_label_h + n_rows * (cell_h + label_h)

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font_sm = load_font(14)
    font_lbl = load_font(13)

    # Column headers (step numbers)
    _, step_imgs = rows[0]
    for col_idx, (step_n, _) in enumerate(step_imgs):
        x = label_col_w + col_idx * thumb_w + thumb_w // 2
        label = "final" if step_n == STEPS - 1 else f"step {step_n + 1}"
        draw.text((x, step_label_h // 2), label, fill=(200, 200, 200),
                  font=font_sm, anchor="mm")

    for row_idx, (variant_label, step_imgs) in enumerate(rows):
        y_base = step_label_h + row_idx * (cell_h + label_h)

        # Row label
        draw.text(
            (label_col_w // 2, y_base + cell_h // 2),
            variant_label, fill=(255, 220, 100),
            font=font_lbl, anchor="mm",
        )

        for col_idx, (_, img) in enumerate(step_imgs):
            thumb = img.resize((thumb_w, thumb_w), Image.LANCZOS)
            x = label_col_w + col_idx * thumb_w
            canvas.paste(thumb, (x, y_base))

        # Variant label below the row
        draw.text(
            (label_col_w + (n_cols * thumb_w) // 2, y_base + cell_h + label_h // 2),
            variant_label, fill=(160, 160, 160),
            font=font_sm, anchor="mm",
        )

    return canvas


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}  |  Steps: {STEPS}  |  Snap at: {[s+1 for s in SNAP_STEPS]}")

    rows = []

    for v in VARIANTS:
        label = v.pop("label")
        group_id = v.pop("group")

        print(f"\n── {label} ──")
        pipe = SymmetricDiffusionPipeline.from_pretrained(
            MODEL, wallpaper_group=group_id, torch_dtype=dtype,
        )
        pipe.to(device)

        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            width=WIDTH,
            seed=SEED,
            save_intermediate_steps=SNAP_STEPS,
            **v,
        )

        if isinstance(result, tuple):
            final_img, intermediates = result
        else:
            final_img, intermediates = result, {}

        # Collect steps in order
        step_imgs = []
        for s in SNAP_STEPS:
            img = intermediates.get(s, final_img)
            step_imgs.append((s, img))
            # Save individual step images
            slug = label.lower().replace(" ", "_").replace("–", "-")
            img.save(os.path.join(OUT_DIR, f"{slug}_step{s+1:03d}.png"))

        # Save final tile
        final_img.save(os.path.join(OUT_DIR, f"{label.lower().replace(' ','_').replace('–','-')}_final.png"))
        tiled = pipe.make_tiled_preview(final_img, 3, 3)
        tiled.save(os.path.join(OUT_DIR, f"{label.lower().replace(' ','_').replace('–','-')}_tiled.png"))
        print(f"  Saved final + {len(step_imgs)} snapshots")

        rows.append((label, step_imgs))

        # Free memory between runs
        del pipe
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    # Build comparison grid
    print("\nBuilding comparison grid…")
    grid = make_comparison_strip(rows)
    grid_path = os.path.join(OUT_DIR, "comparison_grid.png")
    grid.save(grid_path)
    print(f"Saved grid: {grid_path}")


if __name__ == "__main__":
    run()
