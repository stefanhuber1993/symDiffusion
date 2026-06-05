"""
Grid A: NO-symmetric-noise exploration.

The previous timing grid showed that NO-sym-noise + skip10% + every3 was dramatically
better than all sym-noise variants. This grid isolates which of those levers matter
and finds the best combination.

Axes: symmetric_noise (True/False) × skip_start (0%, 10%, 20%) × every_n (1, 3, 5)
"""

import os, sys, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from symmetric_diffusion import SymmetricDiffusionPipeline

PROMPT = "red and blue fish, tightly intertwined, 3d rendering, detailed"
NEGATIVE = "blurry, ugly, watermark"
SEED = 42
STEPS = 30
CFG = 7.5
WIDTH = 512
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUT_DIR = "experiments/nosym_noise_grid"

SNAP_EVERY = 5
SNAP_STEPS = list(range(SNAP_EVERY - 1, STEPS, SNAP_EVERY))

VARIANTS = [
    # label,               group, sym_noise, skip,  every_n
    ("p1  baseline",       "p1",  True,      0.0,   1),
    ("pg  SYM noise\nevery 1 (default)", "pg", True,  0.0,  1),
    ("pg  NO sym noise\nevery 1  skip 0%",  "pg", False, 0.0,  1),
    ("pg  NO sym noise\nevery 1  skip 10%", "pg", False, 0.10, 1),
    ("pg  NO sym noise\nevery 1  skip 20%", "pg", False, 0.20, 1),
    ("pg  NO sym noise\nevery 3  skip 0%",  "pg", False, 0.0,  3),
    ("pg  NO sym noise\nevery 3  skip 10%", "pg", False, 0.10, 3),
    ("pg  NO sym noise\nevery 5  skip 0%",  "pg", False, 0.0,  5),
    ("pg  NO sym noise\nevery 5  skip 20%", "pg", False, 0.20, 5),
]


def load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def make_grid(rows, thumb_w=170):
    label_col_w = 240
    step_hdr_h = 24
    cell_h = thumb_w
    n_cols = len(rows[0][1])
    n_rows = len(rows)
    total_w = label_col_w + n_cols * thumb_w
    total_h = step_hdr_h + n_rows * cell_h

    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    fsm = load_font(13)
    flbl = load_font(12)

    for ci, (sn, _) in enumerate(rows[0][1]):
        x = label_col_w + ci * thumb_w + thumb_w // 2
        txt = "final" if sn == STEPS - 1 else f"step {sn+1}"
        draw.text((x, step_hdr_h // 2), txt, fill=(180, 180, 180), font=fsm, anchor="mm")

    for ri, (lbl, step_imgs) in enumerate(rows):
        y0 = step_hdr_h + ri * cell_h
        for li, line in enumerate(lbl.split("\n")):
            draw.text((label_col_w // 2, y0 + 22 + li * 16),
                      line, fill=(255, 215, 80), font=flbl, anchor="mm")
        for ci, (_, img) in enumerate(step_imgs):
            canvas.paste(img.resize((thumb_w, thumb_w), Image.LANCZOS),
                         (label_col_w + ci * thumb_w, y0))

    return canvas


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}")

    rows = []
    for label, group_id, sym_noise, skip, every_n in VARIANTS:
        tag = label.replace("\n", " | ")
        print(f"\n── {tag} ──")
        pipe = SymmetricDiffusionPipeline.from_pretrained(
            MODEL, wallpaper_group=group_id, torch_dtype=dtype)
        pipe.to(device)

        result = pipe(
            prompt=PROMPT, negative_prompt=NEGATIVE,
            num_inference_steps=STEPS, guidance_scale=CFG,
            width=WIDTH, seed=SEED,
            symmetric_noise=sym_noise,
            symmetrize_start_fraction=skip,
            symmetrize_every_n=every_n,
            save_intermediate_steps=SNAP_STEPS,
        )
        final_img, intermediates = result if isinstance(result, tuple) else (result, {})
        step_imgs = [(s, intermediates.get(s, final_img)) for s in SNAP_STEPS]

        slug = label.replace("\n", "_").replace(" ", "_").replace("%", "pct")[:50]
        for s, img in step_imgs:
            img.save(os.path.join(OUT_DIR, f"{slug}_step{s+1:03d}.png"))
        final_img.save(os.path.join(OUT_DIR, f"{slug}_final.png"))
        pipe.make_tiled_preview(final_img).save(os.path.join(OUT_DIR, f"{slug}_tiled.png"))
        rows.append((label, step_imgs))

        del pipe
        if device == "mps": torch.mps.empty_cache()
        elif device == "cuda": torch.cuda.empty_cache()

    grid = make_grid(rows)
    path = os.path.join(OUT_DIR, "nosym_noise_grid.png")
    grid.save(path)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    run()
