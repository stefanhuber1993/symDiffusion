"""
Grid B: Gentle structural hints + proper low-pass noise.

Uses the winning config from Grid A (NO sym noise + every 3 + skip 10%)
and tests:
  - Low-pass noise at various spatial scales (Gaussian blur, no FFT artifacts)
  - Very gentle seed-image blends (0.03–0.08) for blob/cross/four-blobs

The seed images are now symmetrized before blending so they don't break
the symmetry that pg will enforce later.
"""

import os, sys, math, torch
import numpy as np
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
GROUP = "pg"
OUT_DIR = "experiments/gentle_hint_grid"

SNAP_EVERY = 5
SNAP_STEPS = list(range(SNAP_EVERY - 1, STEPS, SNAP_EVERY))

# Winning config from Grid A
BASE_KW = dict(
    symmetric_noise=False,
    symmetrize_every_n=3,
    symmetrize_start_fraction=0.10,
)


# ── Seed image generators ─────────────────────────────────────────────────────

def make_gaussian_blob(w, h, sigma_frac=0.25):
    cx, cy = w / 2, h / 2
    x = np.arange(w)[None, :] - cx
    y = np.arange(h)[:, None] - cy
    blob = np.exp(-(x**2 + y**2) / (2 * (w * sigma_frac)**2))
    arr = (blob * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr]*3, axis=-1))


def make_cross(w, h, bar_frac=0.10):
    bw, bh = int(w * bar_frac), int(h * bar_frac)
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[h//2 - bh//2: h//2 + bh//2, :] = 255
    arr[:, w//2 - bw//2: w//2 + bw//2] = 255
    return Image.fromarray(np.stack([arr]*3, axis=-1))


def make_four_blobs(w, h, sigma_frac=0.12):
    sx, sy = w * sigma_frac, h * sigma_frac
    x = np.arange(w)[None, :].astype(float)
    y = np.arange(h)[:, None].astype(float)
    blob = np.zeros((h, w))
    for bx, by in [(w*0.25, h*0.25), (w*0.75, h*0.25),
                   (w*0.25, h*0.75), (w*0.75, h*0.75)]:
        blob += np.exp(-((x-bx)**2/(2*sx**2) + (y-by)**2/(2*sy**2)))
    blob = np.clip(blob / blob.max(), 0, 1)
    arr = (blob * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr]*3, axis=-1))


SEED_FACTORIES = {
    "blob": lambda w, h: make_gaussian_blob(w, h),
    "cross": lambda w, h: make_cross(w, h),
    "four_blobs": lambda w, h: make_four_blobs(w, h),
}


# ── Experiment variants ───────────────────────────────────────────────────────

VARIANTS = [
    # label,                          extra pipeline kwargs
    ("NO sym noise\npure Gaussian\n(control)",
     dict()),

    ("NO sym noise\nlow-pass σ=3 latent px\n(~24px features)",
     dict(noise_lowpass_sigma=3.0)),

    ("NO sym noise\nlow-pass σ=6 latent px\n(~48px features)",
     dict(noise_lowpass_sigma=6.0)),

    ("NO sym noise\nlow-pass σ=10 latent px\n(~80px features)",
     dict(noise_lowpass_sigma=10.0)),

    ("NO sym noise\nblob blend=0.04",
     dict(noise_init_image="blob", noise_init_blend=0.04)),

    ("NO sym noise\nblob blend=0.08",
     dict(noise_init_image="blob", noise_init_blend=0.08)),

    ("NO sym noise\ncross blend=0.05",
     dict(noise_init_image="cross", noise_init_blend=0.05)),

    ("NO sym noise\nfour-blobs blend=0.05",
     dict(noise_init_image="four_blobs", noise_init_blend=0.05)),

    ("NO sym noise\nlow-pass σ=6 + blob 0.05",
     dict(noise_lowpass_sigma=6.0, noise_init_image="blob", noise_init_blend=0.05)),
]


def load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def make_grid(rows, thumb_w=160):
    label_col_w = 240
    seed_col_w = thumb_w
    step_hdr_h = 24
    cell_h = thumb_w
    n_step_cols = len(rows[0][2])
    n_rows = len(rows)
    total_w = label_col_w + seed_col_w + n_step_cols * thumb_w
    total_h = step_hdr_h + n_rows * cell_h

    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    fsm = load_font(12)
    flbl = load_font(11)

    draw.text((label_col_w + seed_col_w // 2, step_hdr_h // 2), "seed",
              fill=(120, 200, 120), font=fsm, anchor="mm")
    for ci, (sn, _) in enumerate(rows[0][2]):
        x = label_col_w + seed_col_w + ci * thumb_w + thumb_w // 2
        txt = "final" if sn == STEPS - 1 else f"step {sn+1}"
        draw.text((x, step_hdr_h // 2), txt, fill=(180, 180, 180), font=fsm, anchor="mm")

    for ri, (lbl, seed_img, step_imgs) in enumerate(rows):
        y0 = step_hdr_h + ri * cell_h
        for li, line in enumerate(lbl.split("\n")):
            draw.text((label_col_w // 2, y0 + 18 + li * 14),
                      line, fill=(255, 215, 80), font=flbl, anchor="mm")
        # seed preview
        if seed_img is not None:
            sp = seed_img.resize((seed_col_w, seed_col_w), Image.LANCZOS)
        else:
            sp = Image.new("RGB", (seed_col_w, seed_col_w), (50, 50, 50))
            d2 = ImageDraw.Draw(sp)
            d2.text((seed_col_w // 2, seed_col_w // 2), "none",
                    fill=(100, 100, 100), font=fsm, anchor="mm")
        canvas.paste(sp, (label_col_w, y0))
        for ci, (_, img) in enumerate(step_imgs):
            t = img.resize((thumb_w, thumb_w), Image.LANCZOS)
            canvas.paste(t, (label_col_w + seed_col_w + ci * thumb_w, y0))

    return canvas


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}")

    seed_images = {k: f(WIDTH, WIDTH) for k, f in SEED_FACTORIES.items()}
    for k, img in seed_images.items():
        img.save(os.path.join(OUT_DIR, f"seed_{k}.png"))
    print("Saved seed images")

    rows = []
    for label, extra_kw in VARIANTS:
        tag = label.replace("\n", " | ")
        print(f"\n── {tag} ──")

        kw = {**BASE_KW, **extra_kw}
        seed_key = kw.get("noise_init_image")
        seed_pil = seed_images.get(seed_key) if isinstance(seed_key, str) else None
        if isinstance(seed_key, str):
            kw["noise_init_image"] = seed_pil

        pipe = SymmetricDiffusionPipeline.from_pretrained(
            MODEL, wallpaper_group=GROUP, torch_dtype=dtype)
        pipe.to(device)

        result = pipe(
            prompt=PROMPT, negative_prompt=NEGATIVE,
            num_inference_steps=STEPS, guidance_scale=CFG,
            width=WIDTH, seed=SEED,
            save_intermediate_steps=SNAP_STEPS,
            **kw,
        )
        final_img, intermediates = result if isinstance(result, tuple) else (result, {})
        step_imgs = [(s, intermediates.get(s, final_img)) for s in SNAP_STEPS]

        slug = label.replace("\n", "_").replace(" ", "_").replace("/", "-")[:55]
        for s, img in step_imgs:
            img.save(os.path.join(OUT_DIR, f"{slug}_step{s+1:03d}.png"))
        final_img.save(os.path.join(OUT_DIR, f"{slug}_final.png"))
        pipe.make_tiled_preview(final_img).save(os.path.join(OUT_DIR, f"{slug}_tiled.png"))
        rows.append((label, seed_pil, step_imgs))
        print(f"  done")

        del pipe
        if device == "mps": torch.mps.empty_cache()
        elif device == "cuda": torch.cuda.empty_cache()

    grid = make_grid(rows)
    path = os.path.join(OUT_DIR, "gentle_hint_grid.png")
    grid.save(path)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    run()
