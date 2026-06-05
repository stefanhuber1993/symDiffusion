"""
Grid 2: Initial noise structure experiments.

Tests whether biasing the initial latent toward large-scale structure helps
the model form coherent features before symmetrization takes over.

Noise init types:
  - Pure Gaussian (baseline)
  - Low-frequency bandpass (biases toward large features)
  - Very-low-frequency bandpass
  - Gaussian blob: soft bright disk in center, rest dark
  - Hard circle: binary ring pattern
  - Radial gradient: bright center fading outward
  - Cross pattern: two bright stripes

The blob/circle/gradient images are encoded with the VAE and blended into
the Gaussian noise at blend weight 0.3 (adjustable). This acts as a "soft
structural hint" — the diffusion process still has full freedom but is
nudged toward placing large features in specific locations.
"""

import os, sys, math
import torch
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
OUT_DIR = "experiments/noise_init_grid"
GROUP = "pg"

SNAP_EVERY = 5
SNAP_STEPS = list(range(SNAP_EVERY - 1, STEPS, SNAP_EVERY))

BLEND = 0.35  # how strongly the seed image influences initial noise


# ── Seed image generators ─────────────────────────────────────────────────────

def make_gaussian_blob(w, h, sigma_frac=0.3):
    """Bright Gaussian blob in center on black background."""
    cx, cy = w / 2, h / 2
    sigma_x, sigma_y = w * sigma_frac, h * sigma_frac
    x = np.arange(w)[None, :] - cx
    y = np.arange(h)[:, None] - cy
    blob = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
    blob = (blob * 255).astype(np.uint8)
    rgb = np.stack([blob] * 3, axis=-1)
    return Image.fromarray(rgb)


def make_hard_circle(w, h, radius_frac=0.35, ring_width_frac=0.08):
    """Bright ring at given radius on black background."""
    cx, cy = w / 2, h / 2
    r = min(w, h) * radius_frac
    ring_w = min(w, h) * ring_width_frac
    x = np.arange(w)[None, :] - cx
    y = np.arange(h)[:, None] - cy
    dist = np.sqrt(x**2 + y**2)
    ring = np.where(np.abs(dist - r) < ring_w, 255, 0).astype(np.uint8)
    rgb = np.stack([ring] * 3, axis=-1)
    return Image.fromarray(rgb)


def make_radial_gradient(w, h):
    """Bright center, dark edges (smooth radial falloff)."""
    cx, cy = w / 2, h / 2
    x = np.arange(w)[None, :] - cx
    y = np.arange(h)[:, None] - cy
    dist = np.sqrt(x**2 + y**2)
    max_d = math.sqrt(cx**2 + cy**2)
    grad = np.clip(1.0 - dist / max_d, 0, 1)
    arr = (grad * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def make_cross(w, h, bar_frac=0.12):
    """Two bright orthogonal bars on black background (cross/plus shape)."""
    bw = int(w * bar_frac)
    bh = int(h * bar_frac)
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[h // 2 - bh // 2: h // 2 + bh // 2, :] = 255
    arr[:, w // 2 - bw // 2: w // 2 + bw // 2] = 255
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def make_four_blobs(w, h, sigma_frac=0.15):
    """Four Gaussian blobs near the corners (good for pg glide symmetry)."""
    cx, cy = w / 2, h / 2
    sx, sy = w * sigma_frac, h * sigma_frac
    x = np.arange(w)[None, :].astype(float)
    y = np.arange(h)[:, None].astype(float)
    centers = [(w * 0.25, h * 0.25), (w * 0.75, h * 0.25),
               (w * 0.25, h * 0.75), (w * 0.75, h * 0.75)]
    blob = np.zeros((h, w))
    for bx, by in centers:
        blob += np.exp(-((x - bx)**2 / (2 * sx**2) + (y - by)**2 / (2 * sy**2)))
    blob = np.clip(blob / blob.max(), 0, 1)
    arr = (blob * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


# ──────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    ("pure Gaussian\n(baseline)",
     dict(noise_init_image=None, noise_target_freq=0.0)),

    ("low-freq noise\nfreq=0.04",
     dict(noise_init_image=None, noise_target_freq=0.04)),

    ("very-low-freq noise\nfreq=0.02",
     dict(noise_init_image=None, noise_target_freq=0.02)),

    ("Gaussian blob\ncenter",
     dict(noise_init_image="blob", noise_init_blend=BLEND, noise_target_freq=0.0)),

    ("Hard circle\nring",
     dict(noise_init_image="circle", noise_init_blend=BLEND, noise_target_freq=0.0)),

    ("Radial gradient\nbright center",
     dict(noise_init_image="radial", noise_init_blend=BLEND, noise_target_freq=0.0)),

    ("Four blobs\ncorners",
     dict(noise_init_image="four_blobs", noise_init_blend=BLEND, noise_target_freq=0.0)),
]

SEED_IMAGE_FACTORIES = {
    "blob": lambda w, h: make_gaussian_blob(w, h),
    "circle": lambda w, h: make_hard_circle(w, h),
    "radial": lambda w, h: make_radial_gradient(w, h),
    "four_blobs": lambda w, h: make_four_blobs(w, h),
}


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
    n_cols = len(rows[0][2]) + 1  # +1 for seed image preview
    n_rows = len(rows)
    total_w = label_col_w + n_cols * thumb_w
    total_h = step_hdr_h + n_rows * (cell_h + row_label_h)

    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    fsm = load_font(13)
    flbl = load_font(12)

    # Column headers
    draw.text((label_col_w + thumb_w // 2, step_hdr_h // 2), "seed",
              fill=(120, 200, 120), font=fsm, anchor="mm")
    for ci, (sn, _) in enumerate(rows[0][1]):
        x = label_col_w + (ci + 1) * thumb_w + thumb_w // 2
        txt = "final" if sn == STEPS - 1 else f"step {sn+1}"
        draw.text((x, step_hdr_h // 2), txt, fill=(180, 180, 180),
                  font=fsm, anchor="mm")

    for ri, (lbl, seed_img, step_imgs) in enumerate(rows):
        y0 = step_hdr_h + ri * (cell_h + row_label_h)
        for li, line in enumerate(lbl.split("\n")):
            draw.text((label_col_w // 2, y0 + 20 + li * 16),
                      line, fill=(255, 215, 80), font=flbl, anchor="mm")
        # seed image preview
        if seed_img is not None:
            thumb = seed_img.resize((thumb_w, thumb_w), Image.LANCZOS)
        else:
            thumb = Image.new("RGB", (thumb_w, thumb_w), (60, 60, 60))
            d = ImageDraw.Draw(thumb)
            d.text((thumb_w // 2, thumb_w // 2), "N/A", fill=(120, 120, 120),
                   font=fsm, anchor="mm")
        canvas.paste(thumb, (label_col_w, y0))
        for ci, (_, img) in enumerate(step_imgs):
            t = img.resize((thumb_w, thumb_w), Image.LANCZOS)
            canvas.paste(t, (label_col_w + (ci + 1) * thumb_w, y0))

    return canvas


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}")

    # Build seed images at generation resolution
    seed_images = {k: f(WIDTH, WIDTH) for k, f in SEED_IMAGE_FACTORIES.items()}
    for k, img in seed_images.items():
        img.save(os.path.join(OUT_DIR, f"seed_{k}.png"))
    print("Saved seed images")

    rows = []
    for label, kwargs in VARIANTS:
        print(f"\n── {label.replace(chr(10), ' | ')} ──")

        # Resolve seed image string → PIL image
        resolved_kwargs = dict(kwargs)
        seed_key = resolved_kwargs.get("noise_init_image")
        seed_pil = seed_images.get(seed_key) if isinstance(seed_key, str) else None
        if isinstance(seed_key, str):
            resolved_kwargs["noise_init_image"] = seed_pil

        pipe = SymmetricDiffusionPipeline.from_pretrained(
            MODEL, wallpaper_group=GROUP, torch_dtype=dtype)
        pipe.to(device)

        result = pipe(
            prompt=PROMPT, negative_prompt=NEGATIVE,
            num_inference_steps=STEPS, guidance_scale=CFG,
            width=WIDTH, seed=SEED,
            save_intermediate_steps=SNAP_STEPS,
            symmetric_noise=True,
            symmetrize_every_n=1,
            **resolved_kwargs,
        )
        final_img, intermediates = result if isinstance(result, tuple) else (result, {})

        step_imgs = [(s, intermediates.get(s, final_img)) for s in SNAP_STEPS]
        slug = label.replace("\n", "_").replace(" ", "_")[:40]
        for s, img in step_imgs:
            img.save(os.path.join(OUT_DIR, f"{slug}_step{s+1:03d}.png"))
        final_img.save(os.path.join(OUT_DIR, f"{slug}_final.png"))
        pipe.make_tiled_preview(final_img).save(
            os.path.join(OUT_DIR, f"{slug}_tiled.png"))
        rows.append((label, seed_pil, step_imgs))
        print(f"  done — {len(step_imgs)} snapshots")

        del pipe
        if device == "mps": torch.mps.empty_cache()
        elif device == "cuda": torch.cuda.empty_cache()

    grid = make_grid(rows)
    path = os.path.join(OUT_DIR, "noise_init_grid.png")
    grid.save(path)
    print(f"\nSaved grid: {path}")


if __name__ == "__main__":
    run()
