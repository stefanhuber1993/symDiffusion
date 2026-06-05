"""
Experiment 1: every_n sweep with NO symmetric noise.
Tests every_n = 1, 2, 3, 4, 5 to find where the gray-collapse threshold is.
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
OUT_DIR = "experiments/everyn_sweep"
SNAP_STEPS = [4, 9, 14, 19, 24, 29]

VARIANTS = [
    # label,                                    group, sym_noise, skip,  every_n
    ("p1  reference",                            "p1",  True,  0.0, 1),
    ("pg  SYM noise  every 1\n(old default)",    "pg",  True,  0.0, 1),
    ("pg  NO sym noise\nevery 1  skip 0%",        "pg",  False, 0.0, 1),
    ("pg  NO sym noise\nevery 1  skip 10%",       "pg",  False, 0.10, 1),
    ("pg  NO sym noise\nevery 2  skip 0%",        "pg",  False, 0.0, 2),
    ("pg  NO sym noise\nevery 2  skip 10%",       "pg",  False, 0.10, 2),
    ("pg  NO sym noise\nevery 3  skip 0%",        "pg",  False, 0.0, 3),
    ("pg  NO sym noise\nevery 3  skip 10%",       "pg",  False, 0.10, 3),
    ("pg  NO sym noise\nevery 5  skip 10%",       "pg",  False, 0.10, 5),
]


def load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def make_grid(rows, thumb_w=175):
    lw, hdr = 240, 24
    total_w = lw + len(rows[0][1]) * thumb_w
    total_h = hdr + len(rows) * thumb_w
    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    fsm, flbl = load_font(13), load_font(12)
    for ci, (sn, _) in enumerate(rows[0][1]):
        draw.text((lw + ci*thumb_w + thumb_w//2, hdr//2),
                  "final" if sn == STEPS-1 else f"step {sn+1}",
                  fill=(180,180,180), font=fsm, anchor="mm")
    for ri, (lbl, imgs) in enumerate(rows):
        y0 = hdr + ri * thumb_w
        for li, line in enumerate(lbl.split("\n")):
            draw.text((lw//2, y0+20+li*15), line,
                      fill=(255,215,80), font=flbl, anchor="mm")
        for ci, (_, img) in enumerate(imgs):
            canvas.paste(img.resize((thumb_w, thumb_w), Image.LANCZOS),
                         (lw + ci*thumb_w, y0))
    return canvas


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}")
    rows = []
    for label, group, sym_noise, every_n in VARIANTS:
        print(f"\n── {label.replace(chr(10),' | ')} ──")
        pipe = SymmetricDiffusionPipeline.from_pretrained(MODEL, wallpaper_group=group, torch_dtype=dtype)
        pipe.to(device)
        result = pipe(prompt=PROMPT, negative_prompt=NEGATIVE,
                      num_inference_steps=STEPS, guidance_scale=CFG,
                      width=WIDTH, seed=SEED,
                      symmetric_noise=sym_noise, symmetrize_every_n=every_n,
                      save_intermediate_steps=SNAP_STEPS)
        final, ints = result if isinstance(result, tuple) else (result, {})
        imgs = [(s, ints.get(s, final)) for s in SNAP_STEPS]
        slug = label.replace("\n","_").replace(" ","_")[:50]
        for s, img in imgs:
            img.save(os.path.join(OUT_DIR, f"{slug}_step{s+1:03d}.png"))
        final.save(os.path.join(OUT_DIR, f"{slug}_final.png"))
        pipe.make_tiled_preview(final).save(os.path.join(OUT_DIR, f"{slug}_tiled.png"))
        rows.append((label, imgs))
        del pipe
        if device == "mps": torch.mps.empty_cache()
    make_grid(rows).save(os.path.join(OUT_DIR, "everyn_sweep.png"))
    print(f"\nSaved: {OUT_DIR}/everyn_sweep.png")

if __name__ == "__main__":
    run()
