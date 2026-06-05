"""
Experiment 2: Best config (NO sym noise + every_n=3) across wallpaper groups.
Compares old default (SYM noise + every_n=1) vs new config for each group.
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
OUT_DIR = "experiments/groups_best_config"
SNAP_STEPS = [4, 9, 14, 19, 24, 29]

# (label, group, sym_noise, every_n)
VARIANTS = [
    # Reference
    ("p1",           "p1",  True,  1),
    # Old vs new for each interesting group
    ("p2  OLD",      "p2",  True,  1),
    ("p2  NEW",      "p2",  False, 3),
    ("p4  OLD",      "p4",  True,  1),
    ("p4  NEW",      "p4",  False, 3),
    ("pg  OLD",      "pg",  True,  1),
    ("pg  NEW",      "pg",  False, 3),
    ("p6  OLD",      "p6",  True,  1),
    ("p6  NEW",      "p6",  False, 3),
]


def load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def make_grid(rows, thumb_w=170):
    lw, hdr = 180, 24
    total_w = lw + len(rows[0][1]) * thumb_w
    total_h = hdr + len(rows) * thumb_w
    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    fsm, flbl = load_font(13), load_font(13)
    for ci, (sn, _) in enumerate(rows[0][1]):
        draw.text((lw + ci*thumb_w + thumb_w//2, hdr//2),
                  "final" if sn == STEPS-1 else f"step {sn+1}",
                  fill=(180,180,180), font=fsm, anchor="mm")
    for ri, (lbl, imgs) in enumerate(rows):
        y0 = hdr + ri * thumb_w
        col = (255, 215, 80) if "NEW" in lbl else (180, 180, 180) if "OLD" in lbl else (120, 200, 120)
        draw.text((lw//2, y0 + thumb_w//2), lbl, fill=col, font=flbl, anchor="mm")
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
        print(f"\n── {label} ──")
        pipe = SymmetricDiffusionPipeline.from_pretrained(MODEL, wallpaper_group=group, torch_dtype=dtype)
        pipe.to(device)
        result = pipe(prompt=PROMPT, negative_prompt=NEGATIVE,
                      num_inference_steps=STEPS, guidance_scale=CFG,
                      width=WIDTH, seed=SEED,
                      symmetric_noise=sym_noise, symmetrize_every_n=every_n,
                      save_intermediate_steps=SNAP_STEPS)
        final, ints = result if isinstance(result, tuple) else (result, {})
        imgs = [(s, ints.get(s, final)) for s in SNAP_STEPS]
        slug = label.replace(" ","_")
        for s, img in imgs:
            img.save(os.path.join(OUT_DIR, f"{slug}_step{s+1:03d}.png"))
        final.save(os.path.join(OUT_DIR, f"{slug}_final.png"))
        pipe.make_tiled_preview(final).save(os.path.join(OUT_DIR, f"{slug}_tiled.png"))
        rows.append((label, imgs))
        del pipe
        if device == "mps": torch.mps.empty_cache()
    make_grid(rows).save(os.path.join(OUT_DIR, "groups_best_config.png"))
    print(f"\nSaved: {OUT_DIR}/groups_best_config.png")

if __name__ == "__main__":
    run()
