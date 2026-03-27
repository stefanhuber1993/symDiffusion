#!/usr/bin/env python3
"""Quick comparison: symmetrize noise pred vs not, and multiple groups."""

import torch, os, time
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32
OUT = "experiments"
os.makedirs(OUT, exist_ok=True)

from symmetric_diffusion import SymmetricDiffusionPipeline

prompt = "tessellation of colorful fish, seamless pattern, detailed, realistic"
neg = "blurry, low quality, text, watermark, flat, simple"


def run(pipe, name, sym_noise_pred=True, seed=42, guidance=7.5):
    start = time.time()
    img = pipe(prompt=prompt, negative_prompt=neg, seed=seed,
               guidance_scale=guidance, symmetrize_noise_pred=sym_noise_pred)
    print(f"  {name}: {time.time()-start:.0f}s")
    img.save(f"{OUT}/{name}.png")
    pipe.make_tiled_preview(img, 3, 3).save(f"{OUT}/{name}_tiled.png")


# === Compare: sym noise pred ON vs OFF for pm ===
print("Loading pm...")
pipe = SymmetricDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    wallpaper_group="pm", torch_dtype=dtype)
pipe.to(device)

print("\n=== pm: noise_pred symmetrization ON vs OFF ===")
run(pipe, "cmp_pm_sym_ON",  sym_noise_pred=True,  seed=42)
run(pipe, "cmp_pm_sym_OFF", sym_noise_pred=False, seed=42)

# Also test with different seed
run(pipe, "cmp_pm_sym_ON_s2",  sym_noise_pred=True,  seed=123)
run(pipe, "cmp_pm_sym_OFF_s2", sym_noise_pred=False, seed=123)

del pipe; torch.mps.empty_cache()

# === Test multiple groups with sym_noise_pred=False ===
print("\n=== Multiple groups (noise_pred sym OFF) ===")
for gname in ["p1", "p2", "pm", "pmm", "p4", "p4m"]:
    print(f"Loading {gname}...")
    pipe = SymmetricDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        wallpaper_group=gname, torch_dtype=dtype)
    pipe.to(device)
    run(pipe, f"grp_{gname}_off", sym_noise_pred=False, seed=42)
    del pipe; torch.mps.empty_cache()

print("\nDone!")
