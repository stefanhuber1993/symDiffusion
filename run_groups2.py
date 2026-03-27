#!/usr/bin/env python3
"""Test glide/rotation groups that avoid pure mirrors."""

import torch, os, time
from symmetric_diffusion import SymmetricDiffusionPipeline

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32
OUT = "experiments"

prompt = "tessellation of colorful fish, seamless pattern, detailed, realistic"
neg = "blurry, low quality, text, watermark, flat, simple"

# Groups without pure mirrors: pg, pgg, p2, p3, p6
# Plus pmg (one mirror + one glide) for comparison
groups = ["pg", "pgg", "pmg", "p2", "p3", "p6"]

for gname in groups:
    print(f"Loading {gname}...")
    pipe = SymmetricDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        wallpaper_group=gname, torch_dtype=dtype)
    pipe.to(device)

    start = time.time()
    img = pipe(prompt=prompt, negative_prompt=neg, seed=42)
    print(f"  grp2_{gname}: {time.time()-start:.0f}s")
    img.save(f"{OUT}/grp2_{gname}.png")
    pipe.make_tiled_preview(img, 3, 3).save(f"{OUT}/grp2_{gname}_tiled.png")

    del pipe
    torch.mps.empty_cache()

print("\nDone!")
