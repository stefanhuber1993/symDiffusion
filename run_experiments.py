#!/usr/bin/env python3
"""
Run a batch of experiments to understand artifact sources and test groups.
Saves all results to experiments/ directory.
"""

import torch
import os
import time
from PIL import Image

# Determine device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Device: {device}, dtype: {dtype}")

from symmetric_diffusion import SymmetricDiffusionPipeline
from symmetric_diffusion.wallpaper_groups import get_group
from symmetric_diffusion.symmetrize import build_symmetry_map, symmetrize_latent

OUT = "experiments"
os.makedirs(OUT, exist_ok=True)


def run(pipe, name, prompt, seed=42, steps=50, guidance=7.5, size=512):
    """Generate and save a single experiment."""
    start = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, text, watermark",
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=size, width=size,
        seed=seed,
    )
    elapsed = time.time() - start

    image.save(f"{OUT}/{name}.png")
    tiled = pipe.make_tiled_preview(image, 3, 3)
    tiled.save(f"{OUT}/{name}_tiled.png")
    print(f"  {name}: {elapsed:.1f}s")
    return image


# ============================================================
# Experiment 1: Diagnose artifacts - symmetric noise only vs full method
# ============================================================
print("\n=== Experiment 1: Artifact diagnosis (pm group) ===")

# Load model once
print("Loading model...")
pipe_pm = SymmetricDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    wallpaper_group="pm", torch_dtype=dtype,
)
pipe_pm.to(device)

# 1a. Full method (current): symmetric noise + symmetrize noise_pred every step
run(pipe_pm, "pm_full_method", "intertwined red and blue fish, seamless pattern", seed=42)

# 1b. Different seeds to see variation
for seed in [123, 456, 789]:
    run(pipe_pm, f"pm_seed{seed}", "intertwined red and blue fish, seamless pattern", seed=seed)

# 1c. Different guidance scales
for g in [3.0, 5.0, 7.5, 12.0]:
    run(pipe_pm, f"pm_cfg{g}", "intertwined red and blue fish, seamless pattern",
        seed=42, guidance=g)

# 1d. Different prompts
prompts = [
    ("birds", "tessellation of colorful birds, M.C. Escher style, interlocking"),
    ("butterflies", "tessellation of butterflies, ornamental, seamless pattern"),
    ("geometric", "geometric tessellation pattern, Islamic art style, intricate"),
    ("flowers", "repeating flower pattern, William Morris style, ornate"),
]
for pname, prompt in prompts:
    run(pipe_pm, f"pm_{pname}", prompt, seed=42)


# ============================================================
# Experiment 2: Test multiple wallpaper groups
# ============================================================
print("\n=== Experiment 2: Multiple wallpaper groups ===")

prompt = "tessellation of colorful birds, M.C. Escher style, interlocking, seamless"

groups_to_test = ["p1", "p2", "pm", "pmm", "p4", "p4m"]

for gname in groups_to_test:
    print(f"  Loading group {gname}...")
    pipe_g = SymmetricDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        wallpaper_group=gname, torch_dtype=dtype,
    )
    pipe_g.to(device)
    run(pipe_g, f"group_{gname}", prompt, seed=42)
    del pipe_g
    if device == "mps":
        torch.mps.empty_cache()


print(f"\nAll experiments saved to {OUT}/")
print("Done!")
