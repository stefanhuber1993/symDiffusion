#!/usr/bin/env python3
"""Test symmetrization in pixel space instead of latent space."""

import torch
import numpy as np
from PIL import Image
import os

from symmetric_diffusion import SymmetricDiffusionPipeline
from symmetric_diffusion.pipeline import compute_cell_height
from symmetric_diffusion.wallpaper_groups import get_group
from symmetric_diffusion.symmetrize import build_symmetry_map

def main():
    prompt = "red and blue fish, high contrast, crisp details"
    cfg = 5.0
    steps = 45
    seed = 42
    width = 320
    aspect_ratio = 1.5

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    group = get_group("pg")
    height = compute_cell_height(width, group, aspect_ratio)

    print(f"Device: {device}")
    print(f"Generating with pixel-space symmetrization")
    print(f"Prompt: {prompt}")
    print(f"Size: {width}x{height}")
    print()

    # Generate p1 (no symmetry constraints)
    print("Step 1: Generate p1 (no internal symmetry)...")
    pipe = SymmetricDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        wallpaper_group="p1",
        torch_dtype=dtype,
    )
    pipe.to(device)

    image_p1 = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality",
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=width,
        height=height,
        seed=seed,
        symmetrize_fraction=0.0,  # No symmetry
        use_nn_symmetrize=True,
    )

    os.makedirs("iteration/pixel_space", exist_ok=True)
    image_p1.save("iteration/pixel_space/01_p1_baseline.png")
    print("Saved: 01_p1_baseline.png")

    # Convert to numpy for pixel manipulation
    img_array = np.array(image_p1, dtype=np.float32) / 255.0

    # Now symmetrize in pixel space using pg symmetry
    print("\nStep 2: Symmetrize p1 image in pixel space (pg group)...")
    sym_map = build_symmetry_map(group, height, width)

    # Convert to torch for symmetrization
    img_torch = torch.from_numpy(img_array).to(device).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # Import symmetrize function
    from symmetric_diffusion.symmetrize import symmetrize_latent

    # Symmetrize the image directly
    img_sym = symmetrize_latent(img_torch, group, sym_map, use_nn_copy=True)

    # Convert back to PIL
    img_sym_np = img_sym.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_sym_np = np.clip(img_sym_np * 255, 0, 255).astype(np.uint8)
    image_pg_pixel = Image.fromarray(img_sym_np)
    image_pg_pixel.save("iteration/pixel_space/02_pg_pixel_space_symmetry.png")
    print("Saved: 02_pg_pixel_space_symmetry.png")

    # For comparison: also generate pg with latent-space symmetry
    print("\nStep 3: Generate pg with latent-space symmetry (baseline)...")
    del pipe
    torch.cuda.empty_cache()

    pipe = SymmetricDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        wallpaper_group="pg",
        torch_dtype=dtype,
    )
    pipe.to(device)

    image_pg_latent = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality",
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=width,
        height=height,
        seed=seed,
        symmetrize_fraction=0.6,
        symmetrize_noise_pred=True,
        symmetrize_every_n=8,
        use_nn_symmetrize=True,
    )

    image_pg_latent.save("iteration/pixel_space/03_pg_latent_space_symmetry.png")
    print("Saved: 03_pg_latent_space_symmetry.png")

    print("\n" + "="*60)
    print("Comparison saved!")
    print("  01_p1_baseline.png                  (free generation, no symmetry)")
    print("  02_pg_pixel_space_symmetry.png      (p1 + post-hoc symmetry in pixels)")
    print("  03_pg_latent_space_symmetry.png     (pg with latent-space symmetry)")
    print("\nCompare 02 vs 03 to see if pixel-space symmetry preserves detail better")
    print("="*60)

if __name__ == "__main__":
    main()
