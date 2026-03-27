#!/usr/bin/env python3
"""Test per-step pixel-space symmetrization with intermediate capture."""

import torch
import os

from symmetric_diffusion import SymmetricDiffusionPipeline
from symmetric_diffusion.pipeline import compute_cell_height
from symmetric_diffusion.wallpaper_groups import get_group

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
    print(f"Testing per-step pixel-space symmetrization")
    print(f"Prompt: {prompt}")
    print(f"Size: {width}x{height}, Steps: {steps}")
    print()

    # Capture intermediates at these steps
    capture_steps = [-1, 0, 10, 20, 30, 40, 44]
    os.makedirs("iteration/pixel_sym_per_step", exist_ok=True)

    configs = [
        ("p1_baseline", "p1", False, 0, False, 0),
        ("pg_latent_sym", "pg", True, 0.6, False, 0),
        ("pg_pixel_sym_every_5", "pg", False, 0.0, True, 5),
        ("pg_pixel_sym_every_3", "pg", False, 0.0, True, 3),
        ("pg_pixel_sym_every_1", "pg", False, 0.0, True, 1),
    ]

    for config_name, group_name, use_latent_sym, lat_sym_frac, use_pixel_sym, pix_sym_every_n in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"Group: {group_name}, Latent_sym: {use_latent_sym}, Pixel_sym: {use_pixel_sym} (every {pix_sym_every_n})")
        print(f"{'='*60}")

        group = get_group(group_name)
        height = compute_cell_height(width, group, aspect_ratio)

        pipe = SymmetricDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            wallpaper_group=group_name,
            torch_dtype=dtype,
        )
        pipe.to(device)

        print(f"Generating with intermediates...")
        result = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            seed=seed,
            symmetrize_fraction=lat_sym_frac if use_latent_sym else 0.0,
            symmetrize_noise_pred=True,
            symmetrize_every_n=8 if use_latent_sym else 1,
            noise_target_freq=0.0,
            noise_freq_bandwidth=0.04,
            noise_bandpass_alpha=1.0,
            use_nn_symmetrize=True,
            apply_cell_shift=False,
            pixel_space_symmetrize=use_pixel_sym,
            pixel_symmetrize_every_n=pix_sym_every_n if use_pixel_sym else 1,
            save_intermediate_steps=capture_steps,
        )

        final_image, intermediates = result

        # Save final
        final_path = f"iteration/pixel_sym_per_step/{config_name}_final.png"
        final_image.save(final_path)
        print(f"Saved final: {final_path}")

        # Save intermediates
        for step_idx, img in sorted(intermediates.items()):
            img_path = f"iteration/pixel_sym_per_step/{config_name}_step_{step_idx:02d}.png"
            img.save(img_path)

        print(f"Saved {len(intermediates)} intermediate steps")

        # Clean up
        del pipe
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"Images saved in iteration/pixel_sym_per_step/")
    print(f"Compare:")
    print(f"  - p1_baseline_* vs pg_latent_sym_* (current approach)")
    print(f"  - pg_latent_sym_* vs pg_pixel_sym_every_* (new approaches)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
