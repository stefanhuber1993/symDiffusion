#!/usr/bin/env python3
"""Compare p1 vs pg diffusion process to find where symmetry kills detail."""

import torch
import os
from symmetric_diffusion import SymmetricDiffusionPipeline
from symmetric_diffusion.pipeline import compute_cell_height
from symmetric_diffusion.wallpaper_groups import get_group

def main():
    # Best parameters found so far
    prompt = "red and blue fish, high contrast, crisp details"
    cfg = 5.0
    steps = 45
    seed = 42
    symmetrize_fraction = 0.6
    symmetrize_every_n = 8
    width = 320
    aspect_ratio = 1.5

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Device: {device}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}, CFG: {cfg}, Seed: {seed}")
    print(f"Symmetrize: fraction={symmetrize_fraction}, every_n={symmetrize_every_n}")
    print()

    # Capture intermediates at these steps
    # -1 = initial noise (before any denoising)
    # 0, 5, 10... = after denoising step
    capture_steps = [-1, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44]

    for group_name in ["p1", "pg"]:
        print(f"\n{'='*60}")
        print(f"Generating {group_name.upper()}")
        print(f"{'='*60}")

        group = get_group(group_name)
        height = compute_cell_height(width, group, aspect_ratio)

        print(f"Loading model...")
        pipe = SymmetricDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            wallpaper_group=group_name,
            torch_dtype=dtype,
        )
        pipe.to(device)

        print(f"Generating {width}x{height} with intermediates...")
        result = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            seed=seed,
            symmetrize_fraction=symmetrize_fraction,
            symmetrize_noise_pred=True,
            symmetrize_every_n=symmetrize_every_n,
            noise_target_freq=0.0,
            noise_freq_bandwidth=0.04,
            noise_bandpass_alpha=1.0,
            use_nn_symmetrize=True,
            apply_cell_shift=False,
            save_intermediate_steps=capture_steps,
        )

        final_image, intermediates = result

        # Save final image
        os.makedirs("iteration/debug", exist_ok=True)
        output_path = f"iteration/debug/{group_name}_final.png"
        final_image.save(output_path)
        print(f"Saved final image: {output_path}")

        # Save intermediates
        for step_idx, img in sorted(intermediates.items()):
            img_path = f"iteration/debug/{group_name}_step_{step_idx:02d}.png"
            img.save(img_path)

        print(f"Saved {len(intermediates)} intermediate steps")

        # Clean up to save memory
        del pipe
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"Images saved in iteration/debug/")
    print(f"Compare {group_name}_step_*.png files side-by-side")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
