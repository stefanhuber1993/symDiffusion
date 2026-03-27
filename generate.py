#!/usr/bin/env python3
"""
Generate Escher-style symmetric tessellations using Stable Diffusion.

Examples:
    # p4 symmetry (4-fold rotation, square cell)
    python generate.py --prompt "interlocking lizards pattern" --group p4

    # p6 symmetry (6-fold rotation, hex cell — height auto-computed)
    python generate.py --prompt "a tessellation of fish" --group p6

    # p2 with non-square rectangular cell (aspect ratio 1.5)
    python generate.py --prompt "repeating floral pattern" --group p2 --aspect-ratio 1.5

    # Just tiling, no internal symmetry
    python generate.py --prompt "repeating floral pattern" --group p1

    # List all available wallpaper groups
    python generate.py --list-groups
"""

import argparse
import math
import os
import re


def main():
    parser = argparse.ArgumentParser(
        description="Generate symmetric tileable images with Stable Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prompt", type=str,
                        default="a tessellation of birds, M.C. Escher style")
    parser.add_argument("--negative-prompt", type=str,
                        default="blurry, low quality, text, watermark")
    parser.add_argument("--group", type=str, default="pm",
                        help="Wallpaper group (p1, pm, p4m, etc.)")
    parser.add_argument("--scheduler", type=str, default="ddim",
                        choices=["ddim", "dpm++"],
                        help="Noise scheduler. dpm++ is faster and more stochastic.")
    parser.add_argument("--model", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        help="Model ID. Use 'stabilityai/stable-diffusion-xl-base-1.0' for SDXL.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512,
                        help="Image width in pixels (must be divisible by 8)")
    parser.add_argument("--height", type=int, default=None,
                        help="Image height in pixels. If not set, computed from "
                             "lattice geometry (angle and aspect ratio)")
    parser.add_argument("--aspect-ratio", type=float, default=None,
                        help="Override |a2|/|a1| lattice vector ratio. "
                             "Default: 1.0 for square/hex, free for rectangular. "
                             "Only used when --height is not set.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename. If not set, auto-generated from group, "
                             "size, guidance scale, prompt, and a unique number.")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output files (default: current directory). "
                             "Ignored if --output is an absolute path.")
    parser.add_argument("--no-symmetrize-noise-pred", action="store_true",
                        help="Skip noise_pred symmetrization (only symmetrize latent). "
                             "Can give richer textures while keeping overall structure.")
    parser.add_argument("--symmetrize-fraction", type=float, default=1.0,
                        help="Fraction of (early) steps to apply symmetrization. "
                             "1.0=all steps (exact). 0.6=first 60%% then free detail. "
                             "Smaller = sharper details, looser symmetry.")
    parser.add_argument("--symmetrize-every-n", type=int, default=1,
                        help="Within symmetric phase, symmetrize every N steps. "
                             "Higher N = less bilinear blur (key for hex groups).")
    parser.add_argument("--noise-target-freq", type=float, default=0.0,
                        help="Bandpass-filter initial noise at this frequency "
                             "(cycles/latent-pixel). 0=white noise. Try 0.07-0.12 "
                             "to bias features toward the tile scale (Perlin-like).")
    parser.add_argument("--noise-freq-bw", type=float, default=0.04,
                        help="Gaussian bandwidth for --noise-target-freq bandpass.")
    parser.add_argument("--use-nn-symmetrize", action="store_true",
                        help="Use NN-copy (no bilinear blur) for hex group symmetrization. "
                             "Much sharper for p3/p6 but with slight boundary aliasing.")
    parser.add_argument("--apply-cell-shift", action="store_true",
                        help="Apply random periodic shift to unit cell at each symmetrization step. "
                             "Respects group constraints (continuous/discrete axes). "
                             "Adds positional diversity within the tile.")
    parser.add_argument("--noise-bandpass-alpha", type=float, default=1.0,
                        help="Blend weight for bandpassed noise (0=white, 1=full bandpass). "
                             "Lower values gently bias scale without dominating.")
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--list-groups", action="store_true")

    # Keep --size as a shorthand for --width (backward compat)
    parser.add_argument("--size", type=int, default=None,
                        help="Shorthand for --width (deprecated, use --width)")

    args = parser.parse_args()

    if args.size is not None:
        args.width = args.size

    if args.list_groups:
        from symmetric_diffusion.wallpaper_groups import list_groups, get_group
        print("Available wallpaper groups:\n")
        print(f"  {'Name':6s}  {'Lattice':24s}  {'Order':5s}  {'Angle':6s}  "
              f"{'|a2|/|a1|':9s}  Height for width=512")
        print(f"  {'----':6s}  {'-------':24s}  {'-----':5s}  {'-----':6s}  "
              f"{'---------':9s}  --------------------")
        for name in list_groups():
            g = get_group(name)
            from symmetric_diffusion.pipeline import compute_cell_height
            h = compute_cell_height(512, g)
            ar_str = f"{g.aspect_ratio:.1f}" if g.aspect_ratio else "free"
            print(f"  {name:6s}  {g.lattice:24s}  {g.order:5d}  "
                  f"{g.lattice_angle:5.0f}°  {ar_str:9s}  {h}")
        return

    import torch
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    is_xl = "xl" in args.model.lower()
    # float16 only on CUDA; MPS has incomplete float16 support (NaNs with SDXL)
    dtype = torch.float16 if device == "cuda" else torch.float32

    from symmetric_diffusion import SymmetricDiffusionPipeline
    from symmetric_diffusion.pipeline import compute_cell_height
    from symmetric_diffusion.wallpaper_groups import get_group

    group = get_group(args.group)

    # Compute actual dimensions
    width = args.width
    if args.height is not None:
        height = args.height
    else:
        height = compute_cell_height(width, group, args.aspect_ratio)

    print(f"Device: {device}")
    print(f"Group:  {args.group} (lattice={group.lattice}, "
          f"angle={group.lattice_angle}°, order={group.order})")
    print(f"Size:   {width}x{height} pixels")
    print(f"Prompt: {args.prompt}")
    print()

    print("Loading model...")
    pipe = SymmetricDiffusionPipeline.from_pretrained(
        args.model, wallpaper_group=args.group, torch_dtype=dtype,
        scheduler=args.scheduler,
    )
    pipe.to(device)

    print(f"Generating {width}x{height} image with {args.steps} steps...")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=width,
        height=height,
        seed=args.seed,
        symmetrize_fraction=args.symmetrize_fraction,
        symmetrize_noise_pred=not args.no_symmetrize_noise_pred,
        symmetrize_every_n=args.symmetrize_every_n,
        noise_target_freq=args.noise_target_freq,
        noise_freq_bandwidth=args.noise_freq_bw,
        noise_bandpass_alpha=args.noise_bandpass_alpha,
        use_nn_symmetrize=args.use_nn_symmetrize,
        apply_cell_shift=args.apply_cell_shift,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.output is None:
        # Build a short prompt slug: lowercase words, strip punctuation, max 4 words
        words = re.sub(r"[^a-z0-9 ]", "", args.prompt.lower()).split()
        slug = "_".join(words[:4])
        cfg_str = f"cfg{args.guidance_scale:g}"
        base = os.path.join(args.output_dir, f"{args.group}_{width}x{height}_{cfg_str}_{slug}")
        # Find a unique number to avoid collision
        n = 1
        while os.path.exists(f"{base}_{n:03d}.png"):
            n += 1
        output_path = f"{base}_{n:03d}.png"
    elif not os.path.isabs(args.output):
        output_path = os.path.join(args.output_dir, args.output)
    else:
        output_path = args.output

    stem, ext = os.path.splitext(output_path)
    tiled_path = f"{stem}_tiled{ext}"

    image.save(output_path)
    print(f"Saved single tile: {output_path}")

    tiled = pipe.make_tiled_preview(image, args.tiles, args.tiles)
    tiled.save(tiled_path)
    print(f"Saved tiled preview ({args.tiles}x{args.tiles}): {tiled_path}")


if __name__ == "__main__":
    main()
