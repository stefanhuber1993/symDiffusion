# Symmetric Diffusion — Escher-style Tessellation Generator

## What this project does
Generates seamlessly tileable images with wallpaper group symmetry using Stable Diffusion.
Two key modifications to the diffusion process:
1. **Circular padding** on all Conv2d layers → periodic boundary conditions → seamless tiling
2. **Symmetry enforcement** during denoising → wallpaper group symmetry within each tile

## Project structure
```
symmetric_diffusion/          # Main package
  wallpaper_groups.py         # All 17 wallpaper groups defined in fractional coordinates
  symmetrize.py               # Orbit-based (rect/square) + direct averaging (hex) symmetrization
  circular_padding.py         # Monkey-patches Conv2d to use circular padding
  pipeline.py                 # Modified StableDiffusionPipeline wrapper
  schedule.py                 # Strength schedule parser (historical, not currently used)
generate.py                   # CLI entry point
test_symmetry.py              # Math tests (no GPU/model needed)
run_experiments.py             # Batch experiment runner
run_comparison.py              # ON/OFF noise pred comparison
run_soft.py                    # Soft symmetry ablation
run_groups2.py                 # Glide/rotation group experiments
experiments/                   # Output images from experiments
```

## Running
```bash
# Tests (no GPU needed)
python3 test_symmetry.py

# Generate (requires HuggingFace login: python3 -c "from huggingface_hub import login; login()")
python3 generate.py --prompt "a tessellation of birds" --group p6 --seed 42
python3 generate.py --list-groups  # show all 17 groups with computed heights

# CLI options for lattice geometry:
#   --width 512          Base width (default 512)
#   --height 440         Explicit height (default: auto from lattice geometry)
#   --aspect-ratio 1.5   Override |a2|/|a1| ratio (for rectangular/oblique groups)
# Height is auto-computed as: width * aspect_ratio * sin(lattice_angle)
# e.g. hex groups → 512 * 1.0 * sin(120°) = 440
```

## Tech stack
- Python 3.12, PyTorch, HuggingFace diffusers
- Default model: `stable-diffusion-v1-5/stable-diffusion-v1-5` (SD 2.1 repo was taken down)
- Device: MPS (Apple Silicon), also supports CUDA and CPU

## Symmetrization mathematics

### Fractional coordinates
All symmetry ops work in (u,v) ∈ [0,1)², making them resolution-independent. Each
wallpaper group is defined as a list of `(u,v) → (u',v')` functions.

### Two symmetrization backends (auto-selected)
- **Orbit-based** (rect/square/centered-rect lattice: pm, pmm, p4, p4m, pg, pgg, cm, cmm,
  etc.): Union-find computes pixel orbits, then averages within each orbit. Exactly
  idempotent (P² = P).
- **Bilinear averaging** (hexagonal: p3, p6, p6m, etc.): For each pixel, bilinear-
  interpolates values at all symmetry-related positions (which fall between pixel centers)
  and averages. Required because 120°/60° rotations don't map pixel centers to pixel
  centers on a rectangular grid, causing orbit explosion under union-find (all pixels
  merge into one orbit at any resolution). Multiple passes (5) are used for initial noise
  to converge toward the symmetric subspace.

## Symmetry enforcement approach (RFdiffusion-inspired)

### Current approach (v3)
Inspired by how RFdiffusion enforces point group symmetry in protein diffusion:

1. **Symmetric initial noise by COPYING** (not averaging) for orbit-based groups:
   Generate noise for the asymmetric unit, then copy values to all symmetric partners.
   Each pixel remains N(0,1), preserving per-pixel variance. For hex groups (bilinear
   mode), noise is symmetrized via 5 passes of bilinear averaging then rescaled to
   unit variance.

2. **Per-step enforcement**: At each denoising step:
   a. Symmetrize the latent (prevents floating-point drift)
   b. UNet predicts noise from the symmetric input
   c. Symmetrize the noise prediction (average across symmetric copies)
   d. Normal DDIM step → symmetric x_{t-1}

3. Symmetry is exact by construction at every step. No strength tuning needed.

### Failed approaches (historical)
These are documented so we don't repeat them:

1. **Symmetrize x_{t-1} directly after scheduler step** (v1):
   Collapsed to gray. Symmetrization reduces latent variance at each step.
   Over 50 steps this drives the signal to zero. The only stable fixed point
   is the mean (gray).

2. **Symmetrize predicted x₀, recompute noise** (v2):
   Also collapsed to gray. Recomputing ε from symmetrized x₀ requires dividing
   by sqrt(1-α_t), which explodes for late timesteps (α_t → 1). Even the version
   that computes x_{t-1} directly (avoiding the division) collapsed because the
   fundamental issue is the same as v1.

3. **Symmetrize with blending strength / annealing schedule** (v1.5):
   Low strength (0.1-0.3) + every-N-steps produced visible symmetric shapes but
   heavily washed out. Very sensitive to parameters — not a robust solution.

4. **Symmetric noise by AVERAGING** (early v3):
   Averaging N(0,1) values reduces variance by 1/k for orbit size k. Even with
   sqrt(k) rescaling, the correlations between symmetric partners confused the
   model. Copying is the correct approach.

## Key experimental findings (2026-03-25)

### The symmetric initial noise dominates everything
Tested four configurations for pm group (all same seed):
- Both latent + noise_pred symmetrization: pattern A
- Latent only: pattern A (identical)
- Noise_pred only: pattern A (identical)
- Neither (only symmetric initial noise): pattern A (nearly identical, slight drift)

**Conclusion**: The UNet with circular padding naturally maintains the symmetry
from the initial noise. Per-step symmetrization is a near-no-op correction for
floating-point drift. The symmetric noise is what determines the output.

### Mirror symmetry produces flat/graphic results
| Group | Type | Visual quality |
|-------|------|---------------|
| **p1** | tiling only | Beautiful, detailed, realistic |
| **p2** | 180° rotation | Good depth and texture |
| **p4** | 4-fold rotation | **Excellent** — recognizable objects with rotational symmetry |
| **pm** | mirror | Flat, graphic, abstract — no recognizable objects |
| **pmm** | two mirrors | Very graphic, kaleidoscopic |
| **p4m** | rotation + mirrors | Bold geometric, kaleidoscopic |

**Rotation groups work much better than mirror groups.** The reason:
- Mirror symmetry forces latent channels at (i,j) = channels at (i, W-1-j).
  The VAE encodes directional features (edge orientation, lighting direction) in
  its channels. Mirroring destroys directional information → flat appearance.
- Rotation is more compatible with the VAE's features — a rotated feature
  is still a valid feature, just pointing a different direction.

### Guidance scale effect
- CFG 3.0: More variety, slightly chaotic
- CFG 5.0: Good balance for symmetric patterns
- CFG 7.5: Default, works OK
- CFG 12+: Oversaturated

### Recommended groups for Escher-style tessellations
- **p4** (4-fold rotation): Best quality, recognizable objects
- **p2** (180° rotation): Good quality, more variety
- **pg** (glide reflection): Potentially good — no pure mirror, just mirror + translation
- **p1** (tiling only): Best image quality but no internal symmetry

### Groups to avoid for realistic results
- **pm, pmm, cmm**: Pure mirrors kill realism
- **p4m, p6m**: Rotation + mirrors → kaleidoscope effect

### Lattice geometry and aspect ratio (2026-03-26)
The pixel grid must have the correct aspect ratio for the unit cell geometry.
For non-rectangular lattices the physical unit cell is a parallelogram; the pixel
height = width × (|a2|/|a1|) × sin(lattice_angle). This is auto-computed by
`compute_cell_height()` in pipeline.py. Examples:
- Square/rectangular (90°): height = width × aspect_ratio (square when ar=1)
- Hexagonal (120°): height = width × sin(120°) ≈ width × 0.866 → 512→440
- Oblique (arbitrary angle): height = width × ar × sin(angle)

Without the correct aspect ratio the UNet sees distorted geometry and hex rotations
look sheared rather than true 120° rotations. The tiled preview for hex groups uses
parallelogram tiling (each row shifted by half a tile width).

### Bug fix: pgg rotation center (2026-03-26)
The pgg group had the 180° rotation at the wrong center — (0,0) instead of (1/4,1/4).
This made the operations non-closed and generated an order-8 group instead of order-4.
Fixed to `(-u+0.5, -v+0.5)`. Caught by the group closure test.

## Open questions / future work
- Try fine-tuned models (LoRA trained on Escher/pattern art) which might handle symmetry better
- Consider two-pass approach: generate with symmetry → refine with img2img
- Could try symmetrizing in pixel space (VAE decode → symmetrize → VAE encode) instead of
  latent space, which would respect the VAE's learned representation better (but expensive)
- Hex bilinear symmetrization residual error ~0.1-0.2 after 5 passes — could try more
  passes or iterative refinement for tighter symmetry
