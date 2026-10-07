# View-based symmetric denoising ("plane sampler")

Code: `symmetric_diffusion/view_sampler.py`, runner: `run_views.py`.
Started 2026-10-06 as an alternative to the v3 pipeline in `pipeline.py`.

## Why a new sampler

Problems with the v3 approach (denoise the unit cell directly, circular padding):

1. **Hex groups are generated in a sheared frame.** The latent grid *is* the
   fractional (u,v) grid. For a 120° lattice the physical cell is a
   parallelogram, so the UNet draws in a frame that is sheared by ~30° relative
   to the final (preview) picture, and `rot120` is an affine map, not a
   rotation, in the UNet's frame.
2. **Latent sizes not divisible by 8 break periodicity inside the UNet.**
   440 px → 55 latents; three stride-2 downsamplings on an odd periodic length
   are not periodic. Pixel sizes must be multiples of 64 for exact wrap.
3. **Cell size is tied to the UNet resolution (512 px).** For p6m the
   asymmetric unit is 1/12 of that — tiny motifs.
4. **The UNet has an orientation prior.** In p4 / p6 the motif copies appear
   rotated; the model only ever sees one global orientation, so it prefers
   content that looks fine in any orientation (ornament/texture).
5. Interpolating noise (bilinear hex symmetrisation) changes its spectrum.

## The idea

Treat the **infinite wallpaper plane** as the object being denoised, with a
single parameterisation that is symmetric by construction:

```
physical point p ──B⁻¹──▶ fractional (u,v) mod 1 ──▶ texel ──▶ orbit representative
```

* State = one latent texture over the unit cell (texel grid in fractional
  coordinates). Every texel is mapped to a canonical representative of its
  orbit (min flat index over all group ops, `rep_full`). Only representatives
  are ever read or written, so every symmetric copy reads the *same* value —
  symmetry is exact with no averaging/projection step.
* `B` = lattice basis in physical latent-pixel units (a1 = (L,0),
  a2 = L·ar·(cos θ, sin θ)). This is what fixes the hex shear: views are
  sampled in physical space and mapped to fractional coordinates via `B⁻¹`.
* Each denoising step:
  1. Render K ordinary 64×64 latent **views** of the plane: random translation
     inside the cell, rotation taken from the group's own rotation angles
     (p4: 0/90/180/270, p6: multiples of 60°, ...). Nearest-neighbour lookup
     keeps the noise white (each view pixel reads one i.i.d. texel).
  2. Run the UNet (with CFG) on each view as a normal image — no circular
     padding needed, views are just crops of an infinite plane.
  3. Scatter the predicted ε of every view pixel back to its representative
     texel and average (MultiDiffusion-style least squares). Texels not hit
     this step keep the previous ε.
  4. Plain DDIM step on the texture.
* Output: bilinear render of an axis-aligned patch of the plane, VAE decode.

Rotating views by symmetry rotations means every motif copy is seen upright
in some view, and the fused ε is (approximately) the group-equivariant
average of the UNet — the same principle as Visual Anagrams (Geng et al.,
CVPR 2024) and MultiDiffusion / SyncDiffusion for panoramas.

Mirrors/glides need no special treatment: a mirrored copy is just another
region of the plane that the views happen to see.

## Knobs

| arg | meaning |
|-----|---------|
| `cell_size` | |a1| in latent px (×8 = image px). Smaller than the view (64) → views show several cells and the model sees the repetition. |
| `num_views` | views per step; cost scales linearly (one batch-2 UNet call per view). |
| `render_size` | output patch, latent px. |

## Known limitations / next ideas

* NN lookup at non-90° rotations aliases slightly (duplicated / skipped
  texels). Could use Gaussianity-preserving noise warping
  ("How I Warped Your Noise", Chang et al. ICLR 2024) or a finer texel grid.
* SD VAE latents are not exactly rotation/flip-equivariant; a pixel-space
  model (DeepFloyd IF) or a final pixel-space symmetrisation would help.
* Averaging ε over many views lowers its variance slightly → may soften
  detail. Try fewer, larger views, or stochastic (non-averaged) assignment.
* Symmetry alone does not produce Escher-style interlocking figure/ground —
  that needs an explicit region/tile constraint (see isohedral tilings,
  Kaplan's `tactile`; "Generative Escher Meshes", Aigerman & Groueix 2024).
  For the pure reflection groups (pmm, p3m1, p4m, p6m) tile outlines are
  forced to be mirror-line polygons, so interlocking has to come from several
  figures inside the fundamental domain.
* Scale randomisation of views (zoom in/out) to get coherent motifs at the
  cell scale and detail at smaller scales.

## First results (2026-10-06, p4, SD1.5, seed 42, 25 steps, "red and blue fish ...")

* Mechanics work: exact symmetry, seamless, all orbits covered every step,
  ~2 min per image on MPS (4 views × 25 steps, one batch-2 UNet call per view).
* Quality is **not yet better** than v3 (`experiments/views/baseline/`): the
  plane sampler output (`experiments/views/p4_cell56_s42.png`, `cfg5/`) is
  oversaturated and texture-like, with bright "star" artifacts at the 4-fold
  rotation centres. v3 is also flat/saturated at these settings but has
  cleaner shapes.
* ε shrinkage from averaging is **not** the cause: fused ε std ≈ per-view
  std (1.011 vs 1.015 at t=961, 0.938 vs 0.962 at t=561). `renorm_eps` is
  therefore ~a no-op.
* Memory: batching 4 views × CFG (batch 8) at 64×64 thrashes a 16 GB Mac;
  views are evaluated one at a time. fp16 VAE decode at 768 px → NaN; decode
  in fp32.

Next things to try:
1. `cell_size` ≥ view size (64–96) so a view isn't full of repeated content.
2. Views only at rotation 0 (isolate translation-averaging vs rotation effect).
3. Average x₀ instead of ε, or assign each texel to ONE random view per step
   (stochastic, no averaging) to avoid mixing disagreeing predictions.
4. Rotation-centre artifacts: texels on fixed points get several NN hits per
   view; weight scatter by 1/multiplicity or use a finer texel grid.

## Round 2 (2026-10-06): lessons from Visual Anagrams applied

Visual Anagrams (Geng, Park, Owens, CVPR 2024) key lessons: views must be
orthogonal transforms (pixel permutations) so the noisy image in every view is
still i.i.d. Gaussian; average ε across views (CFG per view); pixel-space
model preferred because latent transforms cause artifacts.

**Core tension found:** a symmetric image needs *identical* noise at symmetric
copies, which is not i.i.d. (each p4 texel appears ~5× in a 64² view, and all
copies meet at rotation centres). Visual Anagrams avoids this because it
constrains the *views*, not the image: every view permutes the SAME pixels,
so all views share one noise sample and agree.

All runs: p4, cell 80, 4 views, 25 steps, CFG 7.5, seed 42
(`experiments/views/ab/`, montages `montage_ab.png`, `montage_hybrid.png`).

| mode | idea | result |
|------|------|--------|
| `eps` | symmetric x_t, fuse ε | structure, but saturated / kaleidoscopic, artifacts at rotation centres (better than cell 56) |
| `eps --no-rotate` | same, no view rotation | ~identical → view rotation is irrelevant in eps mode |
| `x0` | symmetric x0, fresh i.i.d. noise per view each step | clean, natural colours, but smooth blobs (never commits to detail) |
| `sync` | x0 shared, per-view persistent DDIM noise | **collapse to flat colour**: copies see independent noise, disagree, average washes out |
| `hybrid --switch 0.3` | x0 for 30% of steps, then re-noise with symmetric noise → eps | organic creature-like interlocking shapes (eyes, scales); still saturated/busy |
| `hybrid --switch 0.5` | same, switch at 50% | natural colours, no saturation, red/blue shapes that **interlock with no background**; plush texture, but no recognisable fish |

Takeaways:
* Oversaturation/artifacts come from symmetric (non-i.i.d.) noise at HIGH
  noise levels. At lower noise levels symmetric noise is harmless.
* Independent noise per copy gives clean layout but no detail; a full
  i.i.d.-noise symmetric sampler doesn't work (sync collapse).
* The x0→eps hand-over point trades layout cleanliness vs. detail/"creature-ness".

Next: sweep switch 0.35–0.45, more steps (50), stronger object prompts,
then hex groups (p3/p6), which this sampler now renders unsheared.
Rendering at 1024 px with fp32 VAE thrashes 16 GB — use `--render 96`.

## Images in the repo

`experiments/` is gitignored. The key results are copied to `docs/img/`:

* `v3_baseline_p4_tiled.png` — v3 pipeline baseline (p4, 512², CFG 7.5)
* `montage_ab.png` — eps / eps-norot / x0 / x0-norot / sync comparison
* `montage_hybrid.png` — hybrid switch 0.3 vs 0.5
* `views_p4_hybrid0.5.png` — current best plane-sampler result

Reproduce the best one:
```bash
python run_views.py --groups p4 --cell 80 --views 4 --steps 25 --mode hybrid --switch 0.5 --render 96
```
