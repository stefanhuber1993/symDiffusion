"""
View-based symmetric denoising ("plane sampler").

The state of the diffusion process is a single latent texture over the unit
cell, of which only one texel per symmetry orbit (the "representative") is
ever read or written. The infinite wallpaper plane is defined implicitly:

    physical point p  ->  fractional (u,v) = B^-1 p  ->  texel  ->  representative

At every step we render K ordinary 64x64 latent "views" of the plane (random
translation, rotation drawn from the group's own rotation angles), run the
UNet on them as normal images, and scatter the predicted noise back to the
representative texels (averaging over all view pixels that hit them). Then a
plain DDIM step updates the texture.

Consequences:
  * Symmetry is exact by construction (every copy reads the same texel).
  * The UNet always sees undistorted physical geometry — hex groups are no
    longer generated in a sheared frame.
  * Cell size is decoupled from the UNet resolution; views may span several
    cells. No circular padding is needed: views are crops of an infinite plane.
  * Views rotated by the group's rotations show each motif copy upright in
    some view, so the UNet's orientation prior is averaged over all copies.

See docs/view_based_sampler.md for the rationale.
"""

import math
import torch
import numpy as np
from PIL import Image

from .wallpaper_groups import WallpaperGroup


def lattice_basis(group: WallpaperGroup, cell_size: float,
                  aspect_ratio: float | None = None) -> torch.Tensor:
    """2x2 matrix whose columns are a1, a2 in latent-pixel units (x right, y down)."""
    ar = aspect_ratio if aspect_ratio is not None else (group.aspect_ratio or 1.0)
    th = math.radians(group.lattice_angle)
    a1 = (cell_size, 0.0)
    a2 = (cell_size * ar * math.cos(th), cell_size * ar * math.sin(th))
    return torch.tensor([[a1[0], a2[0]], [a1[1], a2[1]]], dtype=torch.float64)


def rotation_angles(group: WallpaperGroup, B: torch.Tensor) -> list[float]:
    """Physical rotation angles (radians) of the group's point group."""
    p = (0.3137, 0.1711)
    d = 1e-4
    Binv = torch.linalg.inv(B)
    angles = set()
    for op in group.operations:
        o = op(*p)
        cols = []
        for dp in ((d, 0.0), (0.0, d)):
            q = op(p[0] + dp[0], p[1] + dp[1])
            cols.append([((q[k] - o[k] + 0.5) % 1.0 - 0.5) / d for k in range(2)])
        A = torch.tensor(cols, dtype=torch.float64).T
        L = B @ A @ Binv
        if torch.linalg.det(L) > 0:
            angles.add(round(math.degrees(math.atan2(L[1, 0], L[0, 0]))) % 360)
    return sorted(math.radians(a) for a in angles)


class PlaneSampler:
    def __init__(self, sd_pipeline, group: WallpaperGroup, cell_size: int = 56,
                 aspect_ratio: float | None = None, view_size: int = 64):
        self.pipe = sd_pipeline
        self.group = group
        self.V = view_size
        self.B = lattice_basis(group, cell_size, aspect_ratio)
        self.Binv = torch.linalg.inv(self.B)
        ar = aspect_ratio if aspect_ratio is not None else (group.aspect_ratio or 1.0)
        # Texel grid over the fractional cell; even sizes for half-cell glides.
        self.N = 2 * round(cell_size / 2)
        self.M = 2 * round(cell_size * ar / 2)
        self.rep_full = self._build_representatives()
        self.angles = rotation_angles(group, self.B)

    # ------------------------------------------------------------------ geometry
    def _texel_index(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        iu = torch.floor(u * self.N).long() % self.N
        iv = torch.floor(v * self.M).long() % self.M
        return iv * self.N + iu

    def _build_representatives(self) -> torch.Tensor:
        """For each texel, the flat index of its orbit representative (min index)."""
        iv, iu = torch.meshgrid(torch.arange(self.M), torch.arange(self.N), indexing="ij")
        u = ((iu + 0.5) / self.N).double().flatten()
        v = ((iv + 0.5) / self.M).double().flatten()
        keys = [self._texel_index(*op(u, v)) for op in self.group.operations]
        return torch.stack(keys).min(dim=0).values

    def _plane_to_frac(self, xy: torch.Tensor):
        f = xy @ self.Binv.T  # (P,2)
        return f[:, 0] % 1.0, f[:, 1] % 1.0

    def _view_coords(self, origin, angle: float, size: int) -> torch.Tensor:
        r = torch.arange(size, dtype=torch.float64) - size / 2 + 0.5
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        c, s = math.cos(angle), math.sin(angle)
        x = origin[0] + c * xx - s * yy
        y = origin[1] + s * xx + c * yy
        return torch.stack([x.flatten(), y.flatten()], dim=1)

    def view_indices(self, origin, angle: float) -> torch.Tensor:
        """Representative texel index for every pixel of a VxV view (NN lookup).

        Nearest-neighbour keeps the noise white (no interpolation blur)."""
        u, v = self._plane_to_frac(self._view_coords(origin, angle, self.V))
        return self.rep_full[self._texel_index(u, v)]

    # ------------------------------------------------------------------ sampling
    def _encode(self, prompt, negative_prompt, device):
        cond, uncond = self.pipe.encode_prompt(
            prompt, device, 1, True, negative_prompt=negative_prompt)
        return cond, uncond

    @torch.no_grad()
    def __call__(self, prompt: str, negative_prompt: str = "",
                 num_inference_steps: int = 30, guidance_scale: float = 7.5,
                 num_views: int = 4, seed: int = 0, render_size: int = 96,
                 mode: str = "x0", rotate: bool = True, switch: float = 0.4,
                 log=print) -> Image.Image:
        """mode:
          "eps" - state is a symmetric x_t; views read it directly (noise is
                  duplicated across symmetric copies -> not i.i.d. in a view);
                  fused eps drives a DDIM step.
          "x0"  - state is the symmetric clean estimate x0. Each view gets
                  FRESH i.i.d. noise: x_t = sqrt(a) x0_view + sqrt(1-a) z, so the
                  UNet always sees in-distribution noise (Visual Anagrams'
                  i.i.d. requirement). Fused Tweedie x0 is the new state
                  (SyncTweedies-style x0 synchronisation). Blurry: fresh noise
                  every step never lets the sampler commit to detail.
          "sync" - like "x0", but views are FIXED for the whole run and each
                  keeps its own deterministic DDIM noise (its last eps
                  prediction): x_t^k = sqrt(a) x0_view + sqrt(1-a) eps_k.
                  Only x0 is shared; per-view noise stays i.i.d.-like and
                  carries detail across steps. Collapses to a flat colour:
                  symmetric copies see independent noise, disagree, and the
                  average washes out.
          "hybrid" - "x0" for the first `switch` fraction of steps (clean global
                  layout, in-distribution noise), then re-noise the layout with
                  SYMMETRIC noise and continue in "eps" mode (detail).
        """
        pipe = self.pipe
        device, dtype = pipe.device, pipe.unet.dtype
        cond, uncond = self._encode(prompt, negative_prompt, device)
        K = num_views
        text = torch.cat([uncond, cond])

        g = torch.Generator("cpu").manual_seed(seed)
        C = pipe.unet.config.in_channels
        T = self.M * self.N
        V = self.V
        reps_dev = self.rep_full.unique().to(device)
        canon = torch.randn(1, C, T, generator=g).to(device)  # x_T, fp32
        x0 = torch.zeros(C, T, device=device)
        eps_prev = canon[0].clone()

        sched = pipe.scheduler
        sched.set_timesteps(num_inference_steps, device=device)
        ac = sched.alphas_cumprod.to(device)
        rng = np.random.default_rng(seed)
        angles = self.angles if rotate else [0.0]

        def sample_views(step):
            idx = []
            for k in range(K):
                origin = (self.B @ torch.tensor(rng.random(2))).tolist()
                angle = angles[(k + step) % len(angles)]
                idx.append(self.view_indices(origin, angle))
            return torch.stack(idx).to(device)  # (K, V*V)

        if mode == "sync":
            idx = sample_views(0)
            z_views = torch.randn(C, K, V * V, generator=g).to(device)

        switch_step = int(switch * num_inference_steps) if mode == "hybrid" else -1
        if mode == "hybrid":
            mode = "x0"
        for step, t in enumerate(sched.timesteps):
            a = ac[t]
            if step == switch_step:
                # SDEdit-style hand-over: x_t = sqrt(a) x0 + sqrt(1-a) * symmetric noise
                mode = "eps"
                canon = (a.sqrt() * x0 + (1 - a).sqrt()
                         * torch.randn(C, T, generator=g).to(device)).unsqueeze(0)
                eps_prev = canon[0].clone()
            if mode != "sync":
                idx = sample_views(step)

            if mode == "eps":
                views = canon[0][:, idx]
            elif mode == "sync":
                views = a.sqrt() * x0[:, idx] + (1 - a).sqrt() * z_views
            else:
                z = torch.randn(C, K, V * V, generator=g).to(device)
                views = a.sqrt() * x0[:, idx] + (1 - a).sqrt() * z
            views = views.permute(1, 0, 2).reshape(K, C, V, V)

            eps_views = []
            for k in range(K):  # one view per UNet call keeps memory at batch 2
                inp = sched.scale_model_input(torch.cat([views[k:k + 1]] * 2).to(dtype), t)
                pred = pipe.unet(inp, t, encoder_hidden_states=text).sample.float()
                pu, pc = pred.chunk(2)
                eps_views.append(pu + guidance_scale * (pc - pu))
            eps_views = torch.cat(eps_views)  # (K,C,V,V)
            if not torch.isfinite(eps_views).all():
                raise RuntimeError(f"non-finite UNet output at step {step}")

            if mode == "eps":
                target = eps_views
            else:
                target = (views - (1 - a).sqrt() * eps_views) / a.sqrt()  # Tweedie x0

            # Scatter back onto representative texels and average
            flat_idx = idx.flatten()
            acc = torch.zeros(C, T, device=device)
            cnt = torch.zeros(T, device=device)
            acc.index_add_(1, flat_idx, target.permute(1, 0, 2, 3).reshape(C, -1))
            cnt.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
            hit = cnt > 0
            fused = acc / cnt.clamp(min=1)

            if mode == "eps":
                eps_prev = torch.where(hit, fused, eps_prev)
                canon = sched.step(eps_prev.unsqueeze(0), t, canon).prev_sample
                x0 = canon[0]
            else:
                x0 = torch.where(hit, fused, x0)
                if mode == "sync":
                    z_views = eps_views.reshape(K, C, V * V).permute(1, 0, 2)

            if step % 5 == 0 or step == switch_step:
                log(f"  [{mode}] step {step:3d}  t={int(t):4d}  x0 std {x0[:, reps_dev].std().item():.3f}  "
                    f"coverage {int(hit[reps_dev].sum())}/{reps_dev.numel()}")

        return self._render(x0.unsqueeze(0), render_size)

    # ------------------------------------------------------------------ output
    @torch.no_grad()
    def _render(self, canon: torch.Tensor, size: int) -> Image.Image:
        """Bilinear render of an axis-aligned size x size latent patch of the plane."""
        pipe = self.pipe
        device = canon.device
        C = canon.shape[1]
        full = canon[0][:, self.rep_full.to(device)].reshape(C, self.M, self.N)

        xy = self._view_coords((size / 2, size / 2), 0.0, size)
        u, v = self._plane_to_frac(xy)
        pj = (u * self.N - 0.5).float().to(device)
        pi = (v * self.M - 0.5).float().to(device)
        j0, i0 = torch.floor(pj), torch.floor(pi)
        wj, wi = pj - j0, pi - i0
        j0, i0 = j0.long() % self.N, i0.long() % self.M
        j1, i1 = (j0 + 1) % self.N, (i0 + 1) % self.M
        lat = ((1 - wi) * (1 - wj) * full[:, i0, j0] + (1 - wi) * wj * full[:, i0, j1]
               + wi * (1 - wj) * full[:, i1, j0] + wi * wj * full[:, i1, j1])
        lat = lat.reshape(1, C, size, size).float()
        if not torch.isfinite(lat).all():
            raise RuntimeError("non-finite latent before decode")

        # fp16 VAE decode overflows to NaN at larger sizes on MPS; decode in fp32
        vae_dtype = pipe.vae.dtype
        pipe.vae.to(torch.float32)
        img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
        pipe.vae.to(vae_dtype)
        img = (img / 2 + 0.5).nan_to_num(0).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray((img * 255).round().astype(np.uint8))
