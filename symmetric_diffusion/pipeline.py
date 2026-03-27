"""
Modified Stable Diffusion pipeline with periodic boundary conditions
and wallpaper group symmetry enforcement.

Approach (inspired by RFdiffusion for symmetric protein design):

1. Generate noise for the asymmetric unit, COPY to fill the full image
   via symmetry operations. This preserves per-pixel variance (no averaging).

2. At each denoising step:
   a. Enforce symmetry on the latent (project onto symmetric subspace).
      Since the latent should already be symmetric, this is a near-no-op
      that just prevents floating-point drift.
   b. The UNet sees a symmetric input and predicts noise.
   c. Average the noise prediction across symmetric copies.
      Since the input was symmetric, the predictions are already nearly
      symmetric — this is a small correction, not a major modification.
   d. Normal DDIM step with the averaged noise → symmetric x_{t-1}.

Symmetry is exact by construction at every step. No strength tuning needed.
"""

import math
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image
import numpy as np

from .wallpaper_groups import WallpaperGroup, get_group
from .symmetrize import symmetrize_latent, build_symmetry_map, SymmetryMap, _symmetrize_nn_copy
from .circular_padding import patch_conv2d_circular


def compute_cell_height(width: int, group: WallpaperGroup,
                        aspect_ratio: float | None = None) -> int:
    """Compute the correct pixel height for a unit cell.

    The pixel grid must have the right aspect ratio so that the UNet sees
    undistorted geometry. For non-rectangular lattices, the physical unit
    cell is a parallelogram; the pixel height corresponds to the vertical
    extent (perpendicular component of a2).

    height = width * (|a2|/|a1|) * sin(lattice_angle)

    Args:
        width: Pixel width (along a1 direction).
        group: The wallpaper group (carries lattice_angle and default aspect_ratio).
        aspect_ratio: Override for |a2|/|a1|. If None, uses the group's default
            (1.0 for square/hex, free for rectangular — defaults to 1.0).

    Returns:
        Height in pixels, rounded to a multiple of 8 (for the VAE).
    """
    ar = aspect_ratio if aspect_ratio is not None else (group.aspect_ratio or 1.0)
    angle_rad = math.radians(group.lattice_angle)
    h = width * ar * math.sin(angle_rad)
    # Round to nearest multiple of 8
    h = max(8, round(h / 8) * 8)
    return h


class SymmetricDiffusionPipeline:
    """Stable Diffusion pipeline that generates symmetric, tileable images.

    Usage:
        pipe = SymmetricDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            wallpaper_group="pm",
        )
        image = pipe("a tessellation of birds")
    """

    def __init__(
        self,
        sd_pipeline: StableDiffusionPipeline,
        group: WallpaperGroup,
    ):
        self.pipe = sd_pipeline
        self.group = group
        self._symmetry_map_cache: dict[tuple[int, int], SymmetryMap] = {}

        patch_conv2d_circular(self.pipe.unet)
        patch_conv2d_circular(self.pipe.vae.decoder)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        wallpaper_group: str = "p1",
        torch_dtype: torch.dtype = torch.float16,
        scheduler: str = "ddim",
        **kwargs,
    ) -> "SymmetricDiffusionPipeline":
        sched_cls = {
            "ddim": DDIMScheduler,
            "dpm++": DPMSolverMultistepScheduler,
        }.get(scheduler, DDIMScheduler)
        sched = sched_cls.from_pretrained(model_id, subfolder="scheduler")
        is_xl = "xl" in model_id.lower()
        pipe_cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
        sd_pipe = pipe_cls.from_pretrained(
            model_id, scheduler=sched, torch_dtype=torch_dtype, **kwargs,
        )
        group = get_group(wallpaper_group)
        return cls(sd_pipe, group)

    def to(self, device: str | torch.device) -> "SymmetricDiffusionPipeline":
        self.pipe = self.pipe.to(device)
        return self

    def _get_symmetry_map(self, H: int, W: int) -> SymmetryMap:
        key = (H, W)
        if key not in self._symmetry_map_cache:
            self._symmetry_map_cache[key] = build_symmetry_map(self.group, H, W)
        return self._symmetry_map_cache[key]

    def _symmetrize(self, x: torch.Tensor, sym_map: SymmetryMap,
                    use_nn: bool = False) -> torch.Tensor:
        """Project onto the symmetric subspace."""
        return symmetrize_latent(x, self.group, sym_map, use_nn_copy=use_nn)

    @staticmethod
    def _apply_cell_shift(latent: torch.Tensor, group: WallpaperGroup) -> torch.Tensor:
        """Apply random periodic shift to the latent tensor.

        Shifts are applied with wrapping via torch.roll, respecting the group's
        symmetry constraints. This allows features to appear at different positions
        within the unit cell while preserving wallpaper symmetry.

        For axes with continuous freedom (no glides/centering): random uniform in [0, 1)
        For axes with discrete freedom (glide/centering ±0.5): random choice {0, 0.5}

        Args:
            latent: Tensor of shape (B, C, H, W)
            group: Wallpaper group defining constraints

        Returns:
            Shifted latent with periodic wrapping applied
        """
        B, C, H, W = latent.shape
        device = latent.device

        # Determine random shifts in fractional cell coordinates
        if group.requires_even_W:
            # Discrete: u ∈ {0, 0.5}
            shift_u = torch.randint(0, 2, (1,), device=device).float() * 0.5
        else:
            # Continuous: u ∈ [0, 1)
            shift_u = torch.rand(1, device=device)

        if group.requires_even_H:
            # Discrete: v ∈ {0, 0.5}
            shift_v = torch.randint(0, 2, (1,), device=device).float() * 0.5
        else:
            # Continuous: v ∈ [0, 1)
            shift_v = torch.rand(1, device=device)

        # Convert fractional shifts to pixel shifts
        shift_j = int(round(shift_u.item() * W)) % W
        shift_i = int(round(shift_v.item() * H)) % H

        # Apply periodic shift via torch.roll (dims 2=H, 3=W)
        if shift_i != 0 or shift_j != 0:
            latent = torch.roll(latent, shifts=(shift_i, shift_j), dims=(2, 3))

        return latent

    @staticmethod
    def _bandpass_noise(
        noise: torch.Tensor, target_freq: float, bandwidth: float
    ) -> torch.Tensor:
        """Apply a Gaussian bandpass filter in Fourier space to bias spatial scale.

        target_freq: peak frequency in cycles per latent pixel (e.g. 0.08 for
            features spanning ~12 pixels).  0 = DC (no filter).
        bandwidth: Gaussian sigma in frequency units (0.03–0.1 typical).
        """
        B, C, H, W = noise.shape
        device = noise.device
        dtype = noise.dtype

        noise_f = torch.fft.rfft2(noise.float())

        ky = torch.fft.fftfreq(H, device=device).reshape(-1, 1)
        kx = torch.fft.rfftfreq(W, device=device).reshape(1, -1)
        k_mag = (ky**2 + kx**2).sqrt()

        mask = torch.exp(-0.5 * ((k_mag - target_freq) / bandwidth) ** 2)
        noise_f = noise_f * mask

        noise = torch.fft.irfft2(noise_f, s=(H, W)).to(dtype)
        std = noise.std()
        if std > 1e-8:
            noise = noise / std
        return noise

    def _make_symmetric_noise(
        self, shape: tuple, sym_map: SymmetryMap,
        generator: torch.Generator, dtype: torch.dtype, device: torch.device,
        target_freq: float = 0.0, freq_bandwidth: float = 0.04,
    ) -> torch.Tensor:
        """Generate noise that is symmetric by construction.

        For orbit-based groups: copy the asymmetric-unit value to all partners
        (preserves N(0,1) per pixel).

        For hex groups: use NN-copy from canonical representative — zero
        bilinear blur compared to the old multi-pass averaging approach.

        If target_freq > 0, first apply a Gaussian bandpass filter to bias the
        spatial scale of features (Perlin-like frequency shaping).
        """
        noise = torch.randn(shape, generator=generator, dtype=dtype).to(device)

        if target_freq > 0.0:
            bp = self._bandpass_noise(noise.clone(), target_freq, freq_bandwidth)
            # Blend: 0=pure white, 1=pure bandpassed. Renormalize std.
            alpha = getattr(self, '_bandpass_alpha', 1.0)
            noise = (1 - alpha) * noise + alpha * bp
            std = noise.std()
            if std > 1e-8:
                noise = noise / std

        B, C, H, W = shape

        if sym_map.use_orbits:
            orbit_flat = sym_map.orbit_flat.to(device)
            n_orbits = sym_map.n_orbits

            flat = noise.reshape(B, C, H * W)

            first_pixel = torch.zeros(n_orbits, dtype=torch.long, device=device)
            seen = torch.zeros(n_orbits, dtype=torch.bool, device=device)
            for idx in range(H * W):
                oid = orbit_flat[idx].item()
                if not seen[oid]:
                    first_pixel[oid] = idx
                    seen[oid] = True

            orbit_values = flat[:, :, first_pixel]
            idx = orbit_flat.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
            noise = orbit_values.gather(2, idx).reshape(B, C, H, W)
        else:
            # Hex groups: one symmetrization pass. NN-copy = zero blur but
            # some aliasing; bilinear = smooth but slightly blurry.
            use_nn = getattr(self, '_use_nn_symmetrize', False)
            noise = symmetrize_latent(noise, self.group, sym_map,
                                      use_nn_copy=use_nn)
            std = noise.std()
            if std > 1e-8:
                noise = noise / std

        return noise

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int | None = None,
        aspect_ratio: float | None = None,
        seed: int | None = None,
        symmetrize_noise_pred: bool = True,
        symmetrize_fraction: float = 1.0,
        symmetrize_every_n: int = 1,
        noise_target_freq: float = 0.0,
        noise_freq_bandwidth: float = 0.04,
        noise_bandpass_alpha: float = 1.0,
        use_nn_symmetrize: bool = False,
        apply_cell_shift: bool = False,
        pixel_space_symmetrize: bool = False,
        pixel_symmetrize_every_n: int = 1,
        save_intermediate_steps: list[int] | None = None,
    ) -> Image.Image | tuple[Image.Image, dict]:
        """Generate a symmetric, tileable image.

        Args:
            prompt: Text prompt for image generation.
            negative_prompt: Negative prompt.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale (5-10 recommended with symmetry).
            width: Image width in pixels (divisible by 8).
            height: Image height in pixels (divisible by 8). If None,
                computed automatically from the lattice geometry.
            aspect_ratio: Override for |a2|/|a1| ratio. If None, uses the
                group's default. Only meaningful when height is None.
            seed: Random seed for reproducibility.
            symmetrize_fraction: Fraction of (early) denoising steps to apply
                symmetrization. 1.0 = all steps (exact symmetry). 0.6 = only
                first 60% of steps, then free generation for detail. Smaller
                values give sharper details but looser symmetry.
            symmetrize_every_n: Within the symmetric phase, only symmetrize
                every N steps. Default 1 (every step). Higher values reduce
                cumulative bilinear blur (key issue for hex groups) while
                still correcting symmetry drift.
            noise_target_freq: If > 0, bandpass-filter the initial noise to
                bias feature scale. In cycles/latent-pixel; e.g. 0.08 for
                features ~12 latents wide. 0 = white noise (default).
            noise_freq_bandwidth: Gaussian sigma for the bandpass (freq units).
            use_nn_symmetrize: Use NN-copy (no bilinear blur) for hex groups.
            apply_cell_shift: Apply random periodic shift to the unit cell at each
                symmetrization step. Respects group constraints (discrete/continuous
                axes). Adds positional diversity. Default False.

        Returns:
            PIL Image of the generated tileable, symmetric pattern.
        """
        if height is None:
            height = compute_cell_height(width, self.group, aspect_ratio)

        device = self.pipe.device
        is_xl = isinstance(self.pipe, StableDiffusionXLPipeline)

        if is_xl:
            # Use diffusers' built-in SDXL prompt encoder (handles dual encoders correctly)
            (cond_emb, uncond_emb,
             pooled_cond, pooled_uncond) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
            )
            text_embeddings = torch.cat([uncond_emb, cond_emb])
            add_text_embeds = torch.cat([pooled_uncond, pooled_cond])
            # time_ids: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
            time_ids = torch.tensor(
                [[height, width, 0, 0, height, width]],
                dtype=add_text_embeds.dtype, device=device,
            ).repeat(2, 1)  # uncond + cond
            unet_kwargs = {"added_cond_kwargs": {"text_embeds": add_text_embeds, "time_ids": time_ids}}
        else:
            # SD 1.5: single text encoder
            text_input = self.pipe.tokenizer(
                prompt, padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            cond_emb = self.pipe.text_encoder(text_input.input_ids.to(device))[0]
            uncond_input = self.pipe.tokenizer(
                negative_prompt, padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            uncond_emb = self.pipe.text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_emb, cond_emb])
            unet_kwargs = {}

        # Prepare symmetric initial noise
        latent_h = height // 8
        latent_w = width // 8
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        sym_map = self._get_symmetry_map(latent_h, latent_w)

        self._bandpass_alpha = noise_bandpass_alpha
        self._use_nn_symmetrize = use_nn_symmetrize
        latents = self._make_symmetric_noise(
            (1, self.pipe.unet.config.in_channels, latent_h, latent_w),
            sym_map, generator, self.pipe.unet.dtype, device,
            target_freq=noise_target_freq, freq_bandwidth=noise_freq_bandwidth,
        )

        # Set up scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        latents = latents * self.pipe.scheduler.init_noise_sigma

        sym_cutoff = int(num_inference_steps * symmetrize_fraction)

        # For debugging: track intermediate latents
        intermediates = {} if save_intermediate_steps else None

        # Capture initial latent (before denoising) if step -1 requested
        if save_intermediate_steps and -1 in save_intermediate_steps:
            lat_scaled = latents / self.pipe.vae.config.scaling_factor
            img = self.pipe.vae.decode(lat_scaled).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            img = img.cpu().permute(0, 2, 3, 1).float().numpy()
            img = (img[0] * 255).round().astype(np.uint8)
            intermediate_images = {-1: Image.fromarray(img)}
        else:
            intermediate_images = {}

        # Denoising loop
        for step_idx, t in enumerate(timesteps):
            do_sym = (step_idx < sym_cutoff) and (step_idx % symmetrize_every_n == 0)
            do_pixel_sym = pixel_space_symmetrize and (step_idx % pixel_symmetrize_every_n == 0)

            # Pixel-space symmetrization: decode → symmetrize → encode
            if do_pixel_sym:
                # Decode to pixel space
                lat_scaled = latents / self.pipe.vae.config.scaling_factor
                pixels = self.pipe.vae.decode(lat_scaled).sample
                # pixels are in [-1, 1], convert to [0, 1]
                pixels = (pixels + 1) / 2

                # Build symmetry map for pixel space (different dimensions than latent)
                B, C, H, W = pixels.shape
                sym_map_pix = build_symmetry_map(self.group, H, W)

                # Symmetrize in pixel space
                pixels_sym = symmetrize_latent(pixels, self.group, sym_map_pix,
                                               use_nn_copy=use_nn_symmetrize)

                # Convert back to [-1, 1] and encode
                pixels_sym = pixels_sym * 2 - 1
                lat_encoded = self.pipe.vae.encode(pixels_sym).latent_dist.sample()
                latents = lat_encoded * self.pipe.vae.config.scaling_factor

            # Enforce symmetry on latent (prevents floating-point drift)
            if do_sym:
                latents = self._symmetrize(latents, sym_map, use_nn=use_nn_symmetrize)
                if apply_cell_shift:
                    latents = self._apply_cell_shift(latents, self.group)

            # UNet prediction on symmetric input
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.pipe.unet(
                latent_input, t, encoder_hidden_states=text_embeddings,
                **unet_kwargs,
            ).sample

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Optionally symmetrize the noise prediction.
            # When True: exact symmetry, but flatter/more graphic look.
            # When False: latent is re-symmetrized next step anyway, so symmetry
            # is approximate but textures are richer.
            if do_sym and symmetrize_noise_pred:
                noise_pred = self._symmetrize(noise_pred, sym_map,
                                              use_nn=use_nn_symmetrize)

            # Normal DDIM step — produces symmetric x_{t-1} since both
            # the latent and noise_pred are symmetric
            latents = self.pipe.scheduler.step(
                noise_pred, t, latents
            ).prev_sample

            # Save intermediate if requested
            if save_intermediate_steps and step_idx in save_intermediate_steps:
                intermediates[step_idx] = latents.clone().detach()

        # Decode
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents).sample

        image = (image / 2 + 0.5).nan_to_num(0.0).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype(np.uint8)
        final_image = Image.fromarray(image)

        # If intermediates were saved, decode them as well
        if save_intermediate_steps and intermediates:
            for step_idx, lat in intermediates.items():
                lat_scaled = lat / self.pipe.vae.config.scaling_factor
                img = self.pipe.vae.decode(lat_scaled).sample
                img = (img / 2 + 0.5).clamp(0, 1)
                img = img.cpu().permute(0, 2, 3, 1).float().numpy()
                img = (img[0] * 255).round().astype(np.uint8)
                intermediate_images[step_idx] = Image.fromarray(img)
            return final_image, intermediate_images

        return final_image

    def make_tiled_preview(
        self, image: Image.Image, tiles_x: int = 3, tiles_y: int = 3
    ) -> Image.Image:
        """Create a tiled preview showing the image repeated in a grid.

        For hexagonal lattice groups, tiles as parallelograms to show
        the true hexagonal geometry. For all other lattices, tiles as
        rectangles.
        """
        w, h = image.size

        if self.group.lattice == "hexagonal":
            return self._make_hex_tiled_preview(image, tiles_x, tiles_y)

        tiled = Image.new("RGB", (w * tiles_x, h * tiles_y))
        for iy in range(tiles_y):
            for ix in range(tiles_x):
                tiled.paste(image, (ix * w, iy * h))
        return tiled

    def _make_hex_tiled_preview(
        self, image: Image.Image, tiles_x: int = 3, tiles_y: int = 3
    ) -> Image.Image:
        """Tile a hexagonal-lattice image as parallelograms.

        The hexagonal unit cell has lattice vectors a1 and a2 at 120°.
        In the output, a1 points right and a2 points up-left at 120°.
        Each row of tiles is shifted left by half a tile width.
        """
        w, h = image.size
        img_arr = np.array(image)

        # Output canvas: wide enough for the parallelogram shifts
        # a1 = (w, 0), a2 = (-w/2, h) in the output pixel space
        extra_w = w * tiles_y // 2
        out_w = w * tiles_x + extra_w
        out_h = h * tiles_y

        # Vectorized: build coordinate grids
        oy = np.arange(out_h)
        ox = np.arange(out_w)
        ox_grid, oy_grid = np.meshgrid(ox, oy)

        # Output pixel to fractional coordinates
        # a1 = (w, 0), a2 = (-w/2, h)
        # P = u*a1 + v*a2 => px = u*w - v*w/2, py = v*h
        # => v = py/h, u = (px + v*w/2) / w = px/w + v/2
        # Shift origin so pattern is centered
        v_frac = oy_grid.astype(np.float64) / h
        u_frac = (ox_grid.astype(np.float64) - extra_w / 2) / w + v_frac / 2

        u_frac = u_frac % 1.0
        v_frac = v_frac % 1.0

        # Bilinear sampling from source image
        pj = u_frac * w - 0.5
        pi = v_frac * h - 0.5

        j0 = np.floor(pj).astype(int) % w
        i0 = np.floor(pi).astype(int) % h
        j1 = (j0 + 1) % w
        i1 = (i0 + 1) % h
        wj = pj - np.floor(pj)
        wi = pi - np.floor(pi)

        wj = wj[:, :, None]
        wi = wi[:, :, None]

        out = (
            (1 - wi) * (1 - wj) * img_arr[i0, j0].astype(np.float32)
            + (1 - wi) * wj * img_arr[i0, j1].astype(np.float32)
            + wi * (1 - wj) * img_arr[i1, j0].astype(np.float32)
            + wi * wj * img_arr[i1, j1].astype(np.float32)
        )
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
