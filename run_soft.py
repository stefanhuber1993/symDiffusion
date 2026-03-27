#!/usr/bin/env python3
"""Test soft symmetry: don't force latent symmetric every step."""

import torch, os, time
from symmetric_diffusion import SymmetricDiffusionPipeline
from symmetric_diffusion.symmetrize import symmetrize_latent

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32
OUT = "experiments"

prompt = "tessellation of colorful fish, seamless pattern, detailed, realistic"
neg = "blurry, low quality, text, watermark, flat, simple"

print("Loading pm...")
pipe = SymmetricDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    wallpaper_group="pm", torch_dtype=dtype)
pipe.to(device)

# Monkey-patch: remove per-step latent symmetrization, keep noise_pred sym
original_call = pipe.__call__.__wrapped__ if hasattr(pipe.__call__, '__wrapped__') else None

# Approach: generate with the pipeline but modify the denoising loop
# to NOT symmetrize the latent, only symmetrize noise_pred
import types
from symmetric_diffusion.schedule import parse_schedule

@torch.no_grad()
def soft_call(self, prompt, negative_prompt="", num_inference_steps=50,
              guidance_scale=7.5, height=512, width=512, seed=None,
              sym_latent=True, sym_noise=True):
    """Modified call with independent control over latent vs noise symmetrization."""
    device = self.pipe.device

    text_input = self.pipe.tokenizer(prompt, padding="max_length",
        max_length=self.pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt")
    text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(device))[0]

    uncond_input = self.pipe.tokenizer(negative_prompt, padding="max_length",
        max_length=self.pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt")
    uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latent_h, latent_w = height // 8, width // 8
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    sym_map = self._get_symmetry_map(latent_h, latent_w)
    latents = self._make_symmetric_noise(
        (1, self.pipe.unet.config.in_channels, latent_h, latent_w),
        sym_map, generator, self.pipe.unet.dtype, device)

    self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * self.pipe.scheduler.init_noise_sigma

    for step_idx, t in enumerate(self.pipe.scheduler.timesteps):
        if sym_latent:
            latents = self._symmetrize(latents, sym_map)

        latent_input = torch.cat([latents] * 2)
        latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
        noise_pred = self.pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if sym_noise:
            noise_pred = self._symmetrize(noise_pred, sym_map)

        latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Final symmetrize before decode
    latents = self._symmetrize(latents, sym_map)

    latents = latents / self.pipe.vae.config.scaling_factor
    image = self.pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    import numpy as np
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).round().astype(np.uint8)
    from PIL import Image
    return Image.fromarray(image)

pipe.soft_call = types.MethodType(soft_call, pipe)

configs = [
    # (name, sym_latent, sym_noise)
    ("soft_both",     True,  True),   # current approach
    ("soft_latonly",   True,  False),  # symmetrize latent only
    ("soft_noiseonly", False, True),   # symmetrize noise pred only
    ("soft_none",      False, False),  # symmetric initial noise only
]

for name, sl, sn in configs:
    start = time.time()
    img = pipe.soft_call(prompt=prompt, negative_prompt=neg, seed=42,
                         sym_latent=sl, sym_noise=sn)
    print(f"  {name}: {time.time()-start:.0f}s  latent={sl} noise={sn}")
    img.save(f"{OUT}/{name}.png")
    pipe.make_tiled_preview(img, 3, 3).save(f"{OUT}/{name}_tiled.png")

print("\nDone!")
