"""Quick runner for the view-based plane sampler (symmetric_diffusion/view_sampler.py)."""
import argparse, time
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from symmetric_diffusion.wallpaper_groups import get_group
from symmetric_diffusion.view_sampler import PlaneSampler

ap = argparse.ArgumentParser()
ap.add_argument("--groups", default="p4,p6,p3,pg,p2,pgg")
ap.add_argument("--prompt", default="red and blue fish, tightly intertwined, 3d rendering, detailed")
ap.add_argument("--negative", default="blurry, lowres, text, watermark, white background")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--views", type=int, default=4)
ap.add_argument("--cell", type=int, default=56, help="cell size |a1| in latent px (x8 = image px)")
ap.add_argument("--render", type=int, default=128, help="output patch size in latent px")
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--mode", default="x0", choices=["x0", "eps", "sync", "hybrid"])
ap.add_argument("--no-rotate", action="store_true")
ap.add_argument("--switch", type=float, default=0.4, help="hybrid: fraction of steps in x0 mode")
ap.add_argument("--out", default="experiments/views")
a = ap.parse_args()

model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(
    model, scheduler=DDIMScheduler.from_pretrained(model, subfolder="scheduler"),
    torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.set_progress_bar_config(disable=True)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for name in a.groups.split(","):
    t0 = time.time()
    s = PlaneSampler(pipe, get_group(name), cell_size=a.cell)
    print(f"{name}: texels {s.M}x{s.N}, orbits {s.rep_full.unique().numel()}, "
          f"rotations {[round(x * 57.2958) for x in s.angles]}")
    img = s(a.prompt, a.negative, a.steps, a.cfg, a.views, a.seed, a.render,
            mode=a.mode, rotate=not a.no_rotate, switch=a.switch)
    f = out / f"{name}_cell{a.cell}_{a.mode}{a.switch if a.mode == 'hybrid' else ''}{'_norot' if a.no_rotate else ''}_s{a.seed}.png"
    img.save(f)
    print(f"  saved {f}  ({time.time() - t0:.0f}s)", flush=True)
