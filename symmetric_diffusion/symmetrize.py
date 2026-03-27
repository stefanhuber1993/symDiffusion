"""
Symmetrize a latent tensor according to a wallpaper group.

Two strategies, automatically selected:

1. **Orbit-based** (rectangular/square lattice groups): Compute pixel orbits
   via union-find, then average within each orbit. Exactly idempotent (P^2 = P).

2. **Bilinear averaging** (hexagonal lattice groups): For each pixel, gather
   values at all symmetry-related positions using bilinear interpolation and
   average. Hex rotations (120deg, 60deg) don't map pixel centers to pixel centers
   on a rectangular grid, so bilinear interpolation is essential for accuracy.
   The result is approximately idempotent with very small residual error.

Both approaches correctly enforce the wallpaper group symmetry. The orbit-based
method is preferred when applicable because it's exactly idempotent.
"""

import math
import torch
import numpy as np
from .wallpaper_groups import WallpaperGroup


# Lattice types where discretized ops map pixels to pixels exactly
_EXACT_LATTICES = {"rectangular", "centered_rectangular", "square"}


def _build_op_maps(group: WallpaperGroup, H: int, W: int) -> np.ndarray:
    """Build per-operation pixel maps (nearest-neighbor).

    Returns:
        op_map: int array of shape (n_ops, H*W) where op_map[k, flat_idx]
            is the flat pixel index that operation k maps flat_idx to.
    """
    n_ops = group.order
    op_map = np.zeros((n_ops, H * W), dtype=np.int64)

    for k, op in enumerate(group.operations):
        for i in range(H):
            for j in range(W):
                u = (j + 0.5) / W
                v = (i + 0.5) / H
                u2, v2 = op(u, v)
                j2 = int(u2 * W) % W
                i2 = int(v2 * H) % H
                op_map[k, i * W + j] = i2 * W + j2

    return op_map


def _build_bilinear_op_data(group: WallpaperGroup, H: int, W: int):
    """Build per-operation bilinear interpolation data for non-exact lattices.

    For each operation and each pixel, computes the 4 neighbor indices and
    bilinear weights needed to sample from the symmetry-related position.

    Returns:
        grids: int64 array of shape (n_ops, H*W, 4) — flat indices of 4 neighbors
        weights: float64 array of shape (n_ops, H*W, 4) — bilinear weights
    """
    n_ops = group.order
    N = H * W
    grids = np.zeros((n_ops, N, 4), dtype=np.int64)
    weights = np.zeros((n_ops, N, 4), dtype=np.float64)

    for k, op in enumerate(group.operations):
        for i in range(H):
            for j in range(W):
                u = (j + 0.5) / W
                v = (i + 0.5) / H
                u2, v2 = op(u, v)

                # Target position in pixel coordinates (0-indexed, center at +0.5)
                pj = (u2 % 1.0) * W - 0.5
                pi = (v2 % 1.0) * H - 0.5

                # Bilinear interpolation neighbors
                j0 = int(math.floor(pj))
                i0 = int(math.floor(pi))
                j1 = j0 + 1
                i1 = i0 + 1

                wj = pj - j0
                wi = pi - i0

                # Wrap indices for periodic boundary
                j0w = j0 % W
                j1w = j1 % W
                i0w = i0 % H
                i1w = i1 % H

                flat_idx = i * W + j
                grids[k, flat_idx, 0] = i0w * W + j0w
                grids[k, flat_idx, 1] = i0w * W + j1w
                grids[k, flat_idx, 2] = i1w * W + j0w
                grids[k, flat_idx, 3] = i1w * W + j1w

                weights[k, flat_idx, 0] = (1 - wi) * (1 - wj)
                weights[k, flat_idx, 1] = (1 - wi) * wj
                weights[k, flat_idx, 2] = wi * (1 - wj)
                weights[k, flat_idx, 3] = wi * wj

    return grids, weights


def _build_orbit_data(op_map: np.ndarray, H: int, W: int):
    """Union-find to compute orbits from operation maps.

    Returns (orbit_flat, orbit_sizes, n_orbits).
    """
    N = H * W
    n_ops = op_map.shape[0]
    parent = np.arange(N, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for k in range(n_ops):
        for idx in range(N):
            union(idx, op_map[k, idx])

    # Relabel to consecutive IDs
    root_to_id = {}
    orbit_flat = np.zeros(N, dtype=np.int64)
    next_id = 0
    for idx in range(N):
        root = find(idx)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        orbit_flat[idx] = root_to_id[root]

    n_orbits = next_id
    orbit_sizes = np.zeros(n_orbits, dtype=np.int64)
    for oid in orbit_flat:
        orbit_sizes[oid] += 1

    return orbit_flat, orbit_sizes, n_orbits


def _build_canonical_source_map(group: WallpaperGroup, H: int, W: int) -> np.ndarray:
    """Build a canonical-copy source map for hex groups (no bilinear blur).

    For each pixel, find the symmetry-related fractional position with the
    lexicographically smallest (u, v) — a consistent total ordering that defines
    a unique canonical representative for each orbit.  Convert to pixel index
    via nearest-neighbour (no averaging).

    Unlike bilinear averaging, this introduces zero low-pass blur.  The only
    artefact is ≤0.5-pixel NN rounding at symmetry boundaries, which the
    diffusion process easily handles.
    """
    N = H * W
    source_map = np.zeros(N, dtype=np.int64)

    for i in range(H):
        for j in range(W):
            u = (j + 0.5) / W
            v = (i + 0.5) / H

            best_u, best_v = u, v
            for op in group.operations:
                u2, v2 = op(u, v)
                if (u2, v2) < (best_u, best_v):
                    best_u, best_v = u2, v2

            j_src = int(best_u * W) % W
            i_src = int(best_v * H) % H
            source_map[i * W + j] = i_src * W + j_src

    return source_map


class SymmetryMap:
    """Precomputed symmetry data for efficient latent symmetrization."""

    def __init__(self, group: WallpaperGroup, H: int, W: int):
        if group.requires_even_H and H % 2 != 0:
            raise ValueError(
                f"Group '{group.name}' has half-cell glide translations along v "
                f"and requires even H, got H={H}"
            )
        if group.requires_even_W and W % 2 != 0:
            raise ValueError(
                f"Group '{group.name}' has half-cell glide translations along u "
                f"and requires even W, got W={W}"
            )
        self.group = group
        self.H = H
        self.W = W
        self.use_orbits = group.lattice in _EXACT_LATTICES

        if self.use_orbits:
            op_map = _build_op_maps(group, H, W)
            orbit_flat, orbit_sizes, n_orbits = _build_orbit_data(op_map, H, W)
            self.orbit_flat = torch.from_numpy(orbit_flat)
            self.orbit_sizes = torch.from_numpy(orbit_sizes)
            self.n_orbits = n_orbits
            self.op_grids = None
            self.op_weights = None
            self.op_map = None
            self.canonical_source = None
        else:
            # Use bilinear interpolation for denoising-loop symmetrization
            grids, weights = _build_bilinear_op_data(group, H, W)
            self.op_grids = torch.from_numpy(grids)
            self.op_weights = torch.from_numpy(weights).float()
            self.orbit_flat = None
            self.orbit_sizes = None
            self.n_orbits = None
            self.op_map = None
            # Also build a no-blur NN-copy map (used for initial noise and
            # optionally as a faster low-blur alternative during denoising)
            self.canonical_source = torch.from_numpy(
                _build_canonical_source_map(group, H, W)
            )


def build_symmetry_map(group: WallpaperGroup, H: int, W: int) -> SymmetryMap:
    """Precompute symmetry data for a given group and spatial size."""
    return SymmetryMap(group, H, W)


def symmetrize_latent(
    latent: torch.Tensor,
    group: WallpaperGroup,
    symmetry_map: SymmetryMap | None = None,
    use_nn_copy: bool = False,
) -> torch.Tensor:
    """Project a latent tensor onto the symmetric subspace of a wallpaper group.

    Args:
        latent: Tensor of shape (B, C, H, W).
        group: The wallpaper group.
        symmetry_map: Precomputed SymmetryMap. Computed on-the-fly if None.
        use_nn_copy: For hex groups, use nearest-neighbour copy instead of
            bilinear averaging.  Zero low-pass blur — ideal for initial noise
            seeding or when cumulative blur is a concern.  Ignored for
            orbit-based groups (they are already blur-free).

    Returns:
        Symmetrized tensor of the same shape.
    """
    B, C, H, W = latent.shape

    if symmetry_map is None:
        symmetry_map = build_symmetry_map(group, H, W)

    if symmetry_map.use_orbits:
        return _symmetrize_orbit(latent, symmetry_map)
    elif use_nn_copy and symmetry_map.canonical_source is not None:
        return _symmetrize_nn_copy(latent, symmetry_map)
    else:
        return _symmetrize_bilinear(latent, symmetry_map)


def _symmetrize_orbit(latent: torch.Tensor, sm: SymmetryMap) -> torch.Tensor:
    """Orbit-based symmetrization (exactly idempotent)."""
    B, C, H, W = latent.shape
    device = latent.device

    orbit_flat = sm.orbit_flat.to(device)
    orbit_sizes = sm.orbit_sizes.to(device).to(latent.dtype)

    flat = latent.reshape(B, C, H * W)
    idx = orbit_flat.unsqueeze(0).unsqueeze(0).expand(B, C, -1)

    orbit_sums = torch.zeros(B, C, sm.n_orbits, device=device, dtype=latent.dtype)
    orbit_sums.scatter_add_(2, idx, flat)
    orbit_means = orbit_sums / orbit_sizes.unsqueeze(0).unsqueeze(0)
    result = orbit_means.gather(2, idx)

    return result.reshape(B, C, H, W)


def _symmetrize_bilinear(latent: torch.Tensor, sm: SymmetryMap) -> torch.Tensor:
    """Bilinear averaging over symmetry operations.

    For each pixel p, computes: (1/|G|) Σ_{g in G} interp(x, g(p))
    where interp uses bilinear interpolation with periodic wrapping.
    """
    B, C, H, W = latent.shape
    device = latent.device
    n_ops = sm.op_grids.shape[0]

    op_grids = sm.op_grids.to(device)      # (n_ops, H*W, 4)
    op_weights = sm.op_weights.to(device).to(latent.dtype)  # (n_ops, H*W, 4)
    flat = latent.reshape(B, C, H * W)

    accumulated = torch.zeros_like(flat)
    for k in range(n_ops):
        for n in range(4):
            idx = op_grids[k, :, n].unsqueeze(0).unsqueeze(0).expand(B, C, -1)
            w = op_weights[k, :, n].unsqueeze(0).unsqueeze(0).expand(B, C, -1)
            accumulated += w * flat.gather(2, idx)

    return (accumulated / n_ops).reshape(B, C, H, W)


def _symmetrize_nn_copy(latent: torch.Tensor, sm: SymmetryMap) -> torch.Tensor:
    """Nearest-neighbour copy symmetrization (zero bilinear blur).

    Each pixel copies the value of its canonical representative — the
    symmetry-related pixel with the lexicographically smallest (u, v).
    All pixels in the same geometric orbit share the same canonical source,
    so the result is exactly symmetric up to ≤0.5-pixel NN rounding.
    """
    B, C, H, W = latent.shape
    device = latent.device
    source = sm.canonical_source.to(device)          # (H*W,)
    flat = latent.reshape(B, C, H * W)
    idx = source.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
    return flat.gather(2, idx).reshape(B, C, H, W)
