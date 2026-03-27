"""
Wallpaper group symmetry definitions for all 17 plane symmetry groups.

Each wallpaper group is defined by:
  - A lattice type (oblique, rectangular, centered rectangular, square, hexagonal)
  - A set of point group operations acting on fractional coordinates within the unit cell

Coordinates are fractional: (u, v) in [0, 1) x [0, 1), where u and v are
fractions of the lattice vectors a1 and a2.

Each symmetry operation is a function (u, v) -> (u', v') that maps fractional
coordinates to their symmetry-equivalent positions (mod 1).

The lattice vectors themselves are:
  - Oblique:     a1, a2 at arbitrary angle
  - Rectangular: a1 ⊥ a2, |a1| ≠ |a2|
  - Square:      a1 ⊥ a2, |a1| = |a2|
  - Hexagonal:   |a1| = |a2|, angle = 120° (or equivalently 60° rhombus)
  - Centered rectangular: |a1| = |a2|, arbitrary angle (rhombic)

For pixel-space implementation, we work with rectangular images where the
lattice is mapped to the image dimensions. For non-rectangular lattices
(oblique, hexagonal), a shearing transform is needed — this is handled
at the pipeline level.
"""

from dataclasses import dataclass, field
from typing import Callable, Tuple, List
import math


# Type alias: a symmetry operation maps (u, v) -> (u', v') in fractional coords
SymOp = Callable[[float, float], Tuple[float, float]]


@dataclass
class WallpaperGroup:
    """Definition of a wallpaper group.

    Attributes:
        name: International notation name (p1, pm, p4m, etc.)
        lattice: Lattice type (oblique, rectangular, centered_rectangular, square, hexagonal)
        operations: List of point group operations in fractional coordinates.
            Each operation is a function (u, v) -> (u', v') where coordinates
            are taken mod 1 to stay within the unit cell.
        lattice_angle: Angle between lattice vectors in degrees (90 for rectangular/square,
            120 for hexagonal, arbitrary for oblique).
        aspect_ratio: Ratio |a2|/|a1|. None means free (user can choose).
            1.0 for square and hexagonal lattices.
        requires_even_H: Group has half-cell glide translations along v (v ± 0.5),
            which only map pixel centres exactly when H is even.
        requires_even_W: Group has half-cell glide translations along u (u ± 0.5),
            which only map pixel centres exactly when W is even.
    """
    name: str
    lattice: str
    operations: List[SymOp] = field(default_factory=list)
    lattice_angle: float = 90.0
    aspect_ratio: float | None = None  # None = free, user picks
    requires_even_H: bool = False
    requires_even_W: bool = False

    @property
    def order(self) -> int:
        """Number of symmetry operations (order of the point group)."""
        return len(self.operations)


def _mod1(u: float, v: float) -> Tuple[float, float]:
    """Wrap fractional coordinates to [0, 1)."""
    return (u % 1.0, v % 1.0)


# ============================================================================
# Helper: common symmetry operations in fractional coordinates
# ============================================================================
# These are the building blocks. All coordinates are fractional (u, v) in [0,1).
# The translations (like +0.5) appear because some groups have glide reflections
# or screw axes that combine point operations with fractional translations.

def _identity(u, v):
    return (u, v)

def _rot180(u, v):
    """180° rotation about origin."""
    return _mod1(-u, -v)

def _rot90(u, v):
    """90° counterclockwise rotation (square lattice)."""
    return _mod1(-v, u)

def _rot270(u, v):
    """270° counterclockwise rotation (square lattice)."""
    return _mod1(v, -u)

def _rot60(u, v):
    """60° rotation (hexagonal lattice, axial coordinates).

    In a hexagonal lattice with basis vectors at 120°, a 60° rotation
    in fractional coordinates is: (u, v) -> (u - v, u).
    """
    return _mod1(u - v, u)

def _rot120(u, v):
    """120° rotation (hexagonal lattice)."""
    return _mod1(-v, u - v)

def _rot240(u, v):
    """240° rotation (hexagonal lattice)."""
    return _mod1(v - u, -u)

def _rot300(u, v):
    """300° rotation (hexagonal lattice)."""
    return _mod1(v, v - u)

def _mirror_x(u, v):
    """Mirror across the v-axis (horizontal mirror line): u -> -u."""
    return _mod1(-u, v)

def _mirror_y(u, v):
    """Mirror across the u-axis (vertical mirror line): v -> -v."""
    return _mod1(u, -v)

def _mirror_diag(u, v):
    """Mirror across the diagonal u=v."""
    return _mod1(v, u)

def _mirror_antidiag(u, v):
    """Mirror across the anti-diagonal u=-v."""
    return _mod1(-v, -u)

def _glide_x(u, v):
    """Glide reflection: mirror u -> -u, then translate v by 1/2."""
    return _mod1(-u, v + 0.5)

def _glide_y(u, v):
    """Glide reflection: mirror v -> -v, then translate u by 1/2."""
    return _mod1(u + 0.5, -v)


# ============================================================================
# Hexagonal lattice mirrors (in fractional coordinates with 120° angle)
# ============================================================================
# For hexagonal lattice, mirrors are along the lattice directions and their
# bisectors. In fractional coords (u,v) with a1, a2 at 120°:

def _hex_mirror_1(u, v):
    """Mirror that fixes a1 direction: (u,v) -> (u-v, -v)."""
    return _mod1(u - v, -v)

def _hex_mirror_2(u, v):
    """Mirror that fixes a2 direction: (u,v) -> (-u, v-u)."""
    return _mod1(-u, v - u)

def _hex_mirror_3(u, v):
    """Mirror that fixes a1+a2 direction: (u,v) -> (v, u)."""
    return _mod1(v, u)

def _hex_mirror_4(u, v):
    """Mirror perpendicular to a1: (u,v) -> (-u+v, v)."""
    return _mod1(v - u, v)

def _hex_mirror_5(u, v):
    """Mirror perpendicular to a2: (u,v) -> (u, u-v)."""
    return _mod1(u, u - v)

def _hex_mirror_6(u, v):
    """Mirror perpendicular to a1+a2: (u,v) -> (-v, -u)."""
    return _mod1(-v, -u)


# ============================================================================
# All 17 wallpaper groups
# ============================================================================

def _build_groups() -> dict[str, WallpaperGroup]:
    """Build definitions for all 17 wallpaper groups."""
    groups = {}

    # --- Oblique lattice (parallelogram) ---

    # p1: No symmetry beyond translation
    groups["p1"] = WallpaperGroup(
        name="p1",
        lattice="oblique",
        operations=[_identity],
        lattice_angle=90.0,  # default to rectangular for simplicity
    )

    # p2: 180° rotation
    groups["p2"] = WallpaperGroup(
        name="p2",
        lattice="oblique",
        operations=[_identity, _rot180],
        lattice_angle=90.0,
    )

    # --- Rectangular lattice ---

    # pm: One mirror axis (along v, reflecting u)
    groups["pm"] = WallpaperGroup(
        name="pm",
        lattice="rectangular",
        operations=[_identity, _mirror_x],
    )

    # pg: glide axis vertical — mirror flips u, translates v by 1/2
    groups["pg"] = WallpaperGroup(
        name="pg",
        lattice="rectangular",
        operations=[_identity, _glide_x],
        requires_even_H=True,
    )

    # pgh: glide axis horizontal — mirror flips v, translates u by 1/2
    # Same abstract group as pg; different orientation relative to the image axes.
    groups["pgh"] = WallpaperGroup(
        name="pgh",
        lattice="rectangular",
        operations=[_identity, _glide_y],
        requires_even_W=True,
    )

    # pmm: Two perpendicular mirror axes
    groups["pmm"] = WallpaperGroup(
        name="pmm",
        lattice="rectangular",
        operations=[_identity, _mirror_x, _mirror_y, _rot180],
    )

    # pmg: One mirror + one glide perpendicular to it
    groups["pmg"] = WallpaperGroup(
        name="pmg",
        lattice="rectangular",
        operations=[_identity, _mirror_x, _glide_y,
                    lambda u, v: _mod1(-u + 0.5, -v)],  # rot180 + (0.5, 0)
        requires_even_W=True,
    )

    # pgg: Two perpendicular glide reflections
    # The 180° rotation center is at (1/4, 1/4), NOT the origin.
    # It's generated by composing the two glide reflections.
    groups["pgg"] = WallpaperGroup(
        name="pgg",
        lattice="rectangular",
        operations=[_identity,
                    lambda u, v: _mod1(-u + 0.5, -v + 0.5),  # 180° about (1/4, 1/4)
                    _glide_x,                                  # (-u, v+0.5)
                    lambda u, v: _mod1(u + 0.5, -v)],          # glide along u-axis
        requires_even_H=True,
        requires_even_W=True,
    )

    # --- Centered rectangular (rhombic) lattice ---

    # cm: Mirror + centering (glide from the centering)
    groups["cm"] = WallpaperGroup(
        name="cm",
        lattice="centered_rectangular",
        operations=[_identity, _mirror_x,
                    lambda u, v: _mod1(u + 0.5, v + 0.5),  # centering translation
                    lambda u, v: _mod1(-u + 0.5, v + 0.5)],  # mirror + centering
        requires_even_H=True,
        requires_even_W=True,
    )

    # cmm: Two mirrors + centering
    groups["cmm"] = WallpaperGroup(
        name="cmm",
        lattice="centered_rectangular",
        operations=[
            _identity, _mirror_x, _mirror_y, _rot180,
            lambda u, v: _mod1(u + 0.5, v + 0.5),
            lambda u, v: _mod1(-u + 0.5, v + 0.5),
            lambda u, v: _mod1(u + 0.5, -v + 0.5),
            lambda u, v: _mod1(-u + 0.5, -v + 0.5),
        ],
        requires_even_H=True,
        requires_even_W=True,
    )

    # --- Square lattice ---

    # p4: 4-fold rotation
    groups["p4"] = WallpaperGroup(
        name="p4",
        lattice="square",
        operations=[_identity, _rot90, _rot180, _rot270],
        aspect_ratio=1.0,
    )

    # p4m: 4-fold rotation + mirrors along axes and diagonals
    groups["p4m"] = WallpaperGroup(
        name="p4m",
        lattice="square",
        operations=[
            _identity, _rot90, _rot180, _rot270,
            _mirror_x, _mirror_y, _mirror_diag, _mirror_antidiag,
        ],
        aspect_ratio=1.0,
    )

    # p4g: 4-fold rotation + glide reflections
    groups["p4g"] = WallpaperGroup(
        name="p4g",
        lattice="square",
        operations=[
            _identity, _rot90, _rot180, _rot270,
            lambda u, v: _mod1(-u + 0.5, v + 0.5),   # glide-mirror x
            lambda u, v: _mod1(u + 0.5, -v + 0.5),   # glide-mirror y
            lambda u, v: _mod1(v + 0.5, u + 0.5),     # glide-mirror diag
            lambda u, v: _mod1(-v + 0.5, -u + 0.5),   # glide-mirror antidiag
        ],
        aspect_ratio=1.0,
        requires_even_H=True,
        requires_even_W=True,
    )

    # --- Hexagonal lattice ---

    # p3: 3-fold rotation
    groups["p3"] = WallpaperGroup(
        name="p3",
        lattice="hexagonal",
        operations=[_identity, _rot120, _rot240],
        lattice_angle=120.0,
        aspect_ratio=1.0,
    )

    # p3m1: 3-fold rotation + mirrors through rotation center
    groups["p3m1"] = WallpaperGroup(
        name="p3m1",
        lattice="hexagonal",
        operations=[
            _identity, _rot120, _rot240,
            _hex_mirror_3, _hex_mirror_1, _hex_mirror_2,
        ],
        lattice_angle=120.0,
        aspect_ratio=1.0,
    )

    # p31m: 3-fold rotation + mirrors NOT through rotation center
    groups["p31m"] = WallpaperGroup(
        name="p31m",
        lattice="hexagonal",
        operations=[
            _identity, _rot120, _rot240,
            _hex_mirror_4, _hex_mirror_5, _hex_mirror_6,
        ],
        lattice_angle=120.0,
        aspect_ratio=1.0,
    )

    # p6: 6-fold rotation
    groups["p6"] = WallpaperGroup(
        name="p6",
        lattice="hexagonal",
        operations=[_identity, _rot60, _rot120, _rot180, _rot240, _rot300],
        lattice_angle=120.0,
        aspect_ratio=1.0,
    )

    # p6m: 6-fold rotation + all mirrors (full hexagonal symmetry)
    groups["p6m"] = WallpaperGroup(
        name="p6m",
        lattice="hexagonal",
        operations=[
            _identity, _rot60, _rot120, _rot180, _rot240, _rot300,
            _hex_mirror_1, _hex_mirror_2, _hex_mirror_3,
            _hex_mirror_4, _hex_mirror_5, _hex_mirror_6,
        ],
        lattice_angle=120.0,
        aspect_ratio=1.0,
    )

    return groups


_GROUPS = _build_groups()


def get_group(name: str) -> WallpaperGroup:
    """Get a wallpaper group by its international notation name.

    Args:
        name: One of the 17 wallpaper group names:
            p1, p2, pm, pg, pmm, pmg, pgg, cm, cmm,
            p4, p4m, p4g, p3, p3m1, p31m, p6, p6m

    Returns:
        WallpaperGroup instance with all symmetry operations defined.
    """
    if name not in _GROUPS:
        available = ", ".join(sorted(_GROUPS.keys()))
        raise ValueError(f"Unknown wallpaper group '{name}'. Available: {available}")
    return _GROUPS[name]


def list_groups() -> list[str]:
    """Return names of all 17 wallpaper groups."""
    return sorted(_GROUPS.keys())
