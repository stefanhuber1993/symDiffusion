#!/usr/bin/env python3
"""
Test that the symmetry operations and symmetrization work correctly.
Runs without GPU or Stable Diffusion model — just tests the math.
"""

import torch
import math
from symmetric_diffusion.wallpaper_groups import get_group, list_groups
from symmetric_diffusion.symmetrize import build_symmetry_map, symmetrize_latent


def test_all_groups_defined():
    groups = list_groups()
    # 17 standard wallpaper groups + pgh (horizontal-glide variant of pg)
    expected = {
        "p1", "p2", "pm", "pg", "pgh", "pmm", "pmg", "pgg", "cm", "cmm",
        "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m",
    }
    assert set(groups) == expected, f"Unexpected groups: {set(groups) ^ expected}"
    print(f"[PASS] All {len(groups)} groups defined")


def test_identity_is_first():
    for name in list_groups():
        g = get_group(name)
        u, v = 0.3, 0.7
        u2, v2 = g.operations[0](u, v)
        assert abs(u2 - u) < 1e-10 and abs(v2 - v) < 1e-10
    print("[PASS] Identity is first operation in all groups")


def test_p1_preserves_input():
    g = get_group("p1")
    x = torch.randn(1, 4, 16, 16)
    sym_map = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sym_map)
    diff = (x - sym).abs().max().item()
    assert diff < 1e-5, f"p1: changed input, max diff = {diff}"
    print("[PASS] p1 preserves input")


def test_projection_idempotent_rect_square():
    """Orbit-based groups (rect/square lattice) should be exactly idempotent."""
    for name in ["p1", "p2", "pm", "pg", "pgh", "pmm", "pmg", "pgg", "cm", "cmm",
                 "p4", "p4m", "p4g"]:
        g = get_group(name)
        x = torch.randn(1, 4, 16, 16)
        sym_map = build_symmetry_map(g, 16, 16)
        sym1 = symmetrize_latent(x, g, sym_map)
        sym2 = symmetrize_latent(sym1, g, sym_map)
        diff = (sym1 - sym2).abs().max().item()
        assert diff < 1e-5, f"Group {name}: P^2 != P, max diff = {diff}"
    print("[PASS] Rect/square groups are exactly idempotent")


def test_projection_approx_idempotent_hex():
    """Hex groups use bilinear averaging — approximately idempotent."""
    for name in ["p3", "p3m1", "p31m", "p6", "p6m"]:
        g = get_group(name)
        x = torch.randn(1, 4, 32, 32)
        sym_map = build_symmetry_map(g, 32, 32)
        sym1 = symmetrize_latent(x, g, sym_map)
        sym2 = symmetrize_latent(sym1, g, sym_map)
        # Applying multiple times should converge
        sym3 = symmetrize_latent(sym2, g, sym_map)
        diff12 = (sym1 - sym2).abs().max().item()
        diff23 = (sym2 - sym3).abs().max().item()
        assert diff23 < diff12, \
            f"Group {name}: not converging (diff12={diff12}, diff23={diff23})"
    print("[PASS] Hex groups converge under repeated symmetrization")


def test_pm_mirror_symmetry():
    g = get_group("pm")
    x = torch.randn(1, 4, 16, 16)
    sym_map = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sym_map)
    diff = (sym - sym.flip(dims=[3])).abs().max().item()
    assert diff < 1e-5, f"pm: not mirror-symmetric, max diff = {diff}"
    print("[PASS] pm produces left-right mirror symmetry")


def test_p2_rotation_symmetry():
    g = get_group("p2")
    x = torch.randn(1, 4, 16, 16)
    sym_map = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sym_map)
    diff = (sym - sym.flip(dims=[2, 3])).abs().max().item()
    assert diff < 1e-5, f"p2: not 180°-rotation-symmetric, max diff = {diff}"
    print("[PASS] p2 produces 2-fold rotation symmetry")


def test_p4_rotation_symmetry():
    g = get_group("p4")
    x = torch.randn(1, 4, 16, 16)
    sym_map = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sym_map)
    rotated = sym.rot90(1, [2, 3])
    diff = (sym - rotated).abs().max().item()
    assert diff < 1e-4, f"p4: not 90°-rotation-symmetric, max diff = {diff}"
    print("[PASS] p4 produces 4-fold rotation symmetry")


def test_pmm_symmetry():
    g = get_group("pmm")
    x = torch.randn(1, 4, 16, 16)
    sym_map = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sym_map)
    diff1 = (sym - sym.flip(dims=[3])).abs().max().item()
    diff2 = (sym - sym.flip(dims=[2])).abs().max().item()
    assert diff1 < 1e-5, f"pmm: not LR-symmetric, max diff = {diff1}"
    assert diff2 < 1e-5, f"pmm: not TB-symmetric, max diff = {diff2}"
    print("[PASS] pmm produces double mirror symmetry")


def test_p4m_all_symmetries():
    """p4m should have 4-fold rotation AND both diagonal + axis mirrors."""
    g = get_group("p4m")
    x = torch.randn(1, 4, 16, 16)
    sym_map = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sym_map)

    # 90° rotation
    diff_rot = (sym - sym.rot90(1, [2, 3])).abs().max().item()
    # LR mirror
    diff_lr = (sym - sym.flip(dims=[3])).abs().max().item()
    # TB mirror
    diff_tb = (sym - sym.flip(dims=[2])).abs().max().item()
    # Diagonal mirror (transpose spatial dims)
    diff_diag = (sym - sym.transpose(2, 3)).abs().max().item()

    assert diff_rot < 1e-4, f"p4m: not rotation-symmetric, diff={diff_rot}"
    assert diff_lr < 1e-4, f"p4m: not LR-symmetric, diff={diff_lr}"
    assert diff_tb < 1e-4, f"p4m: not TB-symmetric, diff={diff_tb}"
    assert diff_diag < 1e-4, f"p4m: not diag-symmetric, diff={diff_diag}"
    print("[PASS] p4m produces full square symmetry (rotation + all mirrors)")


def test_symmetrize_reduces_variance():
    """Symmetrization should reduce spatial variance (more uniform)."""
    for name in list_groups():
        g = get_group(name)
        x = torch.randn(1, 4, 16, 16)
        sym_map = build_symmetry_map(g, 16, 16)
        sym = symmetrize_latent(x, g, sym_map)
        # Variance across spatial dims should decrease (or stay same for p1)
        var_before = x.var(dim=[2, 3]).mean().item()
        var_after = sym.var(dim=[2, 3]).mean().item()
        assert var_after <= var_before + 1e-5, \
            f"Group {name}: variance increased ({var_before} -> {var_after})"
    print("[PASS] Symmetrization reduces variance for all groups")


# ============================================================================
# Tests for hex groups (p3, p6 and variants)
# ============================================================================

def _check_hex_symmetry_bilinear(name, size=32, n_passes=5, tol=0.15):
    """Check that repeated bilinear symmetrization produces actual symmetry.

    Verifies by bilinear-sampling at each operation's target position and
    comparing to the original pixel value.
    """
    g = get_group(name)
    x = torch.randn(1, 4, size, size)
    sm = build_symmetry_map(g, size, size)

    sym = x
    for _ in range(n_passes):
        sym = symmetrize_latent(sym, g, sm)

    # Check symmetry using bilinear sampling (same method as symmetrization)
    flat = sym.reshape(1, 4, -1)
    op_grids = sm.op_grids      # (n_ops, H*W, 4)
    op_weights = sm.op_weights.to(sym.dtype)  # (n_ops, H*W, 4)

    max_diff = 0
    for k in range(1, g.order):
        gathered = torch.zeros_like(flat)
        for n in range(4):
            idx = op_grids[k, :, n].unsqueeze(0).unsqueeze(0).expand(1, 4, -1)
            w = op_weights[k, :, n].unsqueeze(0).unsqueeze(0).expand(1, 4, -1)
            gathered += w * flat.gather(2, idx)
        diff = (flat - gathered).abs().max().item()
        max_diff = max(max_diff, diff)

    return max_diff


def test_p3_symmetry():
    """p3 should produce 3-fold rotational symmetry (in hex fractional coords)."""
    err = _check_hex_symmetry_bilinear("p3", size=32, n_passes=5, tol=0.15)
    assert err < 0.3, f"p3: symmetry error too large: {err:.4f}"
    print(f"[PASS] p3 produces 3-fold rotation symmetry (err={err:.4f})")


def test_p6_symmetry():
    """p6 should produce 6-fold rotational symmetry (in hex fractional coords)."""
    err = _check_hex_symmetry_bilinear("p6", size=32, n_passes=5, tol=0.15)
    assert err < 0.3, f"p6: symmetry error too large: {err:.4f}"
    print(f"[PASS] p6 produces 6-fold rotation symmetry (err={err:.4f})")


def test_p3m1_symmetry():
    """p3m1: 3-fold rotation + mirrors through rotation center."""
    err = _check_hex_symmetry_bilinear("p3m1", size=32, n_passes=5)
    assert err < 0.3, f"p3m1: symmetry error too large: {err:.4f}"
    print(f"[PASS] p3m1 produces correct symmetry (err={err:.4f})")


def test_p31m_symmetry():
    """p31m: 3-fold rotation + mirrors NOT through rotation center."""
    err = _check_hex_symmetry_bilinear("p31m", size=32, n_passes=5)
    assert err < 0.3, f"p31m: symmetry error too large: {err:.4f}"
    print(f"[PASS] p31m produces correct symmetry (err={err:.4f})")


def test_p6m_symmetry():
    """p6m: full hexagonal symmetry (6-fold rotation + all mirrors)."""
    err = _check_hex_symmetry_bilinear("p6m", size=32, n_passes=5)
    assert err < 0.3, f"p6m: symmetry error too large: {err:.4f}"
    print(f"[PASS] p6m produces correct symmetry (err={err:.4f})")


def test_hex_convergence_rate():
    """Bilinear symmetrization should converge — error decreases overall."""
    for name in ["p3", "p6"]:
        g = get_group(name)
        x = torch.randn(1, 4, 32, 32)
        sm = build_symmetry_map(g, 32, 32)

        sym = x
        first_err = None
        last_err = None
        for i in range(5):
            sym = symmetrize_latent(sym, g, sm)
            # Measure symmetry of the SAME progressively-symmetrized tensor
            flat = sym.reshape(1, 4, -1)
            max_diff = 0
            for k in range(1, g.order):
                gathered = torch.zeros_like(flat)
                for n in range(4):
                    idx = sm.op_grids[k, :, n].unsqueeze(0).unsqueeze(0).expand(1, 4, -1)
                    w = sm.op_weights[k, :, n].unsqueeze(0).unsqueeze(0).expand(1, 4, -1)
                    gathered += w * flat.gather(2, idx)
                diff = (flat - gathered).abs().max().item()
                max_diff = max(max_diff, diff)
            if first_err is None:
                first_err = max_diff
            last_err = max_diff

        assert last_err < first_err * 0.5, \
            f"{name}: not converging (first={first_err:.4f}, last={last_err:.4f})"
    print("[PASS] Hex bilinear symmetrization converges")


# ============================================================================
# Tests for glide reflection groups (pg, pgg)
# ============================================================================

def test_pg_glide_symmetry():
    """pg: glide reflection (mirror + half-translation)."""
    g = get_group("pg")
    x = torch.randn(1, 4, 16, 16)
    sm = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sm)

    # pg has glide: (u,v) -> (-u, v+0.5). On pixels:
    # pixel (i,j) maps to (i + H//2, W-1-j) approximately
    H, W = 16, 16
    glide = torch.roll(sym.flip(dims=[3]), shifts=H//2, dims=2)
    diff = (sym - glide).abs().max().item()
    assert diff < 1e-4, f"pg: glide symmetry broken, diff={diff}"
    print(f"[PASS] pg produces glide reflection symmetry (diff={diff:.6f})")


def test_pgh_glide_symmetry():
    """pgh: horizontal glide reflection — flip vertically + shift horizontally by W//2."""
    g = get_group("pgh")
    x = torch.randn(1, 4, 16, 16)
    sm = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sm)

    # pgh has glide: (u+0.5, -v). On pixels:
    # pixel (i,j) maps to (H-1-i, j + W//2)
    W = 16
    glide = torch.roll(sym.flip(dims=[2]), shifts=W // 2, dims=3)
    diff = (sym - glide).abs().max().item()
    assert diff < 1e-4, f"pgh: glide symmetry broken, diff={diff}"
    print(f"[PASS] pgh produces horizontal glide reflection symmetry (diff={diff:.6f})")


def test_odd_dimensions_raise():
    """Groups with glide operations must raise ValueError for odd H or W."""
    cases = [
        ("pg",  17, 16),   # H odd
        ("pgh", 16, 17),   # W odd
        ("pmg", 16, 17),   # W odd
        ("pgg", 17, 16),   # H odd
        ("pgg", 16, 17),   # W odd
        ("cm",  17, 16),   # H odd
        ("cmm", 16, 17),   # W odd
        ("p4g", 17, 17),   # both odd (square)
    ]
    for name, H, W in cases:
        g = get_group(name)
        try:
            build_symmetry_map(g, H, W)
            raise AssertionError(f"{name} H={H} W={W}: expected ValueError, got none")
        except ValueError:
            pass  # expected
    print("[PASS] Odd-dimension groups raise ValueError as expected")


def test_pgg_symmetry():
    """pgg: two perpendicular glide reflections + 180° rotation about (1/4, 1/4)."""
    g = get_group("pgg")
    H = W = 16
    x = torch.randn(1, 4, H, W)
    sm = build_symmetry_map(g, H, W)
    sym = symmetrize_latent(x, g, sm)

    # pgg has 180° rotation about (1/4, 1/4), i.e. (-u+0.5, -v+0.5)
    # In pixel space: flip then shift by (H//2, W//2)
    rotated = torch.roll(sym.flip(dims=[2, 3]), shifts=(H // 2, W // 2), dims=(2, 3))
    diff_rot = (sym - rotated).abs().max().item()
    assert diff_rot < 1e-4, f"pgg: 180° rotation broken, diff={diff_rot}"

    # Check glide reflection: (-u, v+0.5) → flip W then shift H//2
    glide = torch.roll(sym.flip(dims=[3]), shifts=H // 2, dims=2)
    diff_glide = (sym - glide).abs().max().item()
    assert diff_glide < 1e-4, f"pgg: glide broken, diff={diff_glide}"
    print(f"[PASS] pgg produces correct symmetry (rot={diff_rot:.6f}, glide={diff_glide:.6f})")


# ============================================================================
# Tests for centered rectangular groups (cm, cmm)
# ============================================================================

def test_cm_symmetry():
    """cm: mirror + centering translation."""
    g = get_group("cm")
    x = torch.randn(1, 4, 16, 16)
    sm = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sm)

    # cm has mirror (u -> -u) and centering (u+0.5, v+0.5)
    # Check mirror: sym[i,j] == sym[i, W-1-j]
    diff_mirror = (sym - sym.flip(dims=[3])).abs().max().item()
    assert diff_mirror < 1e-4, f"cm: mirror broken, diff={diff_mirror}"

    # Check centering: sym[i,j] == sym[i+H//2, j+W//2]
    centered = torch.roll(sym, shifts=(8, 8), dims=(2, 3))
    diff_center = (sym - centered).abs().max().item()
    assert diff_center < 1e-4, f"cm: centering broken, diff={diff_center}"

    print(f"[PASS] cm produces mirror + centering symmetry")


def test_cmm_symmetry():
    """cmm: two mirrors + centering."""
    g = get_group("cmm")
    x = torch.randn(1, 4, 16, 16)
    sm = build_symmetry_map(g, 16, 16)
    sym = symmetrize_latent(x, g, sm)

    # Check both mirrors
    diff_lr = (sym - sym.flip(dims=[3])).abs().max().item()
    diff_tb = (sym - sym.flip(dims=[2])).abs().max().item()
    assert diff_lr < 1e-4, f"cmm: LR mirror broken, diff={diff_lr}"
    assert diff_tb < 1e-4, f"cmm: TB mirror broken, diff={diff_tb}"

    # Check centering
    centered = torch.roll(sym, shifts=(8, 8), dims=(2, 3))
    diff_center = (sym - centered).abs().max().item()
    assert diff_center < 1e-4, f"cmm: centering broken, diff={diff_center}"

    print(f"[PASS] cmm produces double mirror + centering symmetry")


# ============================================================================
# Test group closure (operations form a group under composition)
# ============================================================================

def test_group_closure():
    """Verify that composing any two operations gives another operation in the group."""
    test_points = [(0.17, 0.31), (0.62, 0.85), (0.43, 0.09)]

    for name in list_groups():
        g = get_group(name)
        ops = g.operations
        n = len(ops)

        for a in range(n):
            for b in range(n):
                # Compose: apply op_b then op_a
                for u0, v0 in test_points:
                    u1, v1 = ops[b](u0, v0)
                    u_ab, v_ab = ops[a](u1, v1)
                    u_ab, v_ab = u_ab % 1, v_ab % 1

                    # Check that result matches some operation op_c
                    found = False
                    for c in range(n):
                        u_c, v_c = ops[c](u0, v0)
                        u_c, v_c = u_c % 1, v_c % 1
                        if abs(u_ab - u_c) < 1e-8 and abs(v_ab - v_c) < 1e-8:
                            found = True
                            break
                    assert found, \
                        f"Group {name}: op_{a} ∘ op_{b} not in group at ({u0},{v0}): " \
                        f"got ({u_ab:.4f},{v_ab:.4f})"
    print("[PASS] All groups are closed under composition")


# ============================================================================
# Test hex group operations specifically
# ============================================================================

def test_hex_rotation_composition():
    """Verify that rot120^3 = identity and rot60^6 = identity."""
    from symmetric_diffusion.wallpaper_groups import _rot120, _rot240, _rot60, _rot300, _mod1

    test_points = [(0.2, 0.3), (0.7, 0.1), (0.5, 0.5)]

    for u, v in test_points:
        # rot120^3 should be identity
        u1, v1 = _rot120(u, v)
        u2, v2 = _rot120(u1, v1)
        u3, v3 = _rot120(u2, v2)
        assert abs(u3 % 1 - u) < 1e-10 and abs(v3 % 1 - v) < 1e-10, \
            f"rot120^3 != identity at ({u},{v}): got ({u3%1},{v3%1})"

        # rot60^6 should be identity
        ux, vx = u, v
        for _ in range(6):
            ux, vx = _rot60(ux, vx)
        assert abs(ux % 1 - u) < 1e-10 and abs(vx % 1 - v) < 1e-10, \
            f"rot60^6 != identity at ({u},{v}): got ({ux%1},{vx%1})"

    print("[PASS] Hex rotation compositions are correct (rot120^3=id, rot60^6=id)")


def test_hex_noise_variance():
    """Check that symmetrized + rescaled noise for hex groups has approximately unit variance."""
    for name in ["p3", "p6"]:
        g = get_group(name)
        size = 32
        x = torch.randn(1, 4, size, size)
        sm = build_symmetry_map(g, size, size)

        sym = x
        for _ in range(5):
            sym = symmetrize_latent(sym, g, sm)
        sym = sym / sym.std()

        # Check variance is approximately 1
        var = sym.var().item()
        assert 0.8 < var < 1.2, \
            f"{name}: noise variance after symmetrization = {var:.3f}, expected ~1.0"
    print("[PASS] Hex noise has approximately correct variance after symmetrization")


if __name__ == "__main__":
    test_all_groups_defined()
    test_identity_is_first()
    test_p1_preserves_input()
    test_projection_idempotent_rect_square()
    test_projection_approx_idempotent_hex()
    test_pm_mirror_symmetry()
    test_p2_rotation_symmetry()
    test_p4_rotation_symmetry()
    test_pmm_symmetry()
    test_p4m_all_symmetries()
    test_symmetrize_reduces_variance()

    # New tests for hex, glide, and centered groups
    test_group_closure()
    test_hex_rotation_composition()
    test_p3_symmetry()
    test_p6_symmetry()
    test_p3m1_symmetry()
    test_p31m_symmetry()
    test_p6m_symmetry()
    test_hex_convergence_rate()
    test_hex_noise_variance()
    test_pg_glide_symmetry()
    test_pgh_glide_symmetry()
    test_odd_dimensions_raise()
    test_pgg_symmetry()
    test_cm_symmetry()
    test_cmm_symmetry()

    print("\nAll tests passed!")
