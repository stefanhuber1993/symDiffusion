"""
Monkey-patch Conv2d layers to use circular (periodic) padding.

This makes the generated image seamlessly tileable: the left edge wraps
to the right, the top wraps to the bottom. Combined with wallpaper group
symmetry, this produces Escher-style periodic tessellations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_conv2d_circular(model: nn.Module) -> None:
    """Recursively patch all Conv2d layers in a model to use circular padding.

    This replaces the padding in each Conv2d with manual circular (wrap-around)
    padding via F.pad, ensuring that convolutions see periodic boundary conditions.

    Args:
        model: Any nn.Module (typically the UNet from Stable Diffusion).
            Modified in-place.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.padding != (0, 0):
            # Replace with our circular-padding wrapper
            setattr(model, name, _CircularConv2d(module))
        else:
            # Recurse into child modules
            patch_conv2d_circular(module)


class _CircularConv2d(nn.Module):
    """Wrapper around a Conv2d that replaces its padding with circular padding."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.padding = conv.padding
        # Remove the original padding so we apply it ourselves
        self.conv.padding = (0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad with mode='circular' for 2D: (left, right, top, bottom)
        pad_h, pad_w = self.padding
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="circular")
        return self.conv(x)
