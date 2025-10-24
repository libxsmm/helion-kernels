import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(
    static_shapes=True,
)
def l1_norm_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L1 normalization to the input tensor using Helion.
    L1 normalization divides by the mean of absolute values along dimension 1.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
    
    Returns:
        Output tensor with L1 normalization applied, same shape as input
    """
    batch_size, dim = x.size()
    out = torch.empty_like(x)
    
    for tile_b in hl.tile(batch_size):
        # Get batch slice
        batch_data = x[tile_b, :].to(torch.float32)  # Shape: [tile_b, dim]
        
        # Compute mean of absolute values along dimension 1
        abs_values = torch.abs(batch_data)
        mean_abs = torch.mean(abs_values, dim=-1, keepdim=True)  # [tile_b, 1]
        
        # Normalize by dividing by mean absolute value
        normalized = batch_data / mean_abs
        
        out[tile_b, :] = normalized.to(out.dtype)
    
    return out


class Model:
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return x / torch.mean(torch.abs(x), dim=1, keepdim=True)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the L1 norm kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch dimension size
        dim: Feature dimension size
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float16)
    
    # Test L1 normalization
    run_example(l1_norm_kernel, pytorch_baseline, (x,))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 2048
    dim = 4096
    check(batch_size, dim)


if __name__ == "__main__":
    main()
