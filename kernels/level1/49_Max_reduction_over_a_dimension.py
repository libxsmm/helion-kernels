import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


# TODO: Generalize kernel over dim
@helion.kernel(static_shapes=True)
def max_reduction_kernel(
    x: torch.Tensor,
    dim: hl.constexpr[int],
) -> torch.Tensor:
    """
    Performs max reduction over dimension 1 using Helion.
    Fixed implementation for 3D tensor reducing over middle dimension.
    
    Args:
        x: Input tensor of shape [batch_size, dim1, dim2]
        dim: Dimension to reduce over (expected to be 1)
        
    Returns:
        Output tensor after max reduction, shape [batch_size, dim2]
    """
    assert dim == 1, f"Kernel specialized for reduction dim 1 not {dim}"
    batch_size, dim1, dim2 = x.size()
    
    out = torch.empty([batch_size, dim2], dtype=x.dtype, device=x.device)
    
    # Tile over batch and last dimension, reduce over middle dimension
    for tile_b, tile_d2 in hl.tile([batch_size, dim2]):
        # Initialize with negative infinity
        max_val = hl.full([tile_b, tile_d2], float('-inf'), dtype=x.dtype)
        
        # Manual reduction over dimension 1
        for d1 in range(dim1):
            current_val = x[tile_b, d1, tile_d2]
            max_val = torch.maximum(max_val, current_val)
        
        out[tile_b, tile_d2] = max_val
    
    return out


class Model:
    """
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return torch.max(x, dim=self.dim)[0]


def pytorch_baseline(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(dim=dim)
    return model.forward(x)


def check(
    batch_size: int,
    dim1: int,
    dim2: int,
    reduce_dim: int,
) -> None:
    """
    Checks the correctness of the max reduction kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch size
        dim1: First dimension size
        dim2: Second dimension size
        reduce_dim: Dimension to reduce over
    """
    x = torch.randn([batch_size, dim1, dim2], device=DEVICE, dtype=torch.float16)
    
    # Test max reduction
    run_example(
        lambda x: max_reduction_kernel(x, reduce_dim),
        lambda x: pytorch_baseline(x, reduce_dim),
        (x,)
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    reduce_dim = 1
    
    check(batch_size, dim1, dim2, reduce_dim)


if __name__ == "__main__":
    main()
