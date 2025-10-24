import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(static_shapes=True)
def cumprod_kernel(
    x: torch.Tensor,
    dim: hl.constexpr[int],
) -> torch.Tensor:
    """
    Performs cumulative product along dimension 1 using Helion.
    Fixed implementation for 2D tensor cumprod over second dimension.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len]
        dim: Dimension to perform cumprod over (expected to be 1)
        
    Returns:
        Output tensor with cumulative product applied along dimension 1
    """
    assert dim == 1, f"Kernel specialized for reduction dim 1 not {dim}"
    batch_size, seq_len = x.size()
    
    out = torch.empty([batch_size, seq_len], dtype=torch.float32, device=x.device)
    
    # Tile over batch dimension
    for tile_b in hl.tile(batch_size):
        # Initialize running product
        running_prod = hl.full([tile_b], 1.0, dtype=torch.float32)
        
        # Sequential cumulative product over sequence dimension
        for i in range(seq_len):
            # Multiply current element to running product
            running_prod = running_prod * x[tile_b, i].to(torch.float32)
            
            # Store cumulative product
            out[tile_b, i] = running_prod
    
    return out


class Model:
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return torch.cumprod(x, dim=self.dim, dtype=torch.float32)


def pytorch_baseline(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(dim=dim)
    return model.forward(x)


def check(batch_size: int, input_shape: tuple, dim: int) -> None:
    """
    Checks the correctness of the cumprod kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch size
        input_shape: Shape of input tensor (excluding batch dimension)
        dim: Dimension to perform cumprod over
    """
    x = torch.randn([batch_size] + list(input_shape), device=DEVICE, dtype=torch.float16)
    
    # Test cumulative product
    run_example(
        lambda x: cumprod_kernel(x, dim),
        lambda x: pytorch_baseline(x, dim),
        (x,)
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 32768
    input_shape = (32768,)
    dim = 1
    
    check(batch_size, input_shape, dim)


if __name__ == "__main__":
    main()
