import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(static_shapes=True)
def cumsum_exclusive_kernel(
    x: torch.Tensor,
    dim: hl.constexpr[int],
) -> torch.Tensor:
    """
    Performs exclusive cumulative sum along dimension 1 using Helion.
    Fixed implementation for 2D tensor exclusive cumsum over second dimension.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len]
        dim: Dimension to perform exclusive cumsum over (expected to be 1)
        
    Returns:
        Output tensor with exclusive cumulative sum applied along dimension 1
    """
    assert dim == 1, f"Kernel specialized for reduction dim 1 not {dim}"
    batch_size, seq_len = x.size()
    
    out = torch.empty([batch_size-1, seq_len+1], dtype=torch.float32, device=x.device)
    
    # Tile over batch dimension
    for tile_b in hl.tile(batch_size-1):
        # Initialize running sum
        running_sum = hl.zeros([tile_b], dtype=torch.float32)
        
        # Sequential exclusive cumulative sum over sequence dimension
        for i in range(seq_len+1):
            # Store current running sum (exclusive - doesn't include current element)
            out[tile_b, i] = running_sum
            
            # Add current element to running sum for next iteration
            running_sum = running_sum + x[tile_b, i].to(torch.float32)
    
    return out


class Model:
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        exclusive_cumsum = torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), x), dim=self.dim)[:-1]
        return torch.cumsum(exclusive_cumsum, dim=self.dim, dtype=torch.float32)


def pytorch_baseline(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(dim=dim)
    return model.forward(x)


def check(batch_size: int, input_shape: tuple, dim: int) -> None:
    """
    Checks the correctness of the exclusive cumsum kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch size
        input_shape: Shape of input tensor (excluding batch dimension)
        dim: Dimension to perform exclusive cumsum over
    """
    x = torch.randn([batch_size] + list(input_shape), device=DEVICE, dtype=torch.float16)

    # Test exclusive cumulative sum
    run_example(
        lambda x: cumsum_exclusive_kernel(x, dim),
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
