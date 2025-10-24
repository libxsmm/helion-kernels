import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(static_shapes=True)
def masked_cumsum_kernel(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: hl.constexpr[int],
) -> torch.Tensor:
    """
    Performs masked cumulative sum along dimension 1 using Helion.
    Fixed implementation for 2D tensor masked cumsum over second dimension.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len]
        mask: Boolean mask tensor of same shape as x
        dim: Dimension to perform masked cumsum over (expected to be 1)
        
    Returns:
        Output tensor with masked cumulative sum applied along dimension 1
    """
    assert dim == 1, f"Kernel specialized for reduction dim 1 not {dim}"
    batch_size, seq_len = x.size()
    
    out = torch.empty([batch_size, seq_len], dtype=torch.float32, device=x.device)
    
    # Tile over batch dimension
    for tile_b in hl.tile(batch_size):
        # Initialize running sum
        running_sum = hl.zeros([tile_b], dtype=torch.float32)
        
        # Sequential masked cumulative sum over sequence dimension
        for i in range(seq_len):
            # Apply mask: only add to running sum if mask is True
            masked_value = torch.where(
                mask[tile_b, i],
                x[tile_b, i].to(torch.float32),
                torch.zeros_like(running_sum)
            )
            
            # Add masked value to running sum
            running_sum = running_sum + masked_value
            
            # Store cumulative sum
            out[tile_b, i] = running_sum
    
    return out


class Model:
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return torch.cumsum(x * mask, dim=self.dim, dtype=torch.float32)


def pytorch_baseline(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(dim=dim)
    return model.forward(x, mask)


def check(batch_size: int, input_shape: tuple, dim: int) -> None:
    """
    Checks the correctness of the masked cumsum kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch size
        input_shape: Shape of input tensor (excluding batch dimension)
        dim: Dimension to perform masked cumsum over
    """
    x = torch.randn([batch_size] + list(input_shape), device=DEVICE, dtype=torch.float16)
    mask = torch.randint(0, 2, x.shape, device=DEVICE).bool()  # Random boolean mask
    
    # Test masked cumulative sum
    run_example(
        lambda x, mask: masked_cumsum_kernel(x, mask, dim),
        lambda x, mask: pytorch_baseline(x, mask, dim),
        (x, mask)
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
