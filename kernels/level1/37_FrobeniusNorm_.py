import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


# TODO: Fix 'helion.exc.InternalError: AssertionError: pid already set'
@helion.kernel(
    static_shapes=True,
)
# Working config
# @helion.kernel(config=helion.Config(block_sizes=[1, 1, 1, 1], flatten_loops=[False], indexing='pointer', l2_groupings=[1, 1], load_eviction_policies=['', '', '', '', ''], loop_orders=[[0, 1], [0, 1]], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[], reduction_loops=[4096]), static_shapes=True)
def frobenius_norm_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Frobenius norm normalization to the input tensor using Helion.
    Computes the norm manually avoiding multi-dimensional reductions.
    
    Args:
        x: Input tensor of arbitrary shape
    
    Returns:
        Output tensor with Frobenius norm normalization applied, same shape as input
    """
    batch_size, features, height, width = x.size()
    
    # Reshape to merge spatial dimensions: (B, C, H*W)
    x_reshaped = x.view(batch_size, features, -1)
    out_reshaped = torch.empty_like(x_reshaped)
    
    # Compute sum of squares by accumulating element-wise
    sq = torch.zeros([], dtype=torch.float32, device=x.device)
    
    # First pass: compute sum of squares without multi-dim reduction
    for tile_b, tile_c in hl.tile([batch_size, features]):
        sum_squares = hl.zeros([], dtype=torch.float16)
        elements = x_reshaped[tile_b, tile_c, :].to(torch.float32)
        squared = elements * elements
        # Sum over the tile elements one dimension at a time
        tile_sum = torch.sum(squared, dim=-1)
        tile_sum = torch.sum(tile_sum, dim=-1)
        sum_squares = sum_squares + torch.sum(tile_sum)
        hl.atomic_add(sq, [], sum_squares)
    
    # Second pass: normalize elements
    for tile_b, tile_c in hl.tile([batch_size, features]):
        sm = hl.load(
            sq,
            [],
        )
        frobenius_norm = torch.sqrt(sm)
        # Compute Frobenius norm as sqrt of sum of squares
        elements_1 = x_reshaped[tile_b, tile_c, :].to(torch.float32)
        normalized = elements_1 / frobenius_norm
        out_reshaped[tile_b, tile_c, :] = normalized.to(out_reshaped.dtype)
    
    return out_reshaped.view(batch_size, features, height, width)


class Model:
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        norm = torch.norm(x, p='fro', dtype=x.dtype)
        return x / norm


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(x)


def check(batch_size: int, features: int, dim1: int, dim2: int) -> None:
    """
    Checks the correctness of the Frobenius norm kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch dimension size
        features: Number of feature channels
        dim1: Height dimension
        dim2: Width dimension
    """
    x = torch.randn([batch_size, features, dim1, dim2], device=DEVICE, dtype=torch.float16)
    
    # Test Frobenius norm normalization
    run_example(frobenius_norm_kernel, pytorch_baseline, (x,))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 50
    features = 64
    dim1 = 256
    dim2 = 256
    check(batch_size, features, dim1, dim2)


if __name__ == "__main__":
    main()
