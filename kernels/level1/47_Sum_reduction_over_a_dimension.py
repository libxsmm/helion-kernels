import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


# TODO: Generalize kernel over dim
@helion.kernel(static_shapes=True)
def sum_reduction_kernel(
    x: torch.Tensor,
    dim: hl.constexpr[int],
) -> torch.Tensor:
    """
    Performs sum reduction over dimension 1 using Helion.
    Fixed implementation for 3D tensor reducing over middle dimension.

    Args:
        x: Input tensor of shape [batch_size, dim1, dim2]
        dim: Dimension to reduce over (expected to be 1)

    Returns:
        Output tensor after sum reduction with keepdim=True, shape [batch_size, 1, dim2]
    """
    assert dim == 1, f"Kernel specialized for reduction dim 1 not {dim}"
    batch_size, dim1, dim2 = x.size()

    out = torch.empty([batch_size, 1, dim2], dtype=x.dtype, device=x.device)

    # Tile over batch and last dimension, reduce over middle dimension
    for tile_b, tile_d2 in hl.tile([batch_size, dim2]):
        # Initialize accumulator
        sum_val = hl.zeros([tile_b, tile_d2], dtype=torch.float32)

        # Manual reduction over dimension 1
        for d1 in range(dim1):
            sum_val = sum_val + x[tile_b, d1, tile_d2].to(torch.float32)

        out[tile_b, 0, tile_d2] = sum_val.to(x.dtype)

    return out


class Model:
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return torch.sum(x, dim=self.dim, keepdim=True)


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
    Checks the correctness of the sum reduction kernel against PyTorch baseline.

    Args:
        batch_size: Batch size
        dim1: First dimension size
        dim2: Second dimension size
        reduce_dim: Dimension to reduce over
    """
    x = torch.randn([batch_size, dim1, dim2], device=DEVICE, dtype=torch.float16)

    # Test sum reduction
    run_example(
        lambda x: sum_reduction_kernel(x, reduce_dim),
        lambda x: pytorch_baseline(x, reduce_dim),
        (x,),
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
