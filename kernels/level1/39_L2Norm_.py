import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def l2_norm_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L2 normalization to the input tensor using Helion.
    L2 normalization divides by the L2 norm along dimension 1.

    Args:
        x: Input tensor of shape (batch_size, dim)

    Returns:
        Output tensor with L2 normalization applied, same shape as input
    """
    batch_size, dim = x.size()
    out = torch.empty_like(x)

    for tile_b in hl.tile(batch_size):
        # Get batch slice
        batch_data = x[tile_b, :].to(torch.float32)  # Shape: [tile_b, dim]

        # Compute L2 norm along dimension 1
        squared = batch_data * batch_data
        sum_squared = torch.sum(squared, dim=-1, keepdim=True)  # [tile_b, 1]
        l2_norm = torch.sqrt(sum_squared)

        # Normalize by dividing by L2 norm
        normalized = batch_data / l2_norm

        out[tile_b, :] = normalized.to(out.dtype)

    return out


class Model:
    """
    Simple model that performs L2 normalization.
    """

    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, dim, *).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return x / torch.norm(x, p=2, dim=1, keepdim=True)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the L2 norm kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        dim: Feature dimension size
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float16)

    # Test L2 normalization
    run_example(l2_norm_kernel, pytorch_baseline, (x,))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 2048
    dim = 4096
    check(batch_size, dim)


if __name__ == "__main__":
    main()
