import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Softmax activation to the input tensor using Helion.
    Softmax is applied along dimension 1 (features).

    Args:
        x: Input tensor of shape (batch_size, num_features)

    Returns:
        Output tensor with Softmax applied, same shape as input
    """
    batch_size, num_features = x.size()
    out = torch.empty_like(x)

    for tile_batch in hl.tile(batch_size):
        # Process each batch element
        row_max = hl.full([tile_batch], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_batch], dtype=torch.float32)

        # First pass: find max for numerical stability
        for tile_feat in hl.tile(num_features):
            x_slice = x[tile_batch, tile_feat]
            row_max = torch.maximum(row_max, torch.amax(x_slice, dim=-1))

        # Second pass: compute exp and sum
        for tile_feat in hl.tile(num_features):
            x_slice = x[tile_batch, tile_feat]
            exp_slice = torch.exp(x_slice - row_max[:, None])
            out[tile_batch, tile_feat] = exp_slice
            row_sum = row_sum + torch.sum(exp_slice, dim=-1)

        # Third pass: normalize
        for tile_feat in hl.tile(num_features):
            out[tile_batch, tile_feat] = out[tile_batch, tile_feat] / row_sum[:, None]

    return out


class Model:
    """
    Simple model that performs a Softmax activation.
    """

    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the Softmax kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        dim: Feature dimension size
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float32)

    # Test Softmax activation
    run_example(softmax_kernel, pytorch_baseline, (x,))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 4096
    dim = 2048
    check(batch_size, dim)


if __name__ == "__main__":
    main()
