import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def log_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Applies LogSoftmax activation to the input tensor using Helion.
    LogSoftmax is applied along dimension 1 (features).

    Args:
        x: Input tensor of shape (batch_size, num_features)

    Returns:
        Output tensor with LogSoftmax applied, same shape as input
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

        # Second pass: compute exp and sum for log-sum-exp
        for tile_feat in hl.tile(num_features):
            x_slice = x[tile_batch, tile_feat]
            exp_slice = torch.exp(x_slice - row_max[:, None])
            row_sum = row_sum + torch.sum(exp_slice, dim=-1)

        # Compute log_sum_exp = max + log(sum)
        log_sum_exp = row_max + torch.log(row_sum)

        # Third pass: compute log_softmax = x - log_sum_exp
        for tile_feat in hl.tile(num_features):
            x_slice = x[tile_batch, tile_feat]
            out[tile_batch, tile_feat] = x_slice - log_sum_exp[:, None]

    return out


class Model:
    """
    Simple model that performs a LogSoftmax activation.
    """

    def __init__(self, dim: int = 1):
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return torch.log_softmax(x, dim=self.dim)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(dim=1)
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the LogSoftmax kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        dim: Feature dimension size
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float32)

    # Test LogSoftmax activation
    run_example(log_softmax_kernel, pytorch_baseline, (x,))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 4096
    dim = 2048
    check(batch_size, dim)


if __name__ == "__main__":
    main()
