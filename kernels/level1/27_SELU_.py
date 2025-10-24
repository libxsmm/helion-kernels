import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def selu_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Applies SELU activation to the input tensor using Helion.
    SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    where alpha ≈ 1.6733 and scale ≈ 1.0507

    Args:
        x: Input tensor of any shape

    Returns:
        Output tensor with SELU applied, same shape as input
    """
    # Get the shape and flatten for processing
    original_shape = x.shape
    x_flat = x.view(-1)
    total_elements = x_flat.size(0)

    out_flat = torch.empty_like(x_flat)

    for tile_idx in hl.tile(total_elements):
        # Apply SELU
        input_slice = x_flat[tile_idx]
        output_slice = torch.selu(input_slice)
        out_flat[tile_idx] = output_slice

    return out_flat.view(original_shape)


class Model:
    """
    Simple model that performs a SELU activation.
    """

    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return torch.selu(x)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the SELU kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        dim: Feature dimension size
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float32)

    # Test SELU activation
    run_example(selu_kernel, pytorch_baseline, (x,))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 4096
    dim = 2048
    check(batch_size, dim)


if __name__ == "__main__":
    main()
