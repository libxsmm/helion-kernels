import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
import math


@helion.kernel(static_shapes=True)
def mingpt_newgelu_kernel(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Implementation of the GELU activation function using Helion.
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415

    Args:
        x: Input tensor of any shape

    Returns:
        Output tensor with GELU activation applied element-wise
    """
    # Get input shape and create output tensor
    out = torch.empty_like(x)

    # Constants for GELU computation
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    coeff = 0.044715

    # Tile over all dimensions of the input
    for tile_indices in hl.tile(x.shape):
        # Apply GELU formula: 0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        x_val = x[tile_indices].to(torch.float32)

        # Compute x^3
        x_cubed = torch.pow(x_val, 3.0)

        # Compute the argument to tanh
        tanh_arg = sqrt_2_over_pi * (x_val + coeff * x_cubed)

        # Apply tanh
        tanh_val = torch.tanh(tanh_arg)

        # Final GELU computation
        result = 0.5 * x_val * (1.0 + tanh_val)

        out[tile_indices] = result.to(out.dtype)

    return out


class Model:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        pass

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the MinGPT NewGELU kernel against PyTorch baseline.

    Args:
        batch_size: Batch size
        dim: Feature dimension
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float16)

    # Test MinGPT NewGELU
    run_example(lambda x: mingpt_newgelu_kernel(x), lambda x: pytorch_baseline(x), (x,))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 8192
    dim = 8192

    check(batch_size, dim)


if __name__ == "__main__":
    main()
