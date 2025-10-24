import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(
    static_shapes=True,
)
def leaky_relu_kernel(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Applies LeakyReLU activation to the input tensor using Helion.
    
    Args:
        x: Input tensor of any shape
        negative_slope: The negative slope of the activation function
    
    Returns:
        Output tensor with LeakyReLU applied, same shape as input
    """
    # Get the shape and flatten for processing
    original_shape = x.shape
    x_flat = x.view(-1)
    total_elements = x_flat.size(0)
    
    out_flat = torch.empty_like(x_flat)
    
    for tile_idx in hl.tile(total_elements):
        # Apply LeakyReLU: max(0, x) + negative_slope * min(0, x)
        input_slice = x_flat[tile_idx]
        output_slice = torch.nn.functional.leaky_relu(input_slice, negative_slope=negative_slope)
        out_flat[tile_idx] = output_slice
    
    return out_flat.view(original_shape)


class Model:
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(negative_slope=0.01)
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the LeakyReLU kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch dimension size
        dim: Feature dimension size
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float32)
    
    # Test LeakyReLU activation
    run_example(
        lambda x: leaky_relu_kernel(x, negative_slope=0.01),
        pytorch_baseline,
        (x,)
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 4096
    dim = 2048
    check(batch_size, dim)


if __name__ == "__main__":
    main()
