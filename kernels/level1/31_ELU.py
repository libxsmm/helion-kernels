import torch
import torch.nn.functional as F
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(
    static_shapes=True,
)
def elu_kernel(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Applies ELU activation to the input tensor using Helion.
    ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
    
    Args:
        x: Input tensor of any shape
        alpha: The alpha parameter for the ELU function
    
    Returns:
        Output tensor with ELU applied, same shape as input
    """
    # Get the shape and flatten for processing
    original_shape = x.shape
    x_flat = x.view(-1)
    total_elements = x_flat.size(0)
    
    out_flat = torch.empty_like(x_flat)
    
    for tile_idx in hl.tile(total_elements):
        # Apply ELU
        input_slice = x_flat[tile_idx]
        output_slice = F.elu(input_slice, alpha=alpha)
        out_flat[tile_idx] = output_slice
    
    return out_flat.view(original_shape)


class Model:
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        return F.elu(x, alpha=self.alpha)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(alpha=1.0)
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the ELU kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch dimension size
        dim: Feature dimension size
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float32)
    
    # Test ELU activation
    run_example(
        lambda x: elu_kernel(x, alpha=1.0),
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
