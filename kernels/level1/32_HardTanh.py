import torch
import torch.nn.functional as F
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(
    static_shapes=True,
)
def hardtanh_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Applies HardTanh activation to the input tensor using Helion.
    HardTanh(x) = max(min_val, min(max_val, x))
    
    Args:
        x: Input tensor of any shape
    
    Returns:
        Output tensor with HardTanh applied, same shape as input
    """
    # Get the shape and flatten for processing
    original_shape = x.shape
    x_flat = x.view(-1)
    total_elements = x_flat.size(0)
    
    out_flat = torch.empty_like(x_flat)
    
    for tile_idx in hl.tile(total_elements):
        # Apply HardTanh with default min_val=-1, max_val=1
        input_slice = x_flat[tile_idx]
        output_slice = F.hardtanh(input_slice, min_val=-1., max_val=1.)
        out_flat[tile_idx] = output_slice
    
    return out_flat.view(original_shape)


class Model:
    """
    Simple model that performs a HardTanh activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return F.hardtanh(x, min_val=-1., max_val=1.)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(x)


def check(batch_size: int, dim: int) -> None:
    """
    Checks the correctness of the HardTanh kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch dimension size
        dim: Feature dimension size
    """
    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float32)
    
    # Test HardTanh activation
    run_example(hardtanh_kernel, pytorch_baseline, (x,))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 4096
    dim = 2048
    check(batch_size, dim)


if __name__ == "__main__":
    main()
