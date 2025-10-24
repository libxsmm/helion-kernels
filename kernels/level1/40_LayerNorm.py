import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(
    static_shapes=True,
)
def layernorm_kernel(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Applies Layer Normalization to the input tensor using Helion.
    Manual implementation computing mean and variance step by step.
    
    Args:
        x: Input tensor of shape (batch_size, features, dim1, dim2)
        weight: Scale parameter of shape (features, dim1, dim2)
        bias: Shift parameter of shape (features, dim1, dim2)
        eps: Small value for numerical stability
    
    Returns:
        Output tensor with Layer Normalization applied, same shape as input
    """
    batch_size, features, dim1, dim2 = x.size()

    # Pre-flatten the tensor outside the loop
    x_flat = x.view(batch_size, -1)
    weight_flat = weight.view(-1)
    bias_flat = bias.view(-1)

    out_flat = torch.empty_like(x_flat)
    
    for tile_b in hl.tile(batch_size):
        # Get flattened batch data: [tile_b, features * dim1 * dim2]
        batch_data = x_flat[tile_b, :].to(torch.float32)
        
        # Compute mean over features and spatial
        mean_val = torch.mean(batch_data, dim=-1, keepdim=True)  # [tile_b, 1]
        
        # Compute variance over features and spatial
        centered = batch_data - mean_val
        squared_diff = centered * centered
        
        var_val = torch.mean(squared_diff, dim=-1, keepdim=True)  # [tile_b, 1]
        
        # Normalize
        normalized = centered / torch.sqrt(var_val + eps)
        
        # Apply affine transformation
        result = normalized * weight_flat[None, :] + bias_flat[None, :]
        
        out_flat[tile_b, :] = result.to(out_flat.dtype)
    
    return out_flat.view(batch_size, features, dim1, dim2)


class Model:
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple, dtype: torch.dtype = None):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    normalized_shape = x.shape[1:]  # All dimensions except batch
    model = Model(normalized_shape=normalized_shape, dtype=x.dtype)
    model.ln = model.ln.to(x.device)
    return model.forward(x)


def check(batch_size: int, features: int, dim1: int, dim2: int) -> None:
    """
    Checks the correctness of the LayerNorm kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch dimension size
        features: Number of feature channels
        dim1: First spatial dimension
        dim2: Second spatial dimension
    """
    x = torch.randn([batch_size, features, dim1, dim2], device=DEVICE, dtype=torch.float16)
    
    # Create a LayerNorm layer and extract its parameters
    normalized_shape = (features, dim1, dim2)
    ln = nn.LayerNorm(normalized_shape=normalized_shape, dtype=x.dtype).to(DEVICE)
    
    # Test LayerNorm with extracted parameters
    run_example(
        lambda x: layernorm_kernel(x, ln.weight, ln.bias),
        lambda x: pytorch_baseline(x),
        (x,)
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    check(batch_size, features, dim1, dim2)


if __name__ == "__main__":
    main()
