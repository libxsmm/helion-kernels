import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(
    static_shapes=True,
)
def batchnorm_kernel(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Applies Batch Normalization to the input tensor using Helion.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, height, width)
        weight: Scale parameter of shape (num_features,)
        bias: Shift parameter of shape (num_features,)
        running_mean: Running mean of shape (num_features,)
        running_var: Running variance of shape (num_features,)
        eps: Small value for numerical stability
    
    Returns:
        Output tensor with Batch Normalization applied, same shape as input
    """
    batch_size, num_features, height, width = x.size()
    out = torch.empty_like(x)
    
    for tile_b, tile_c, tile_h, tile_w in hl.tile([batch_size, num_features, height, width]):
        # Get input slice
        input_slice = x[tile_b, tile_c, tile_h, tile_w]
        
        # Get normalization parameters for this channel
        mean = running_mean[tile_c]
        var = running_var[tile_c]
        w = weight[tile_c]
        b = bias[tile_c]
        
        # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
        normalized = (input_slice - mean[:, None, None]) / torch.sqrt(var[:, None, None] + eps)
        output_slice = normalized * w[:, None, None] + b[:, None, None]
        
        out[tile_b, tile_c, tile_h, tile_w] = output_slice
    
    return out


class Model:
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(num_features=x.size(1))
    model.bn = model.bn.to(x.device)
    model.bn.eval()  # Set to eval mode to use running stats
    return model.forward(x)


def check(batch_size: int, features: int, dim1: int, dim2: int) -> None:
    """
    Checks the correctness of the BatchNorm kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch dimension size
        features: Number of feature channels
        dim1: Height dimension
        dim2: Width dimension
    """
    x = torch.randn([batch_size, features, dim1, dim2], device=DEVICE, dtype=torch.float16)
    
    # Create a BatchNorm layer and extract its parameters
    bn = nn.BatchNorm2d(features).to(DEVICE).eval()
    
    # Test BatchNorm with extracted parameters
    run_example(
        lambda x: batchnorm_kernel(x, bn.weight, bn.bias, bn.running_mean, bn.running_var),
        lambda x: pytorch_baseline(x),
        (x,)
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 64
    features = 64
    dim1 = 512
    dim2 = 512
    check(batch_size, features, dim1, dim2)


if __name__ == "__main__":
    main()
