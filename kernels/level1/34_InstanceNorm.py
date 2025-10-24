import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def instancenorm_kernel(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Applies Instance Normalization to the input tensor using Helion.
    Simple implementation that processes spatial dimensions together.

    Args:
        x: Input tensor of shape (batch_size, num_features, height, width)
        weight: Scale parameter of shape (num_features,)
        bias: Shift parameter of shape (num_features,)
        eps: Small value for numerical stability

    Returns:
        Output tensor with Instance Normalization applied, same shape as input
    """
    batch_size, num_features, height, width = x.size()

    # Reshape to merge spatial dimensions: (B, C, H*W)
    x_reshaped = x.view(batch_size, num_features, -1)
    out_reshaped = torch.empty_like(x_reshaped)

    for tile_b, tile_c in hl.tile([batch_size, num_features]):
        # Get the flattened spatial data for this batch and channel
        spatial_flat = x_reshaped[tile_b, tile_c, :].to(
            torch.float32
        )  # Shape: [tile_b, tile_c, H*W]

        # Compute mean and variance across the spatial dimension
        var_val, mean_val = torch.var_mean(spatial_flat, dim=-1, keepdim=True)

        # Get weight and bias for this channel
        w = weight[tile_c]
        b = bias[tile_c]

        # Apply normalization
        normalized = (spatial_flat - mean_val) / torch.sqrt(var_val + eps)
        result = normalized * w[None, :, None] + b[None, :, None]

        out_reshaped[tile_b, tile_c, :] = result.to(out_reshaped.dtype)

    # Reshape back to original dimensions
    return out_reshaped.view(batch_size, num_features, height, width)


# Example multi-stage implementation - raises Helion error:
#   Error: NotImplementedError("multiple reduction dimensions")
# @helion.kernel(
#     static_shapes=True,
# )
def instancenorm_kernel_multi_stage(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Applies Instance Normalization to the input tensor using Helion.
    Instance normalization computes mean and variance per instance per channel.

    Args:
        x: Input tensor of shape (batch_size, num_features, height, width)
        weight: Scale parameter of shape (num_features,)
        bias: Shift parameter of shape (num_features,)
        eps: Small value for numerical stability

    Returns:
        Output tensor with Instance Normalization applied, same shape as input
    """
    batch_size, num_features, height, width = x.size()
    out = torch.empty_like(x)

    for tile_b, tile_c in hl.tile([batch_size, num_features]):
        # For each batch and channel, compute mean and variance across spatial dimensions
        instance_mean = hl.zeros([tile_b, tile_c], dtype=torch.float32)
        instance_var = hl.zeros([tile_b, tile_c], dtype=torch.float32)

        # First pass: compute mean
        for tile_h, tile_w in hl.tile([height, width]):
            spatial_slice = x[tile_b, tile_c, tile_h, tile_w]
            instance_mean = instance_mean + torch.sum(spatial_slice, dim=(-2, -1))

        instance_mean = instance_mean / (height * width)

        # Second pass: compute variance
        for tile_h, tile_w in hl.tile([height, width]):
            spatial_slice = x[tile_b, tile_c, tile_h, tile_w]
            centered = spatial_slice - instance_mean[:, :, None, None]
            instance_var = instance_var + torch.sum(centered * centered, dim=(-2, -1))

        instance_var = instance_var / (height * width)

        # Third pass: normalize and apply affine transformation
        for tile_h, tile_w in hl.tile([height, width]):
            spatial_slice = x[tile_b, tile_c, tile_h, tile_w]
            w = weight[tile_c]
            b = bias[tile_c]

            # Apply instance normalization: (x - mean) / sqrt(var + eps) * weight + bias
            normalized = (spatial_slice - instance_mean[:, :, None, None]) / torch.sqrt(
                instance_var[:, :, None, None] + eps
            )
            output_slice = normalized * w[None, :, None, None] + b[None, :, None, None]

            out[tile_b, tile_c, tile_h, tile_w] = output_slice

    return out


class Model:
    """
    Simple model that performs Instance Normalization.
    """

    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        self.inorm = nn.InstanceNorm2d(num_features=num_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(num_features=x.size(1))
    model.inorm = model.inorm.to(x.device)
    return model.forward(x)


def check(batch_size: int, features: int, dim1: int, dim2: int) -> None:
    """
    Checks the correctness of the InstanceNorm kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        features: Number of feature channels
        dim1: Height dimension
        dim2: Width dimension
    """
    x = torch.randn(
        [batch_size, features, dim1, dim2], device=DEVICE, dtype=torch.float16
    )

    # Create an InstanceNorm layer and extract its parameters
    inorm = nn.InstanceNorm2d(num_features=features, affine=True).to(DEVICE)

    # Test InstanceNorm with extracted parameters
    run_example(
        lambda x: instancenorm_kernel(x, inorm.weight, inorm.bias),
        lambda x: pytorch_baseline(x),
        (x,),
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 50
    features = 64
    dim1 = 512
    dim2 = 512
    check(batch_size, features, dim1, dim2)


if __name__ == "__main__":
    main()
