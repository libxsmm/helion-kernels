import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def groupnorm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: hl.constexpr,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Applies Group Normalization to the input tensor using Helion.
    Implementation with sequential single-dimension reductions.

    Args:
        x: Input tensor of shape (batch_size, num_features, height, width)
        weight: Scale parameter of shape (num_features,)
        bias: Shift parameter of shape (num_features,)
        num_groups: Number of groups to divide channels into
        eps: Small value for numerical stability

    Returns:
        Output tensor with Group Normalization applied, same shape as input
    """
    batch_size, num_features, height, width = x.size()
    channels_per_group = num_features // num_groups

    # Reshape to merge spatial dimensions: (B, C, H*W)
    x_reshaped = x.view(batch_size, num_features, -1)
    out_reshaped = torch.empty_like(x_reshaped)

    for tile_b, tile_c in hl.tile(
        [batch_size, num_features], block_size=[None, channels_per_group]
    ):
        # Extract group data using slicing
        group_data = x_reshaped[tile_b, tile_c, :].to(
            torch.float32
        )  # Shape: [tile_b, channels_per_group, H*W]

        # Compute mean by reducing one dimension at a time
        # First reduce over height and width
        mean_w = torch.mean(
            group_data, dim=-1, keepdim=True
        )  # [tile_b, channels_per_group, 1]
        # Then reduce over channels
        mean_val = torch.mean(mean_w, dim=-2, keepdim=True)  # [tile_b, 1, 1]

        # Compute variance by reducing one dimension at a time
        centered = group_data - mean_val
        squared_diff = centered * centered

        # First reduce over height and width
        var_w = torch.mean(
            squared_diff, dim=-1, keepdim=True
        )  # [tile_b, channels_per_group, 1]
        # Then reduce over channels
        var_val = torch.mean(var_w, dim=-2, keepdim=True)  # [tile_b, 1, 1]

        # Normalize
        normalized = centered / torch.sqrt(var_val + eps)

        # Get weight and bias for this group
        group_weight = weight[tile_c]
        group_bias = bias[tile_c]

        # Apply affine transformation
        result = normalized * group_weight[None, :, None] + group_bias[None, :, None]

        # Store result
        out_reshaped[tile_b, tile_c, :] = result.to(out_reshaped.dtype)

    return out_reshaped.view(batch_size, num_features, height, width)


class Model:
    """
    Simple model that performs Group Normalization.
    """

    def __init__(self, num_features: int, num_groups: int, dtype: torch.dtype = None):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        self.gn = nn.GroupNorm(
            num_groups=num_groups, num_channels=num_features, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.gn(x)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    num_groups = 8
    model = Model(num_features=x.size(1), num_groups=num_groups, dtype=x.dtype)
    model.gn = model.gn.to(x.device)
    return model.forward(x)


def check(
    batch_size: int, features: int, num_groups: int, dim1: int, dim2: int
) -> None:
    """
    Checks the correctness of the GroupNorm kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        features: Number of feature channels
        num_groups: Number of groups to divide channels into
        dim1: Height dimension
        dim2: Width dimension
    """
    x = torch.randn(
        [batch_size, features, dim1, dim2], device=DEVICE, dtype=torch.float32
    )

    # Create a GroupNorm layer and extract its parameters
    gn = nn.GroupNorm(num_groups=num_groups, num_channels=features, dtype=x.dtype).to(
        DEVICE
    )

    # Test GroupNorm with extracted parameters
    run_example(
        lambda x: groupnorm_kernel(x, gn.weight, gn.bias, num_groups),
        lambda x: pytorch_baseline(x),
        (x,),
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 8
    features = 16
    num_groups = 4
    dim1 = 512
    dim2 = 512
    check(batch_size, features, num_groups, dim1, dim2)


if __name__ == "__main__":
    main()
