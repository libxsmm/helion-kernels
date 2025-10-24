import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def rmsnorm_kernel(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies RMS Normalization to the input tensor using Helion.
    RMS normalization normalizes by the root mean square across the feature dimension.

    Args:
        x: Input tensor of shape (batch_size, num_features, height, width)
        eps: Small value for numerical stability

    Returns:
        Output tensor with RMS Normalization applied, same shape as input
    """
    batch_size, num_features, height, width = x.size()
    out = torch.empty_like(x)

    for tile_b, tile_h, tile_w in hl.tile([batch_size, height, width]):
        # Get spatial slice across all features for this position
        spatial_slice = x[tile_b, :, tile_h, tile_w].to(
            torch.float32
        )  # Shape: [tile_b, num_features, tile_h, tile_w]

        # Compute mean of squared values across feature dimension
        mean_squared = torch.mean(
            spatial_slice * spatial_slice, dim=-3, keepdim=True
        )  # [tile_b, 1, tile_h, tile_w]

        # Compute RMS
        rms = torch.sqrt(mean_squared + eps)

        # Normalize by RMS
        normalized = spatial_slice / rms

        out[tile_b, :, tile_h, tile_w] = normalized.to(out.dtype)

    return out


class Model:
    """
    Simple model that performs RMS Normalization.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model(num_features=x.size(1), eps=1e-5)
    return model.forward(x)


def check(batch_size: int, features: int, dim1: int, dim2: int) -> None:
    """
    Checks the correctness of the RMSNorm kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        features: Number of feature channels
        dim1: Height dimension
        dim2: Width dimension
    """
    x = torch.randn(
        [batch_size, features, dim1, dim2], device=DEVICE, dtype=torch.float16
    )

    # Test RMSNorm
    run_example(lambda x: rmsnorm_kernel(x, eps=1e-5), pytorch_baseline, (x,))


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
