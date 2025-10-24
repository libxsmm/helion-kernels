import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def mse_loss_kernel(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Mean Squared Error loss using Helion.

    Args:
        predictions: Predicted values tensor of shape [batch_size, seq_len]
        targets: Target values tensor of same shape as predictions

    Returns:
        Scalar tensor containing the MSE loss
    """
    batch_size, seq_len = predictions.size()
    total_elements = batch_size * seq_len

    # Create temporary tensor to store sum
    sum_tensor = torch.zeros([], dtype=torch.float32, device=predictions.device)

    # Tile over batch and sequence dimensions
    for tile_b, tile_s in hl.tile([batch_size, seq_len]):
        # Compute squared error
        diff = predictions[tile_b, tile_s] - targets[tile_b, tile_s]  # [tile_m, tile_n]
        squared_error = diff * diff  # [tile_m, tile_n]

        # Sum over the tile elements one dimension at a time
        squared_error = squared_error.sum(-1)
        squared_error = squared_error.sum(-1)

        # Add to sum
        hl.atomic_add(sum_tensor, [], squared_error)

    # Compute mean
    mse_loss = sum_tensor / total_elements

    return mse_loss.to(predictions.dtype)


class Model:
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """

    def __init__(self):
        pass

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)


def pytorch_baseline(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(predictions, targets)


def check(batch_size: int, input_shape: tuple) -> None:
    """
    Checks the correctness of the MSE loss kernel against PyTorch baseline.

    Args:
        batch_size: Batch size
        input_shape: Shape of input tensor (excluding batch dimension)
    """
    scale = torch.rand((), device=DEVICE)
    predictions = (
        torch.randn(
            [batch_size] + list(input_shape), device=DEVICE, dtype=torch.float16
        )
        * scale
    )
    targets = torch.randn(
        [batch_size] + list(input_shape), device=DEVICE, dtype=torch.float16
    )

    # Test MSE loss
    run_example(
        lambda pred, targ: mse_loss_kernel(pred, targ),
        lambda pred, targ: pytorch_baseline(pred, targ),
        (predictions, targets),
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 32768
    input_shape = (32768,)

    check(batch_size, input_shape)


if __name__ == "__main__":
    main()
