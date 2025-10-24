import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes Hinge Loss for binary classification tasks using Helion.

    Args:
        predictions: Prediction tensor of shape (batch_size, input_shape)
        targets: Target tensor of shape (batch_size,) with values in {-1, 1}

    Returns:
        Scalar tensor containing the mean hinge loss
    """
    batch_size = predictions.size(0)
    input_size = predictions.size(1)

    # Allocate output for loss values
    losses = torch.empty(
        [batch_size, input_size], dtype=torch.float32, device=predictions.device
    )

    # Compute hinge loss: max(0, 1 - predictions * targets)
    for tile_batch, tile_input in hl.tile([batch_size, input_size]):
        pred_slice = predictions[tile_batch, tile_input]
        target_slice = targets[tile_batch]

        # Compute 1 - predictions * targets
        margin = 1.0 - pred_slice * target_slice[:, None]

        # Apply max(0, margin) = clamp(margin, min=0)
        hinge_values = torch.clamp(margin, min=0.0)

        losses[tile_batch, tile_input] = hinge_values

    # Return mean of all loss values
    return losses.mean()


class Model:
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        pass

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))


def pytorch_baseline(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(predictions, targets)


def check(batch_size: int, input_shape: tuple[int, ...]) -> None:
    """
    Checks the correctness of the hinge loss kernel against PyTorch baseline.

    Args:
        batch_size: Number of samples in batch
        input_shape: Shape of input features
    """
    predictions = torch.randn(
        [batch_size, *input_shape], device=DEVICE, dtype=torch.float32
    )
    targets = torch.randint(0, 2, (batch_size,), device=DEVICE).float() * 2 - 1

    # Test hinge loss computation
    run_example(hinge_loss, pytorch_baseline, (predictions, targets))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 32768
    input_shape = (32768,)
    check(batch_size, input_shape)


if __name__ == "__main__":
    main()
