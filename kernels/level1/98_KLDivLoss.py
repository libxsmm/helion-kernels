import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def kl_div_kernel(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Kullback-Leibler Divergence using Helion with batchmean reduction.

    KL(P||Q) = sum(P * log(P/Q)) where P is target and Q is prediction

    Args:
        predictions: Predicted probabilities (Q), shape (batch_size, features)
        targets: Target probabilities (P), shape (batch_size, features)

    Returns:
        KL divergence loss with batchmean reduction
    """
    batch_size, features = predictions.size()

    # Compute total loss
    total_loss = torch.zeros([], dtype=torch.float32, device=predictions.device)

    for tile_b in hl.tile(batch_size):
        for tile_f in hl.tile(features):
            pred_slice = predictions[tile_b, tile_f]
            target_slice = targets[tile_b, tile_f]

            # Compute KL divergence: target * log(target / pred)
            eps = 1e-8
            pred_clamped = torch.clamp(pred_slice, min=eps)
            target_clamped = torch.clamp(target_slice, min=eps)

            # KL = target * (log(target) - log(pred))
            kl_contribution = target_clamped * (
                torch.log(target_clamped) - torch.log(pred_clamped)
            )

            # Sum over the tile elements one dimension at a time
            kl_contribution = kl_contribution.sum(-1)
            kl_contribution = kl_contribution.sum(-1)

            # Accumulate total loss
            hl.atomic_add(total_loss, [], kl_contribution)

    # Return batchmean reduction
    return total_loss / batch_size


class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.kl_div(
            torch.log(predictions), targets, reduction="batchmean"
        )


def pytorch_baseline(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    model = model.to(predictions.device)
    return model.forward(predictions, targets)


def check(batch_size: int, features: int) -> None:
    """
    Checks the correctness of the KL divergence kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        features: Number of features
    """
    # Create test data - probabilities that sum to 1
    scale = torch.rand((), device=DEVICE)
    predictions = (
        torch.rand(batch_size, features, device=DEVICE, dtype=torch.float16) * scale
    ).softmax(dim=-1)
    targets = torch.rand(
        batch_size, features, device=DEVICE, dtype=torch.float16
    ).softmax(dim=-1)

    # Test default batchmean reduction
    run_example(kl_div_kernel, pytorch_baseline, (predictions, targets))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 8192 * 2
    features = 8192 * 2
    check(batch_size, features)


if __name__ == "__main__":
    main()
