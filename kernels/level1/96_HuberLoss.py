import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper])
def huber_loss_kernel(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Computes Smooth L1 (Huber) Loss using Helion.
    
    Args:
        predictions: Predicted values tensor of shape [batch_size, ...]
        targets: Target values tensor of same shape as predictions
        beta: Threshold for switching between L1 and L2 loss (default: 1.0)
        
    Returns:
        Scalar tensor containing the Huber loss
    """
    batch_size, seq_len = predictions.size()  # predictions: [batch_size, seq_len]
    # targets: [batch_size, seq_len]
    total_elements = batch_size * seq_len
    
    # Create temporary tensor to store sum
    sum_tensor = torch.zeros([], dtype=torch.float32, device=predictions.device)  # sum_tensor: [1]
    
    # Tile over batch and sequence dimensions
    for tile_b, tile_s in hl.tile([batch_size, seq_len]):
        # Compute absolute difference
        diff = predictions[tile_b, tile_s] - targets[tile_b, tile_s]  # diff: [tile_b, tile_s]
        abs_diff = torch.abs(diff)  # abs_diff: [tile_b, tile_s]
        
        # Apply Huber loss formula:
        # if |diff| < beta: 0.5 * diff^2 / beta
        # else: |diff| - 0.5 * beta
        smooth_l1 = torch.where(
            abs_diff < beta,
            0.5 * diff * diff / beta,  # L2 loss region
            abs_diff - 0.5 * beta      # L1 loss region
        )  # smooth_l1: [tile_b, tile_s]

        # Sum over the tile elements one dimension at a time
        smooth_l1 = smooth_l1.sum(-1)
        smooth_l1 = smooth_l1.sum(-1)
        
        # Add to sum
        hl.atomic_add(sum_tensor, [], smooth_l1)
    
    # Compute mean
    huber_loss = sum_tensor / total_elements
    
    return huber_loss.to(predictions.dtype)


class Model:
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        return torch.nn.functional.smooth_l1_loss(predictions, targets)


def pytorch_baseline(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(predictions, targets)


def check(batch_size: int, input_shape: tuple) -> None:
    """
    Checks the correctness of the Huber loss kernel against PyTorch baseline.
    
    Args:
        batch_size: Batch size
        input_shape: Shape of input tensor (excluding batch dimension)
    """
    scale = torch.rand((), device=DEVICE)
    predictions = torch.randn([batch_size] + list(input_shape), device=DEVICE, dtype=torch.float16) * scale
    targets = torch.randn([batch_size] + list(input_shape), device=DEVICE, dtype=torch.float16)
    
    # Test Huber loss
    run_example(
        lambda pred, targ: huber_loss_kernel(pred, targ),
        lambda pred, targ: pytorch_baseline(pred, targ),
        (predictions, targets)
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
