import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper])
def cross_entropy_loss_kernel(
    predictions: torch.Tensor,  # [batch_size, num_classes]
    targets: torch.Tensor,      # [batch_size]
) -> torch.Tensor:
    """
    Computes Cross Entropy Loss using Helion.
    
    Loss = -log(softmax(predictions)[targets])
    
    Args:
        predictions: Raw logits, shape (batch_size, num_classes)
        targets: Target class indices, shape (batch_size,)
        
    Returns:
        Cross entropy loss (mean reduction)
    """
    batch_size, num_classes = predictions.size()
    
    total_loss = torch.zeros([], dtype=torch.float32, device=predictions.device)  # scalar
    
    for tile_b in hl.tile(batch_size):  # [tile_b]
        # Compute log softmax for numerical stability
        # First find max for numerical stability
        max_vals = hl.full([tile_b], float('-inf'), dtype=torch.float32)  # [tile_b]
        
        for tile_c in hl.tile(num_classes):  # [tile_c]
            pred_slice = predictions[tile_b, tile_c]  # [tile_b, tile_c]
            max_vals = torch.maximum(max_vals, pred_slice.amax(dim=-1))  # [tile_b]
        
        # Compute sum of exponentials for softmax denominator
        sum_exp = hl.zeros([tile_b], dtype=torch.float32)  # [tile_b]
        
        for tile_c in hl.tile(num_classes):  # [tile_c]
            pred_slice = predictions[tile_b, tile_c]  # [tile_b, tile_c]
            exp_vals = torch.exp(pred_slice - max_vals[:, None])  # [tile_b, tile_c]
            sum_exp = sum_exp + exp_vals.sum(dim=-1)  # [tile_b]
        
        # Compute log softmax at target indices
        target_indices = targets[tile_b]  # [tile_b]
        
        # Get predictions at target indices
        target_logits = hl.zeros([tile_b], dtype=torch.float32)  # [tile_b]
        
        for tile_c in hl.tile(num_classes):  # [tile_c]
            pred_slice = predictions[tile_b, tile_c]  # [tile_b, tile_c]
            
            # Create mask for target indices
            class_indices = tile_c.index  # [tile_c]
            target_mask = target_indices[:, None] == class_indices[None, :]  # [tile_b, tile_c]
            
            # Extract target logits
            masked_logits = torch.where(target_mask, pred_slice, 0.0)  # [tile_b, tile_c]
            target_logits = target_logits + masked_logits.sum(dim=-1)  # [tile_b]
        
        # Compute log softmax: target_logit - max_val - log(sum_exp)
        log_softmax_target = target_logits - max_vals - torch.log(sum_exp)  # [tile_b]
        
        # Cross entropy loss is negative log softmax
        sample_loss = -log_softmax_target  # [tile_b]
        
        # Accumulate total loss
        hl.atomic_add(total_loss, [], sample_loss.sum())
    
    # Return mean loss
    return total_loss / batch_size  # scalar


class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)


def pytorch_baseline(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    model = Model()
    model = model.to(predictions.device)
    return model.forward(predictions, targets)


def check(batch_size: int, num_classes: int) -> None:
    predictions = torch.rand(batch_size, num_classes, device=DEVICE, dtype=torch.float16)
    targets = torch.randint(0, num_classes, (batch_size,), device=DEVICE)
    
    run_example(
        cross_entropy_loss_kernel,
        pytorch_baseline,
        (predictions, targets)
    )


def main() -> None:
    batch_size = 32768
    num_classes = 4096
    check(batch_size, num_classes)


if __name__ == "__main__":
    main()
