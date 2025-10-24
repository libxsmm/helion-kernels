import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def triplet_margin_loss_kernel(
    anchor: torch.Tensor,  # [batch_size, features]
    positive: torch.Tensor,  # [batch_size, features]
    negative: torch.Tensor,  # [batch_size, features]
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Computes Triplet Margin Loss using Helion.

    Loss = max(0, ||anchor - positive||_2 - ||anchor - negative||_2 + margin)

    Args:
        anchor: Anchor samples, shape (batch_size, features)
        positive: Positive samples, shape (batch_size, features)
        negative: Negative samples, shape (batch_size, features)
        margin: Margin value for triplet loss

    Returns:
        Triplet margin loss (mean reduction)
    """
    batch_size, features = anchor.size()

    total_loss = torch.zeros([], dtype=torch.float32, device=anchor.device)  # scalar

    for tile_b in hl.tile(batch_size):  # [tile_b]
        # Compute distances for each sample in the batch
        pos_dist_sq = hl.zeros([tile_b], dtype=torch.float32)  # [tile_b]
        neg_dist_sq = hl.zeros([tile_b], dtype=torch.float32)  # [tile_b]

        for tile_f in hl.tile(features):  # [tile_f]
            anchor_slice = anchor[tile_b, tile_f]  # [tile_b, tile_f]
            positive_slice = positive[tile_b, tile_f]  # [tile_b, tile_f]
            negative_slice = negative[tile_b, tile_f]  # [tile_b, tile_f]

            # Compute squared differences
            pos_diff = anchor_slice - positive_slice  # [tile_b, tile_f]
            neg_diff = anchor_slice - negative_slice  # [tile_b, tile_f]

            pos_diff_sq = pos_diff * pos_diff  # [tile_b, tile_f]
            neg_diff_sq = neg_diff * neg_diff  # [tile_b, tile_f]

            # Accumulate squared distances
            pos_dist_sq = pos_dist_sq + pos_diff_sq.sum(dim=-1)  # [tile_b]
            neg_dist_sq = neg_dist_sq + neg_diff_sq.sum(dim=-1)  # [tile_b]

        # Compute L2 distances
        pos_dist = torch.sqrt(pos_dist_sq)  # [tile_b]
        neg_dist = torch.sqrt(neg_dist_sq)  # [tile_b]

        # Compute triplet loss: max(0, pos_dist - neg_dist + margin)
        triplet_loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)  # [tile_b]

        # Accumulate total loss
        hl.atomic_add(total_loss, [], triplet_loss.sum())

    # Return mean loss
    return total_loss / batch_size  # scalar


class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """

    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


def pytorch_baseline(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    model = Model(margin=1.0)
    model = model.to(anchor.device)
    return model.forward(anchor, positive, negative)


def check(batch_size: int, features: int) -> None:
    scale = torch.rand((), device=DEVICE)
    anchor = (
        torch.rand(batch_size, features, device=DEVICE, dtype=torch.float16) * scale
    )
    positive = torch.rand(batch_size, features, device=DEVICE, dtype=torch.float16)
    negative = torch.rand(batch_size, features, device=DEVICE, dtype=torch.float16)

    run_example(
        triplet_margin_loss_kernel, pytorch_baseline, (anchor, positive, negative)
    )


def main() -> None:
    batch_size = 32768
    features = 8192
    check(batch_size, features)


if __name__ == "__main__":
    main()
