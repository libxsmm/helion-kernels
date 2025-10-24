import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def gemm_divide_sum_scaling_kernel(
    x: torch.Tensor,  # [batch_size, input_size]
    weight: torch.Tensor,  # [hidden_size, input_size]
    scaling_factor: float = 1.5,
) -> torch.Tensor:
    """
    Performs GEMM -> divide by 2 -> sum -> scaling using Helion.

    Args:
        x: Input tensor of shape (batch_size, input_size)
        weight: Weight matrix of shape (hidden_size, input_size)
        scaling_factor: Scaling factor applied after sum

    Returns:
        Output tensor of shape (batch_size, 1)
    """
    batch_size, input_size = x.size()
    hidden_size, input_size_w = weight.size()

    assert input_size == input_size_w, "Input size mismatch"

    # Create output tensor
    out = torch.empty([batch_size, 1], dtype=torch.float32, device=x.device)

    # Process each batch sample
    for tile_b in hl.tile(batch_size):  # [tile_b]
        # Accumulator for sum across hidden dimension
        batch_sum = hl.zeros([tile_b], dtype=torch.float32)  # [tile_b]

        # Iterate over hidden dimension
        for tile_h in hl.tile(hidden_size):  # [tile_h]
            # Initialize accumulator for GEMM
            gemm_acc = hl.zeros(
                [tile_b, tile_h], dtype=torch.float32
            )  # [tile_b, tile_h]

            # GEMM computation: x @ weight.T
            for tile_i in hl.tile(input_size):  # [tile_i]
                x_slice = x[tile_b, tile_i]  # [tile_b, tile_i]
                weight_slice = weight[tile_h, tile_i]  # [tile_h, tile_i]

                # Matrix multiplication
                gemm_acc = torch.addmm(
                    gemm_acc, x_slice, weight_slice.T
                )  # [tile_b, tile_h]

            # Divide by 2
            divided = gemm_acc / 2.0  # [tile_b, tile_h]

            # Sum across hidden dimension
            batch_sum = batch_sum + divided.sum(dim=-1)  # [tile_b]

        # Apply scaling factor
        scaled_result = batch_sum * scaling_factor  # [tile_b]

        # Store result
        out[tile_b, 0] = scaled_result.to(out.dtype)

    return out


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x, w):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = torch.mm(x, w.T, out_dtype=torch.float32)  # Gemm
        x = x / 2  # Divide
        x = torch.sum(x, dim=1, keepdim=True)  # Sum
        x = x * self.scaling_factor  # Scaling
        return x


def pytorch_baseline(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    input_size = x.size(-1)
    hidden_size = 8192
    scaling_factor = 1.5

    model = Model(input_size, hidden_size, scaling_factor)
    model = model.to(x.device)
    return model.forward(x, w)


def check(batch_size: int, input_size: int, hidden_size: int) -> None:
    x = torch.rand(batch_size, input_size, device=DEVICE, dtype=torch.float16)

    # Create weight parameter
    weight = torch.randn(hidden_size, input_size, device=DEVICE, dtype=x.dtype)

    run_example(
        lambda x, w: gemm_divide_sum_scaling_kernel(x, w, scaling_factor=1.5),
        pytorch_baseline,
        (x, weight),
    )


def main() -> None:
    batch_size = 1024
    input_size = 8192
    hidden_size = 8192
    check(batch_size, input_size, hidden_size)


if __name__ == "__main__":
    main()
