import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def gemm_multiply_leaky_relu_kernel(
    x: torch.Tensor,  # [batch_size, in_features]
    weight: torch.Tensor,  # [out_features, in_features]
    bias: torch.Tensor,  # [out_features]
    multiplier: float = 2.0,
    negative_slope: float = 0.1,
) -> torch.Tensor:
    """
    Performs GEMM, multiplies result, and applies LeakyReLU using Helion.

    Args:
        x: Input tensor of shape (batch_size, in_features)
        weight: Weight matrix of shape (out_features, in_features)
        bias: Bias vector of shape (out_features,)
        multiplier: Scalar multiplier applied after GEMM
        negative_slope: Negative slope for LeakyReLU

    Returns:
        Output tensor after GEMM -> multiply -> LeakyReLU
    """
    batch_size, in_features = x.size()
    out_features, in_features_w = weight.size()

    assert in_features == in_features_w, "Feature dimension mismatch"

    # Create output tensor
    out = torch.empty([batch_size, out_features], dtype=x.dtype, device=x.device)

    # Main computation: GEMM + multiply + LeakyReLU
    for tile_b, tile_o in hl.tile([batch_size, out_features]):  # [tile_b, tile_o]
        # Initialize accumulator for GEMM
        acc = hl.zeros([tile_b, tile_o], dtype=torch.float32)  # [tile_b, tile_o]

        # GEMM computation: x @ weight.T + bias
        for tile_i in hl.tile(in_features):  # [tile_i]
            x_slice = x[tile_b, tile_i]  # [tile_b, tile_i]
            weight_slice = weight[tile_o, tile_i]  # [tile_o, tile_i]

            # Matrix multiplication
            acc = torch.addmm(acc, x_slice, weight_slice.T)  # [tile_b, tile_o]

        # Add bias
        gemm_result = acc + bias[tile_o]  # [tile_b, tile_o]

        # Apply multiplier
        multiplied = gemm_result * multiplier  # [tile_b, tile_o]

        # Apply LeakyReLU: max(0, x) + negative_slope * min(0, x)
        leaky_relu_result = torch.where(
            multiplied >= 0, multiplied, multiplied * negative_slope
        )  # [tile_b, tile_o]

        # Store result
        out[tile_b, tile_o] = leaky_relu_result.to(out.dtype)

    return out


class Model(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """

    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, dtype=torch.float16)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x

    def gemm_model(self):
        return self.gemm


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    in_features = x.size(-1)
    out_features = 8192
    multiplier = 2.0
    negative_slope = 0.1

    model = Model(in_features, out_features, multiplier, negative_slope)
    model = model.to(x.device)
    return model


def check(batch_size: int, in_features: int, out_features: int) -> None:
    x = torch.rand(batch_size, in_features, device=DEVICE, dtype=torch.float16)

    # Create the model to extract weights and bias
    baseline_model = pytorch_baseline(x)
    linear = baseline_model.gemm

    run_example(
        lambda x: gemm_multiply_leaky_relu_kernel(
            x, linear.weight, linear.bias, multiplier=2.0, negative_slope=0.1
        ),
        baseline_model.forward,
        (x,),
    )


def main() -> None:
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    check(batch_size, in_features, out_features)


if __name__ == "__main__":
    main()
