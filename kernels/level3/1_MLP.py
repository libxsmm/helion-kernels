import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor

from collections.abc import Callable


@helion.kernel(
    static_shapes=True,
)
def linear_kernel(
    x: torch.Tensor,  # [batch_size, input_size]
    weight: torch.Tensor,  # [output_size, input_size]
    bias: torch.Tensor,  # [output_size]
    activation: Callable[[torch.Tensor], torch.Tensor] = lambda acc: acc,
) -> torch.Tensor:
    """
    Generic linear layer kernel with optional ReLU activation using Helion.

    Args:
        x: Input tensor of shape (batch_size, input_size)
        weight: Weight matrix of shape (output_size, input_size)
        bias: Bias vector of shape (output_size,)
        apply_relu: Whether to apply ReLU activation after linear transformation

    Returns:
        Output tensor of shape (batch_size, output_size)
    """
    batch_size, input_size = x.size()
    output_size, input_size_w = weight.size()

    assert input_size == input_size_w, "Input size mismatch"

    # Create output tensor
    out = torch.empty([batch_size, output_size], dtype=x.dtype, device=x.device)

    # Linear computation with optional activation
    for tile_b, tile_o in hl.tile([batch_size, output_size]):  # [tile_b, tile_o]
        acc = hl.zeros([tile_b, tile_o], dtype=torch.float32)  # [tile_b, tile_o]

        # GEMM computation: x @ weight.T + bias
        for tile_i in hl.tile(input_size):  # [tile_i]
            x_slice = x[tile_b, tile_i]  # [tile_b, tile_i]
            weight_slice = weight[tile_o, tile_i]  # [tile_o, tile_i]

            acc = torch.addmm(acc, x_slice, weight_slice.T)  # [tile_b, tile_o]

        # Add bias
        linear_out = acc.to(out.dtype) + bias[tile_o]  # [tile_b, tile_o]

        # Apply activation
        out[tile_b, tile_o] = activation(linear_out)

    return out


def mlp_kernel(
    x: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
    weight3: torch.Tensor,
    bias3: torch.Tensor,
) -> torch.Tensor:
    """
    MLP forward pass using multiple calls to the generic linear kernel.

    Args:
        x: Input tensor
        weight1, bias1: First layer parameters
        weight2, bias2: Second layer parameters
        weight3, bias3: Third layer parameters

    Returns:
        Final MLP output
    """

    def activation_relu(acc: Tensor) -> Tensor:
        return torch.relu(acc)

    # Layer 1: Linear + ReLU
    layer1_out = linear_kernel(x, weight1, bias1, activation_relu)

    # Layer 2: Linear + ReLU
    layer2_out = linear_kernel(layer1_out, weight2, bias2, activation_relu)

    # Layer 3: Linear (no activation)
    output = linear_kernel(layer2_out, weight3, bias3)

    return output


class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()

        self.layers = []
        current_input_size = input_size

        for layer_size in layer_sizes:
            self.layers.append(
                nn.Linear(current_input_size, layer_size, dtype=torch.float16)
            )
            self.layers.append(nn.ReLU())
            current_input_size = layer_size

        self.layers.append(
            nn.Linear(current_input_size, output_size, dtype=torch.float16)
        )

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)


def pytorch_baseline(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    input_size = x.size(-1)
    layer_sizes = [2048, 2048]
    output_size = 1024

    model = Model(input_size, layer_sizes, output_size)
    model = model.to(x.device)
    return model


def check(
    batch_size: int, input_size: int, layer_sizes: list, output_size: int
) -> None:
    x = torch.rand(batch_size, input_size, device=DEVICE, dtype=torch.float16)

    # Create MLP layers to extract weights and biases
    baseline_model = pytorch_baseline(x)
    layer1 = baseline_model.layers[0]
    layer2 = baseline_model.layers[2]
    layer3 = baseline_model.layers[4]

    run_example(
        lambda x: mlp_kernel(
            x,
            layer1.weight,
            layer1.bias,
            layer2.weight,
            layer2.bias,
            layer3.weight,
            layer3.bias,
        ),
        baseline_model.forward,
        (x,),
    )


def main() -> None:
    batch_size = 128
    input_size = 512
    layer_sizes = [2048, 2048]
    output_size = 1024
    check(batch_size, input_size, layer_sizes, output_size)


if __name__ == "__main__":
    main()
