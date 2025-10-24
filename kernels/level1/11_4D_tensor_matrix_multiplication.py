import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def tensor_matrix_mul_4d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs 4D tensor-matrix multiplication using Helion.
    C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A: Input 4D tensor of shape (b, i, j, l)
        B: Input matrix of shape (l, k)

    Returns:
        Output tensor of shape (b, i, j, k)
    """
    b, i, j, l = A.size()
    l2, k = B.size()
    assert l == l2, f"size mismatch {l} != {l2}"

    out = torch.empty(
        [b, i, j, k], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device
    )

    for tile_b, tile_i, tile_j, tile_k in hl.tile([b, i, j, k]):
        acc = hl.zeros([tile_b, tile_i, tile_j, tile_k], dtype=torch.float32)
        for tile_l in hl.tile(l):
            acc += torch.einsum(
                "bijl,lk->bijk", A[tile_b, tile_i, tile_j, tile_l], B[tile_l, tile_k]
            )
        out[tile_b, tile_i, tile_j, tile_k] = acc

    return out


class Model:
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """

    def __init__(self):
        pass

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return torch.einsum("bijl,lk->bijk", A, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(b: int, i: int, j: int, l: int, k: int) -> None:
    """
    Checks the correctness of the 4D tensor-matrix multiplication kernel against PyTorch baseline.

    Args:
        b: Batch dimension size
        i: First spatial dimension
        j: Second spatial dimension
        l: Shared dimension between A and B
        k: Output dimension
    """
    A = torch.randn([b, i, j, l], device=DEVICE, dtype=torch.float16)
    B = torch.randn([l, k], device=DEVICE, dtype=torch.float16)

    # Test 4D tensor-matrix multiplication
    run_example(tensor_matrix_mul_4d, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(8, 256, 512, 256, 768)


if __name__ == "__main__":
    main()
