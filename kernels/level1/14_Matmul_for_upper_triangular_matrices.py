import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def upper_triangular_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication for upper triangular matrices using Helion.
    C = A * B (with upper triangular result)

    Args:
        A: Upper triangular matrix of shape (N, N)
        B: Upper triangular matrix of shape (N, N)

    Returns:
        Upper triangular matrix C of shape (N, N)
    """
    N, N2 = A.size()
    N3, N4 = B.size()
    assert N == N2 == N3 == N4, f"size mismatch: A{A.size()}, B{B.size()}"

    out = torch.empty(
        [N, N], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device
    )

    for tile_m, tile_n in hl.tile([N, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(N):
            acc = torch.addmm(acc, A[tile_m, tile_k], B[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return torch.triu(out)


class Model:
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """

    def __init__(self):
        pass

    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return torch.triu(torch.matmul(A, B))


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(N: int) -> None:
    """
    Checks the correctness of the upper triangular matrix multiplication kernel against PyTorch baseline.

    Args:
        N: Size of the upper triangular matrices (N x N)
    """
    # Generate upper triangular matrices
    A = torch.triu(torch.randn([N, N], device=DEVICE, dtype=torch.float16))
    B = torch.triu(torch.randn([N, N], device=DEVICE, dtype=torch.float16))

    # Test upper triangular matrix multiplication
    run_example(upper_triangular_matmul, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(4096)


if __name__ == "__main__":
    main()
