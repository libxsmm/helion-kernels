import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def lower_triangular_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication for lower triangular matrices using Helion.
    C = A * B (with lower triangular result)

    Args:
        A: Lower triangular matrix of shape (N, N)
        B: Lower triangular matrix of shape (N, N)

    Returns:
        Lower triangular matrix C of shape (N, N)
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

    return torch.tril(out)


class Model:
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    """

    def __init__(self):
        pass

    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return torch.tril(torch.matmul(A, B))


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(M: int) -> None:
    """
    Checks the correctness of the lower triangular matrix multiplication kernel against PyTorch baseline.

    Args:
        M: Size of the lower triangular matrices (M x M)
    """
    # Generate lower triangular matrices
    A = torch.tril(torch.randn([M, M], device=DEVICE, dtype=torch.float16))
    B = torch.tril(torch.randn([M, M], device=DEVICE, dtype=torch.float16))

    # Test lower triangular matrix multiplication
    run_example(lower_triangular_matmul, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(4096)


if __name__ == "__main__":
    main()
