import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def square_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs square matrix multiplication using Helion.
    C = A * B

    Args:
        A: Input matrix A of shape (N, N)
        B: Input matrix B of shape (N, N)

    Returns:
        Output matrix C of shape (N, N)
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

    return out


class Model:
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    def __init__(self):
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(N: int) -> None:
    """
    Checks the correctness of the square matrix multiplication kernel against PyTorch baseline.

    Args:
        N: Size of the square matrices (N x N)
    """
    A = torch.randn([N, N], device=DEVICE, dtype=torch.float16)
    B = torch.randn([N, N], device=DEVICE, dtype=torch.float16)

    # Test square matrix multiplication
    run_example(square_matmul, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    N = 2048 * 2
    check(N)


if __name__ == "__main__":
    main()
