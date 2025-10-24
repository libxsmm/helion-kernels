import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def diagonal_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of a diagonal matrix with another matrix using Helion.
    C = diag(A) * B

    Args:
        A: 1D tensor representing the diagonal of the diagonal matrix, shape (N,)
        B: 2D tensor representing the second matrix, shape (N, M)

    Returns:
        Output tensor of shape (N, M)
    """
    N = A.size(0)
    N2, M = B.size()
    assert N == N2, f"size mismatch {N} != {N2}"

    out = torch.empty(
        [N, M], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device
    )

    # For diagonal matrix multiplication: C[i,j] = A[i] * B[i,j]
    for tile_n, tile_m in hl.tile([N, M]):
        # Get diagonal elements and matrix elements
        diag_elements = A[tile_n]
        matrix_elements = B[tile_n, tile_m]

        # Multiply diagonal elements with corresponding rows
        result = diag_elements[:, None] * matrix_elements

        out[tile_n, tile_m] = result

    return out


class Model:
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
        pass

    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return torch.diag(A) @ B


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(N: int, M: int) -> None:
    """
    Checks the correctness of the diagonal matrix multiplication kernel against PyTorch baseline.

    Args:
        N: Size of diagonal matrix and first dimension of B
        M: Second dimension of matrix B
    """
    A = torch.randn([N], device=DEVICE, dtype=torch.float16)
    B = torch.randn([N, M], device=DEVICE, dtype=torch.float16)

    # Test diagonal matrix multiplication
    run_example(diagonal_matmul, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(4096, 4096)


if __name__ == "__main__":
    main()
