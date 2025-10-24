import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(static_shapes=True)
def matrix_scalar_mul_kernel(
    A: torch.Tensor,
    s: float,
) -> torch.Tensor:
    """
    Performs matrix-scalar multiplication using Helion.

    Args:
        A: Input matrix of shape [M, N]
        s: Scalar value

    Returns:
        Output matrix of shape [M, N]
    """
    M, N = A.size()

    out = torch.empty([M, N], dtype=A.dtype, device=A.device)

    for tile_m, tile_n in hl.tile([M, N]):
        # Element-wise multiplication with scalar
        result = A[tile_m, tile_n] * s
        out[tile_m, tile_n] = result

    return out


class Model:
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """

    def __init__(self):
        pass

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return A * s


def pytorch_baseline(A: torch.Tensor, s: float) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, s)


def check(M: int, N: int) -> None:
    """
    Checks the correctness of the matrix-scalar multiplication kernel against PyTorch baseline.

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix A
    """
    A = torch.randn([M, N], device=DEVICE, dtype=torch.float16)
    s = 3.14

    # Test matrix-scalar multiplication
    run_example(
        lambda A, s: matrix_scalar_mul_kernel(A, s),
        lambda A, s: pytorch_baseline(A, s),
        (A, s),
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    M = 1024 * 4
    N = 4096 * 4

    check(M, N)


if __name__ == "__main__":
    main()
