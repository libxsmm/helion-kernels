import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def symmetric_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of two symmetric matrices using Helion.
    C = A * B

    Args:
        A: Input symmetric matrix A, shape (N, N)
        B: Input symmetric matrix B, shape (N, N)

    Returns:
        Output matrix C, shape (N, N)
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
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """

    def __init__(self):
        pass

    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return torch.matmul(A, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(N: int) -> None:
    """
    Checks the correctness of the symmetric matrix multiplication kernel against PyTorch baseline.

    Args:
        N: Size of the symmetric matrices (N x N)
    """
    # Generate symmetric matrices
    A = torch.randn([N, N], device=DEVICE, dtype=torch.float16)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.randn([N, N], device=DEVICE, dtype=torch.float16)
    B = (B + B.T) / 2  # Ensure symmetry

    # Test symmetric matrix multiplication
    run_example(symmetric_matmul, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(4096)


if __name__ == "__main__":
    main()
