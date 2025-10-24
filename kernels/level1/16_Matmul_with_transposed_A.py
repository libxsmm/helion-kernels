import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def matmul_transposed_A(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication with transposed A using Helion.
    C = A.T * B

    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    K, M = A.size()
    K2, N = B.size()
    assert K == K2, f"size mismatch {K} != {K2}"

    out = torch.empty(
        [M, N], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device
    )

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            # A.T[tile_m, tile_k] = A[tile_k, tile_m]
            acc = torch.addmm(acc, A[tile_k, tile_m].T, B[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """

    def __init__(self):
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A.T, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(K: int, M: int, N: int) -> None:
    """
    Checks the correctness of the transposed A matrix multiplication kernel against PyTorch baseline.

    Args:
        K: Shared dimension between A and B
        M: First dimension of A.T (second dimension of A)
        N: Second dimension of B
    """
    A = torch.randn([K, M], device=DEVICE, dtype=torch.float16)
    B = torch.randn([K, N], device=DEVICE, dtype=torch.float16)

    # Test matrix multiplication with transposed A
    run_example(matmul_transposed_A, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    check(K, M, N)


if __name__ == "__main__":
    main()
