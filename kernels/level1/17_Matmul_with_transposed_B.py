import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def matmul_transposed_B(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication with transposed B using Helion.
    C = A * B.T

    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (N, K)

    Returns:
        Output tensor of shape (M, N)
    """
    M, K = A.size()
    N, K2 = B.size()
    assert K == K2, f"size mismatch {K} != {K2}"

    out = torch.empty(
        [M, N], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device
    )

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            # B.T[tile_k, tile_n] = B[tile_n, tile_k]
            acc = torch.addmm(acc, A[tile_m, tile_k], B[tile_n, tile_k].T)
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
        return torch.matmul(A, B.T)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(M: int, K: int, N: int) -> None:
    """
    Checks the correctness of the transposed B matrix multiplication kernel against PyTorch baseline.

    Args:
        M: First dimension of A
        K: Shared dimension between A and B
        N: First dimension of B (second dimension of B.T)
    """
    A = torch.randn([M, K], device=DEVICE, dtype=torch.float16)
    B = torch.randn([N, K], device=DEVICE, dtype=torch.float16)

    # Test matrix multiplication with transposed B
    run_example(matmul_transposed_B, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    check(M, K, N)


if __name__ == "__main__":
    main()
