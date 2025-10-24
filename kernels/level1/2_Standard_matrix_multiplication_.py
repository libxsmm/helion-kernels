import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(
    static_shapes=True,
)
def standard_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs standard matrix multiplication using Helion.
    C = A * B
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    M, K = A.size()
    K2, N = B.size()
    assert K == K2, f"size mismatch {K} != {K2}"
    
    out = torch.empty([M, N], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device)
    
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, A[tile_m, tile_k], B[tile_k, tile_n])
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
        return torch.matmul(A, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(M: int, K: int, N: int) -> None:
    """
    Checks the correctness of the standard matrix multiplication kernel against PyTorch baseline.
    
    Args:
        M: Number of rows in matrix A
        K: Number of columns in matrix A and rows in matrix B
        N: Number of columns in matrix B
    """
    A = torch.randn([M, K], device=DEVICE, dtype=torch.float16)
    B = torch.randn([K, N], device=DEVICE, dtype=torch.float16)
    
    # Test standard matrix multiplication
    run_example(standard_matmul, pytorch_baseline, (A, B))


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
