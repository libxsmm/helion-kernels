import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(static_shapes=True)
def matvec_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Helion.
    
    Args:
        A: Input matrix of shape [M, K]
        B: Input vector of shape [K, 1]
        
    Returns:
        Output vector of shape [M, 1]
    """
    M, K = A.size()
    K2, _ = B.size()
    assert K == K2, f"Dimension mismatch: {K} != {K2}"
    
    out = torch.empty([M, 1], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device)
    
    for tile_m in hl.tile(M):
        # Initialize accumulator
        acc = hl.zeros([tile_m, 1], dtype=torch.float32)

        # Accumulate over K dimension
        for tile_k in hl.tile(K):
            a_slice = A[tile_m, tile_k]  # [tile_m, tile_k]
            b_slice = B[tile_k, :]       # [tile_k, 1]
            acc = torch.addmm(acc, a_slice, b_slice)
        
        out[tile_m, :] = acc
    
    return out


class Model:
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        pass
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(M: int, K: int) -> None:
    """
    Checks the correctness of the matrix-vector multiplication kernel against PyTorch baseline.
    
    Args:
        M: Number of rows in matrix A
        K: Number of columns in matrix A and rows in vector B
    """
    A = torch.randn([M, K], device=DEVICE, dtype=torch.float16)
    B = torch.randn([K, 1], device=DEVICE, dtype=torch.float16)
    
    # Test matrix-vector multiplication
    run_example(
        lambda A, B: matvec_kernel(A, B),
        lambda A, B: pytorch_baseline(A, B),
        (A, B)
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    M = 256 * 8
    K = 4096 * 8
    
    check(M, K)


if __name__ == "__main__":
    main()
