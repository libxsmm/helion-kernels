import torch
import torch.nn as nn
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(static_shapes=True)
def tall_skinny_matmul_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication for tall-skinny matrices using Helion.
    
    Args:
        A: Input matrix of shape [M, K] where M >> K (tall and skinny)
        B: Input matrix of shape [K, N] where K is small
        
    Returns:
        Output matrix of shape [M, N]
    """
    M, K = A.size()
    K2, N = B.size()
    assert K == K2, f"Dimension mismatch: {K} != {K2}"
    
    out = torch.empty([M, N], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device)
    
    for tile_m, tile_n in hl.tile([M, N]):
        # Initialize accumulator
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        
        # Accumulate over K dimension
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, A[tile_m, tile_k], B[tile_k, tile_n])
        
        out[tile_m, tile_n] = acc.to(out.dtype)
    
    return out


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(M: int, N: int) -> None:
    """
    Checks the correctness of the tall-skinny matrix multiplication kernel against PyTorch baseline.
    
    Args:
        M: Large dimension (tall)
        N: Small dimension (skinny)
    """
    A = torch.randn([M, N], device=DEVICE, dtype=torch.float16)
    B = torch.randn([N, M], device=DEVICE, dtype=torch.float16)
    
    # Test matrix multiplication
    run_example(
        lambda A, B: tall_skinny_matmul_kernel(A, B),
        lambda A, B: pytorch_baseline(A, B),
        (A, B)
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    M = 16384 * 2
    N = 16 * 2
    
    check(M, N)


if __name__ == "__main__":
    main()
