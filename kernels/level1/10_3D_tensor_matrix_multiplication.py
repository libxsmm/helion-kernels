import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
from torch import Tensor


@helion.kernel(
    static_shapes=True,
)
def tensor_matrix_mul_3d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs 3D tensor-matrix multiplication using Helion.
    
    Args:
        A: Input 3D tensor of shape (N, M, K)
        B: Input matrix of shape (K, L)
    
    Returns:
        Output tensor of shape (N, M, L)
    """
    N, M, K = A.size()
    K2, L = B.size()
    assert K == K2, f"size mismatch {K} != {K2}"
    
    out = torch.empty([N, M, L], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device)
    
    for tile_n, tile_m, tile_l in hl.tile([N, M, L]):
        acc = hl.zeros([tile_n, tile_m, tile_l], dtype=torch.float32)
        for tile_k in hl.tile(K):
            B_broadcasted = torch.broadcast_to(B[tile_k, tile_l], (tile_n.block_size, tile_k.block_size, tile_l.block_size))
            acc = torch.baddbmm(acc, A[tile_n, tile_m, tile_k], B_broadcasted)
        out[tile_n, tile_m, tile_l] = acc
    
    return out


class Model:
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return torch.matmul(A, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(N: int, M: int, K: int, L: int) -> None:
    """
    Checks the correctness of the 3D tensor-matrix multiplication kernel against PyTorch baseline.
    
    Args:
        N: Batch dimension size
        M: First matrix dimension
        K: Shared dimension between A and B
        L: Second matrix dimension
    """
    A = torch.randn([N, M, K], device=DEVICE, dtype=torch.float16)
    B = torch.randn([K, L], device=DEVICE, dtype=torch.float16)
    
    # Test 3D tensor-matrix multiplication
    run_example(tensor_matrix_mul_3d, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(16, 1024, 2048, 768)


if __name__ == "__main__":
    main()
