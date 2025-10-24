import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel(
    static_shapes=True,
)
def batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batched matrix multiplication using Helion.
    C = A * B where A, B, and C have the same batch dimension.

    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)

    Returns:
        Output tensor of shape (batch_size, m, n)
    """
    batch_size, m, k = A.size()
    batch_size2, k2, n = B.size()
    assert batch_size == batch_size2 and k == k2, (
        f"size mismatch: A{A.size()}, B{B.size()}"
    )

    out = torch.empty(
        [batch_size, m, n], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device
    )

    for tile_b, tile_m, tile_n in hl.tile([batch_size, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc

    return out


class Model:
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """

    def __init__(self):
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return torch.bmm(A, B)


def pytorch_baseline(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(A, B)


def check(batch_size: int, m: int, k: int, n: int) -> None:
    """
    Checks the correctness of the batched matrix multiplication kernel against PyTorch baseline.

    Args:
        batch_size: Batch dimension size
        m: Number of rows in matrix A
        k: Number of columns in matrix A and rows in matrix B
        n: Number of columns in matrix B
    """
    A = torch.randn([batch_size, m, k], device=DEVICE, dtype=torch.float16)
    B = torch.randn([batch_size, k, n], device=DEVICE, dtype=torch.float16)

    # Test batched matrix multiplication
    run_example(batched_matmul, pytorch_baseline, (A, B))


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    check(batch_size, m, k, n)


if __name__ == "__main__":
    main()
