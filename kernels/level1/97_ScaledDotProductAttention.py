import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example
import math


# From Helion attention example.
@helion.kernel(static_shapes=True)
def scaled_dot_product_attention_kernel(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Implements the attention mechanism: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        q_in: Query tensor of shape [..., seq_len_q, head_dim]
        k_in: Key tensor of shape [..., seq_len_k, head_dim]
        v_in: Value tensor of shape [..., seq_len_k, head_dim]

    Returns:
        Output tensor of shape [..., seq_len_q, head_dim]
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


class Model:
    def __init__(self):
        pass

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out


def pytorch_baseline(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch baseline using Model class."""
    model = Model()
    return model.forward(Q, K, V)


def check(
    batch_size: int, num_heads: int, sequence_length: int, embedding_dimension: int
) -> None:
    """
    Checks the correctness of the scaled dot-product attention kernel against PyTorch baseline.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        sequence_length: Sequence length
        embedding_dimension: Embedding dimension per head
    """
    Q = torch.randn(
        [batch_size, num_heads, sequence_length, embedding_dimension],
        device=DEVICE,
        dtype=torch.float16,
    )
    K = torch.randn(
        [batch_size, num_heads, sequence_length, embedding_dimension],
        device=DEVICE,
        dtype=torch.float16,
    )
    V = torch.randn(
        [batch_size, num_heads, sequence_length, embedding_dimension],
        device=DEVICE,
        dtype=torch.float16,
    )

    # Test scaled dot-product attention
    run_example(
        lambda Q, K, V: scaled_dot_product_attention_kernel(Q, K, V),
        lambda Q, K, V: pytorch_baseline(Q, K, V),
        (Q, K, V),
    )


def main() -> None:
    """
    Main function to run correctness checks.
    """
    batch_size = 2
    num_heads = 32
    sequence_length = 1024
    embedding_dimension = 64

    check(batch_size, num_heads, sequence_length, embedding_dimension)


if __name__ == "__main__":
    main()
