import numpy as np
from numba import jit, prange

# ==================== Numba 加速版本 ====================
@jit(nopython=True, cache=True)
def softmax_numba(x):
    """Numba 加速的 softmax"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

@jit(nopython=True, cache=True)
def apply_causal_mask_numba(scores, seq_len):
    """Numba 加速的因果掩码"""
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            scores[i, j] = -np.inf
    return scores

def softmax(x, axis=-1):
    """优先使用 numba 加速版"""
    if len(x.shape) == 2:
        # 对于 2D 直接调用 numba
        return softmax_numba(x.astype(np.float64))
    else:
        # 对于多维，用 numpy
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self,n_head = 4,embed_dim = 128):
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.d_k = embed_dim // n_head

        # 确保 embed_dim 能被 n_head 整除
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"

        # 初始化qkv权重矩阵W_q,W_k,W_v
        # 输入通过他们转换为注意力机制里面的QKV矩阵
        scale = np.sqrt(2.0 / embed_dim) # Kaiming 初始化
        self.W_q = np.random.randn(embed_dim, self.d_k * n_head) * scale
        self.W_k = np.random.randn(embed_dim, self.d_k * n_head) * scale
        self.W_v = np.random.randn(embed_dim, self.d_k * n_head) * scale

        self.W_o = np.random.randn(self.d_k * n_head, embed_dim) * scale

        # 缓存区：为了反向传播留下的案底
        self.cache = {}

    def forward(self,x, mask=None):
        """
        前向传播
        :param x: 输入张量，Shape: (batch_size, seq_len, embed_dim)
        :param mask: 可选的掩码，Shape: (seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 生成 Q, K, V (矩阵乘法)
        # Shape: (batch_size, seq_len, d_k * n_head)
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        # 2. 分割成多个头
        # Shape: (batch_size, n_head, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_head, self.d_k).transpose(0, 2, 1, 3)

        # 3. 计算注意力打分矩阵 (Q 乘以 K 的转置)
        # scores Shape: (batch_size, n_head, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        # 4. 应用因果掩码 (Causal Mask)
        # 生成一个上三角矩阵，k=1 表示对角线往上的元素全是 True
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(causal_mask, -np.inf, scores)

        # 5. 应用额外的掩码（如果提供）
        if mask is not None:
            scores = np.where(mask, -np.inf, scores)

        # 6. Softmax 归一化，得到注意力权重
        probs = softmax(scores, axis=-1)

        # 7. 用权重去乘以 V，得到融合了上下文的新向量
        # context Shape: (batch_size, n_head, seq_len, d_k)
        context = np.matmul(probs, V)

        # 8. 合并多个头
        # Shape: (batch_size, seq_len, d_k * n_head)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_k * self.n_head)

        # 9. 最后过一个线性层 Wo 输出
        out = np.matmul(context, self.W_o)

        # 记下所有的案底，反向传播要用
        self.cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'scores': scores, 'probs': probs, 'context': context
        }
        return out

    def backward(self, grad_output):
        """
        反向传播
        :param grad_output: 来自上一层的梯度，Shape: (batch_size, seq_len, embed_dim)
        """
        x = self.cache['x']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        probs = self.cache['probs']
        context = self.cache['context']

        batch_size = grad_output.shape[0]

        # 1. 对 Wo 求导
        # grad_W_o = context^T * grad_output
        self.grad_W_o = np.matmul(np.swapaxes(context, 1, 2), grad_output)
        grad_context = np.matmul(grad_output, self.W_o.T)

        # 2. 恢复多头形状
        # grad_context: (batch_size, seq_len, n_head, d_k) -> 转置成 (batch_size, n_head, seq_len, d_k)
        seq_len = grad_context.shape[1]
        grad_context = grad_context.reshape(batch_size, seq_len, self.n_head, self.d_k)
        grad_context = grad_context.transpose(0, 2, 1, 3)  # (batch, head, seq, d_k)

        # 3. 对 V 求导
        # grad_V = probs^T * grad_context
        grad_V = np.matmul(probs.transpose(0, 1, 3, 2), grad_context)
        # 恢复形状并计算 W_v 的梯度
        grad_V_reshaped = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_k * self.n_head)
        self.grad_W_v = np.matmul(x.transpose(0, 2, 1), grad_V_reshaped)

        # 4. 对 Softmax 的打分矩阵求导
        grad_probs = np.matmul(grad_context, V.transpose(0, 1, 3, 2))
        grad_scores = probs * (grad_probs - np.sum(probs * grad_probs, axis=-1, keepdims=True))

        # 5. 掩码处的梯度必须设为 0
        seq_len = grad_scores.shape[-1]
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        grad_scores[..., mask] = 0
        grad_scores = grad_scores / np.sqrt(self.d_k)

        # 6. 对 Q 和 K 求导
        grad_Q = np.matmul(grad_scores, K)
        grad_K = np.matmul(grad_scores.transpose(0, 1, 3, 2), Q)

        # 恢复形状并计算 W_q, W_k 的梯度
        grad_Q_reshaped = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_k * self.n_head)
        grad_K_reshaped = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_k * self.n_head)

        self.grad_W_q = np.matmul(x.transpose(0, 2, 1), grad_Q_reshaped)
        self.grad_W_k = np.matmul(x.transpose(0, 2, 1), grad_K_reshaped)

        # 7. 计算传给上一层 (Embedding层) 的梯度
        grad_x_q = np.matmul(grad_Q_reshaped, self.W_q.T)
        grad_x_k = np.matmul(grad_K_reshaped, self.W_k.T)
        grad_x_v = np.matmul(grad_V_reshaped, self.W_v.T)
        grad_x = grad_x_q + grad_x_k + grad_x_v

        # 把每个 Batch 的梯度求和 (因为参数 W 只有一份)
        self.grad_W_q = np.sum(self.grad_W_q, axis=0)
        self.grad_W_k = np.sum(self.grad_W_k, axis=0)
        self.grad_W_v = np.sum(self.grad_W_v, axis=0)
        self.grad_W_o = np.sum(self.grad_W_o, axis=0)

        return grad_x


        
