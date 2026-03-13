import numpy as np

class TransformerBlock:
    """一个完整的 Transformer Decoder Block"""
    def __init__(self, embed_dim, n_head=4):
        from step04_mutilheadattentionlayer import MultiHeadAttention
        from step05_normlayer import LayerNorm
        from step06_feedwardlayer import FFN

        self.attn = MultiHeadAttention(n_head=n_head, embed_dim=embed_dim)
        self.ln1 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, embed_dim * 4)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, x):
        """前向传播：Attn → Add → Norm → FFN → Add → Norm"""
        # 第一个子层：多头注意力 + 残差 + LayerNorm
        attn_out = self.attn.forward(x)
        x = self.ln1.forward(attn_out + x)

        # 第二个子层：前馈网络 + 残差 + LayerNorm
        ff_out = self.ffn.forward(x)
        x = self.ln2.forward(ff_out + x)
        return x

    def backward(self, grad_output):
        """反向传播"""
        # FFN 部分反向
        grad = self.ln2.backward(grad_output)
        grad_ffn = self.ffn.backward(grad[1])
        grad = self.ln1.backward(grad[0] + grad_ffn)

        # Attention 部分反向
        grad_attn = self.attn.backward(grad[0] + grad[1])
        return grad_attn
