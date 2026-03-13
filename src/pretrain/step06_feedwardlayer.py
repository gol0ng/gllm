import numpy as np

class FFN:
    def __init__(self, embed_dim, hidden_dim=None):
        # GPT 默认将内部隐藏层放大 4 倍
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim
            
        # 第一层线性变换 W1, b1
        self.W1 = np.random.randn(embed_dim, hidden_dim) * np.sqrt(2.0 / embed_dim)
        self.b1 = np.zeros(hidden_dim)
        
        # 第二层线性变换 W2, b2
        self.W2 = np.random.randn(hidden_dim, embed_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        self.x = x
        # 第一层线性映射
        self.z1 = np.matmul(x, self.W1) + self.b1
        
        # 非线性激活函数 (这里用最简单的 ReLU: 小于0的变0，大于0的不变)
        self.a1 = np.maximum(0, self.z1)
        
        # 第二层线性映射，缩回原来的维度
        out = np.matmul(self.a1, self.W2) + self.b2
        return out

    def backward(self, grad_output):
        # 第二层反向
        self.grad_W2 = np.matmul(np.swapaxes(self.a1, 1, 2), grad_output)
        self.grad_b2 = np.sum(grad_output, axis=(0, 1))
        grad_a1 = np.matmul(grad_output, self.W2.T)
        
        # ReLU 激活函数的反向：前向时大于 0 的地方梯度原样传递，小于 0 的地方梯度截断为 0
        grad_z1 = grad_a1 * (self.z1 > 0)
        
        # 第一层反向
        self.grad_W1 = np.matmul(np.swapaxes(self.x, 1, 2), grad_z1)
        self.grad_b1 = np.sum(grad_z1, axis=(0, 1))
        grad_x = np.matmul(grad_z1, self.W1.T)
        
        # 累加 Batch 的梯度
        self.grad_W1 = np.sum(self.grad_W1, axis=0)
        self.grad_W2 = np.sum(self.grad_W2, axis=0)
        return grad_x
