import numpy as np

class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.eps = eps
        # 两个可学习参数：gamma (缩放) 和 beta (平移)
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
        
    def forward(self, x):
        # 沿着最后一个维度 (特征维度) 求均值和方差
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        
        # 减去均值，除以标准差 (加上 eps 防止分母为 0)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        
        # 乘以 gamma，加上 beta
        out = self.gamma * self.x_norm + self.beta
        
        # 缓存给反向传播用
        self.x = x
        return out

    def backward(self, grad_output):
        """LayerNorm 反向传播"""
        N = grad_output.shape[-1]

        # 计算对 gamma 和 beta 的梯度
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=(0, 1))
        self.grad_beta = np.sum(grad_output, axis=(0, 1))

        # 计算 x_norm 的梯度
        grad_x_norm = grad_output * self.gamma

        # 计算 var 的梯度
        grad_var = np.sum(grad_x_norm * (self.x - self.mean) * -0.5 * np.power(self.var + self.eps, -1.5), axis=-1, keepdims=True)

        # 计算 mean 的梯度
        grad_mean = np.sum(grad_x_norm * -1.0 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True)
        grad_mean = grad_mean + grad_var * np.sum(-2.0 * (self.x - self.mean), axis=-1, keepdims=True) / N

        # 最终 x 的梯度
        grad_x = grad_x_norm / np.sqrt(self.var + self.eps)
        grad_x = grad_x + grad_var * 2.0 * (self.x - self.mean) / N
        grad_x = grad_x + grad_mean / N

        return grad_x