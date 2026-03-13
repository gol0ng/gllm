import numpy as np

class LMHeadAndLoss:
    def __init__(self, embed_dim, vocab_size):
        """
        初始化大模型的最后一层：输出头和损失计算
        """
        self.vocab_size = vocab_size
        
        # 最后的线性映射矩阵 W_out
        # 负责把 embed_dim (比如 64) 放大回 vocab_size (比如 5000)
        self.W_out = np.random.randn(embed_dim, vocab_size) * np.sqrt(2.0 / embed_dim)
        self.b_out = np.zeros(vocab_size)

    def forward(self, hidden_states, targets=None):
        """
        前向传播
        :param hidden_states: Transformer 最后一层的输出, Shape: (batch, seq_len, embed_dim)
        :param targets: DataLoader 里的目标 Y, Shape: (batch, seq_len)
        """
        self.hidden_states = hidden_states
        
        # 1. 线性映射得到 Logits (每个字的原始打分)
        # logits Shape: (batch, seq_len, vocab_size)
        self.logits = np.matmul(hidden_states, self.W_out) + self.b_out
        
        # 如果没有传目标答案 (比如在真正生成文本的推理阶段)，直接返回 logits 即可
        if targets is None:
            return self.logits, None
            
        # 2. 计算 Cross-Entropy Loss (需要展平数组方便计算)
        # 将 (batch, seq_len) 展平成一维数组，方便一一对应
        batch_size, seq_len = targets.shape
        logits_flat = self.logits.reshape(batch_size * seq_len, self.vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)
        
        # 3. Softmax 转换概率
        # 减去最大值防溢出
        logits_max = np.max(logits_flat, axis=-1, keepdims=True) 
        exp_logits = np.exp(logits_flat - logits_max)
        self.probs_flat = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # 4. 提取正确答案对应的预测概率
        # NumPy 高级索引：去每一行取出 target 对应的那个概率值
        correct_probs = self.probs_flat[np.arange(len(targets_flat)), targets_flat]
        
        # 5. 计算负对数似然 (加一个极小值 eps 防止 log(0) 报错)
        eps = 1e-10
        loss = -np.mean(np.log(correct_probs + eps))
        
        self.targets_flat = targets_flat
        return self.logits, loss

    def backward(self):
        """
        反向传播：极其优雅的 P - Y
        """
        batch_size_x_seq_len = self.probs_flat.shape[0]
        
        # 1. 交叉熵的梯度：预测概率 - 真实标签(One-hot)
        grad_logits_flat = self.probs_flat.copy()
        
        # 真实标签的位置减去 1 (因为 One-hot 编码下真实位置是 1，其他位置是 0)
        # 所以错的选项梯度是 P - 0 = P，对的选项梯度是 P - 1
        grad_logits_flat[np.arange(batch_size_x_seq_len), self.targets_flat] -= 1.0
        
        # 因为我们 loss 算了均值，所以梯度也要除以总样本数
        grad_logits_flat = grad_logits_flat / batch_size_x_seq_len
        
        # 变回原来的 3D 形状 (batch, seq_len, vocab_size)
        grad_logits = grad_logits_flat.reshape(self.logits.shape)
        
        # 2. 线性层 W_out 的反向传播
        # 对权重的梯度
        self.grad_W_out = np.matmul(np.swapaxes(self.hidden_states, 1, 2), grad_logits)
        self.grad_b_out = np.sum(grad_logits, axis=(0, 1))
        
        # 计算传回给 Transformer Block 的残差梯度！
        grad_hidden_states = np.matmul(grad_logits, self.W_out.T)
        
        # 累加 Batch 的权重梯度
        self.grad_W_out = np.sum(self.grad_W_out, axis=0)
        
        return grad_hidden_states

