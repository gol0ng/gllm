import numpy as np

class Embedding:
    def __init__(self,vocab_size,embed_dim = 128):
        '''
        初始化Emebdding层
        :param vocab_size: 词表大小
        :param embed_dim: 向量维度
        '''
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # 初始化
        self.weight = np.random.randn(vocab_size, embed_dim) / np.sqrt(embed_dim)

        self.x_cache = None 

        self.grad_weight = np.zeros_like(self.weight)
    def forward(self,x):
        """
        前向传播：查表获取向量
        :param x: 输入的 ID 矩阵，Shape: (batch_size, block_size)
        :return: 对应的 Embedding 向量，Shape: (batch_size, block_size, embed_dim)
        """
        self.x_cache = x

        out = self.weight[x]
        return out
    def backward(self,grad_output):
        """
        反向传播：计算 Embedding 矩阵的梯度
        :param grad_output: 从下一层传回来的梯度，Shape: (batch_size, block_size, embed_dim)
        :return: Embedding 层是网络的第一层，不需要再往前传梯度给整数 ID，所以这里不返回任何东西。
        """
        # 每次反向传播前，先把上次累加的梯度清零
        self.grad_weight = np.zeros_like(self.weight)

        # np.add.at 是 NumPy 处理索引重复时累加梯度的神器
        # 它等价于：对于 x_cache 里的每一个 ID，把对应的 grad_output 加到 self.grad_weight 对应的行上
        np.add.at(self.grad_weight, self.x_cache, grad_output)
        
        # 我们把累加好的梯度保存在 self.grad_weight 中，所以不需要返回值
        # 之后优化器（如 SGD 或 Adam）会拿着这个去更新 self.weight。

class GllmInputEmbedding:
    def __init__(self, vocab_size, block_size = 512, embed_dim = 128):
        """
        :param vocab_size: 词表大小 (比如 5000)
        :param block_size: 模型的最大视野/上下文长度 (比如 512)
        :param embed_dim: 向量维度 (比如 128)
        """
        # 负责把 "字" 变成向量的抽屉柜
        self.token_emb = Embedding(vocab_size, embed_dim)
        
        # 负责把 "位置 (0,1,2...)" 变成向量的抽屉柜
        self.pos_emb = Embedding(block_size, embed_dim)

    def forward(self,x):
        """
        :param x: DataLoader 传来的 Batch 矩阵，Shape: (batch_size, seq_len)
        :return: 融合了字意和位置的特征矩阵，Shape: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len = x.shape
        
        # 1. 获取字向量
        # Shape: (batch_size, seq_len, embed_dim)
        tok_emb_out = self.token_emb.forward(x) 
        
        # 2. 生成绝对位置序号: [0, 1, 2, ..., seq_len - 1]
        # Shape: (seq_len,)
        positions = np.arange(seq_len)
        
        # 3. 获取位置向量
        # Shape: (seq_len, embed_dim)
        pos_emb_out = self.pos_emb.forward(positions)
        
        # 4. 向量相加！
        # 注意：NumPy 会自动利用 Broadcasting 广播机制，把一维的 pos_emb_out 复制并加到每一个 Batch 上。
        out = tok_emb_out + pos_emb_out
        
        return out
    def backward(self, grad_output):
        """
        反向传播核心：分离并传递梯度
        :param grad_output: 从 Attention 层传回来的总误差梯度，Shape: (batch_size, seq_len, embed_dim)
        """
        # 数学原理：(A + B) 的导数是 1。
        # 所以 Token 和 Position 分别独立接收一份一模一样的总梯度。
        
        # 1. 传给 Token Embedding：直接传
        self.token_emb.backward(grad_output)
        
        # 2. 传给 Position Embedding：需要沿着 batch_size 维度求和！
        # 因为在前向传播时，位置向量被广播到了所有的 Batch 上 (比如 Batch 里有 4 句话，第 0 个位置的向量被用了 4 次)。
        # 根据链式法则，被复制的变量的梯度，等于所有分支梯度的总和。
        grad_pos_emb = np.sum(grad_output, axis=0)
        self.pos_emb.backward(grad_pos_emb)

