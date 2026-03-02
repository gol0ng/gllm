import numpy as np
import gzip
import os


class MNISTDataLoader:
    """MNIST 数据加载器"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _read_idx_images(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, num_rows, num_cols)
        return images

    def _read_idx_labels(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def load_train(self):
        train_images = self._read_idx_images(
            os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')
        )
        train_labels = self._read_idx_labels(
            os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')
        )
        return train_images, train_labels

    def load_test(self):
        test_images = self._read_idx_images(
            os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')
        )
        test_labels = self._read_idx_labels(
            os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')
        )
        return test_images, test_labels


class FNN:
    """前馈神经网络 (Feedforward Neural Network)"""

    def __init__(self, input_size, hidden_sizes, output_size, lr=0.01):
        """
        初始化 FNN

        Args:
            input_size: 输入层大小 (784 for MNIST)
            hidden_sizes: 隐藏层大小列表, e.g. [256, 128]
            output_size: 输出层大小 (10 for MNIST)
            lr: 学习率
        """
        self.lr = lr
        self.layers = []
        self.activations = []

        # 构建网络结构: input -> hidden1 -> hidden2 -> ... -> output
        sizes = [input_size] + hidden_sizes + [output_size]

        # 初始化权重和偏置 (He 初始化)
        for i in range(len(sizes) - 1):
            # He 初始化: W ~ N(0, sqrt(2/fan_in))
            fan_in = sizes[i]
            W = np.random.randn(fan_in, sizes[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros(sizes[i + 1])
            self.layers.append({'W': W, 'b': b})
            # 隐藏层使用 ReLU, 输出层使用 Softmax
            if i < len(sizes) - 2:
                self.activations.append('relu')
            else:
                self.activations.append('softmax')

    def relu(self, x):
        """ReLU 激活函数"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU 导数"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax 激活函数 (数值稳定)"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        前向传播

        Args:
            X: 输入数据 (batch_size, input_size)

        Returns:
            输出概率 (batch_size, output_size)
        """
        self.cache = {'X': [], 'Z': []}

        A = X

        for i, layer in enumerate(self.layers):
            self.cache['X'].append(A)  # 存储每层的输入
            Z = A @ layer['W'] + layer['b']
            self.cache['Z'].append(Z)

            if self.activations[i] == 'relu':
                A = self.relu(Z)
            else:
                A = self.softmax(Z)

        self.cache['X'].append(A)  # 存储输出层输出

        return A

    def compute_loss(self, y_pred, y_true):
        """
        计算交叉熵损失

        Args:
            y_pred: 预测概率 (batch_size, 10)
            y_true: 真实标签 (batch_size,)

        Returns:
            损失值
        """
        batch_size = y_pred.shape[0]
        # 交叉熵: -sum(y_true * log(y_pred)) / batch_size
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        log_likelihood = -np.log(y_pred[np.arange(batch_size), y_true])
        loss = np.sum(log_likelihood) / batch_size
        return loss

    def backward(self, y_true):
        """
        反向传播

        Args:
            y_true: 真实标签 (batch_size,)
        """
        batch_size = self.cache['X'][0].shape[0]

        # 输出层梯度 (Softmax + CrossEntropy)
        y_true_onehot = np.zeros_like(self.cache['X'][-1])
        y_true_onehot[np.arange(batch_size), y_true] = 1

        dZ = self.cache['X'][-1] - y_true_onehot

        # 梯度裁剪，防止梯度爆炸
        dZ = np.clip(dZ, -5, 5)

        # 反向传播
        for i in range(len(self.layers) - 1, -1, -1):
            A_prev = self.cache['X'][i]

            # 梯度
            dW = A_prev.T @ dZ
            db = np.sum(dZ, axis=0)

            # 梯度裁剪
            dW = np.clip(dW, -5, 5)
            db = np.clip(db, -5, 5)

            # 更新权重
            self.layers[i]['W'] -= self.lr * dW / batch_size
            self.layers[i]['b'] -= self.lr * db / batch_size

            # 计算上一层的梯度
            if i > 0:
                dA_prev = dZ @ self.layers[i]['W'].T
                Z_prev = self.cache['Z'][i - 1]
                dZ = dA_prev * self.relu_derivative(Z_prev)
                dZ = np.clip(dZ, -5, 5)

    def train(self, X, y, epochs=10, batch_size=64, verbose=True):
        """
        训练神经网络

        Args:
            X: 训练数据 (n_samples, 784)
            y: 训练标签 (n_samples,)
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0
            n_batches = 0

            # Mini-batch 训练
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 计算损失
                loss = self.compute_loss(y_pred, y_batch)
                total_loss += loss
                n_batches += 1

                # 反向传播
                self.backward(y_batch)

            avg_loss = total_loss / n_batches

            if verbose and (epoch + 1) % 1 == 0:
                y_pred_train = self.predict(X)
                accuracy = np.mean(y_pred_train == y)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        """
        预测类别

        Args:
            X: 输入数据

        Returns:
            预测标签
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 测试数据
            y: 测试标签

        Returns:
            准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def preprocess_data(images, labels):
    """
    预处理数据: 归一化 + 展平

    Args:
        images: 图片数据 (n, 28, 28)
        labels: 标签 (n,)

    Returns:
        X: 归一化并展平的数据 (n, 784)
        y: 标签
    """
    # 展平: (n, 28, 28) -> (n, 784)
    X = images.reshape(images.shape[0], -1)
    # 归一化: 0-255 -> 0-1
    X = X.astype(np.float32) / 255.0
    return X, labels


if __name__ == '__main__':
    # 1. 加载数据
    data_dir = os.path.join(os.path.dirname(__file__), 'MNIST')
    loader = MNISTDataLoader(data_dir)

    train_images, train_labels = loader.load_train()
    test_images, test_labels = loader.load_test()

    print(f"训练集: {train_images.shape[0]} 样本")
    print(f"测试集: {test_images.shape[0]} 样本")

    # 2. 预处理数据
    X_train, y_train = preprocess_data(train_images, train_labels)
    X_test, y_test = preprocess_data(test_images, test_labels)

    print(f"预处理后 X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 3. 创建并训练 FNN
    # 输入 784 维, 隐藏层 [256, 128], 输出 10 类
    model = FNN(input_size=784, hidden_sizes=[256, 128], output_size=10, lr=0.01)

    print("\n开始训练...")
    model.train(X_train, y_train, epochs=10, batch_size=128, verbose=True)

    # 4. 评估模型
    test_accuracy = model.evaluate(X_test, y_test)
    print(f"\n测试集准确率: {test_accuracy:.4f}")
