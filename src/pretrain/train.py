import numpy as np
import time
import os
import json
from src.pretrain import step01_buildvocab, step02_dataloader, step03_embeddinglayer
from src.pretrain.step04_mutilheadattentionlayer import MultiHeadAttention
from src.pretrain.step05_normlayer import LayerNorm
from src.pretrain.step06_feedwardlayer import FFN
from src.pretrain.step07_loss import LMHeadAndLoss


class NanoGPT:
    """完整的 NanoGPT 模型"""
    def __init__(self, vocab_size, block_size, embed_dim, num_layers, n_head):
        print(f"初始化 NanoGPT: vocab={vocab_size}, block={block_size}, dim={embed_dim}, layers={num_layers}, heads={n_head}")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.n_head = n_head

        # 1. 词嵌入 + 位置编码
        self.input_emb = step03_embeddinglayer.GllmInputEmbedding(vocab_size, block_size, embed_dim)

        # 2. 多个 Transformer Block
        self.blocks = []
        for _ in range(num_layers):
            block = {
                'attn': MultiHeadAttention(n_head=n_head, embed_dim=embed_dim),
                'ln1': LayerNorm(embed_dim),
                'ffn': FFN(embed_dim, embed_dim * 4),
                'ln2': LayerNorm(embed_dim)
            }
            self.blocks.append(block)

        # 3. 输出层
        self.lm_head = LMHeadAndLoss(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        """前向传播"""
        out = self.input_emb.forward(x)

        for block in self.blocks:
            residual = out
            attn_out = block['attn'].forward(out)
            out = block['ln1'].forward(attn_out + residual)

            residual = out
            ff_out = block['ffn'].forward(out)
            out = block['ln2'].forward(ff_out + residual)

        logits, loss = self.lm_head.forward(out, targets)
        return logits, loss

    def backward(self):
        """反向传播"""
        grad = self.lm_head.backward()

        for block in reversed(self.blocks):
            grad_ln2 = block['ln2'].backward(grad)
            grad = block['ffn'].backward(grad_ln2)

            grad_ln1 = block['ln1'].backward(grad)
            grad = block['attn'].backward(grad_ln1)

        self.input_emb.backward(grad)

    def get_all_params(self):
        """获取所有参数"""
        params = {
            'config': {
                'vocab_size': self.vocab_size,
                'block_size': self.block_size,
                'embed_dim': self.embed_dim,
                'num_layers': self.num_layers,
                'n_head': self.n_head
            },
            'token_emb': self.input_emb.token_emb.weight,
            'pos_emb': self.input_emb.pos_emb.weight,
            'blocks': [],
            'lm_head_W': self.lm_head.W_out,
            'lm_head_b': self.lm_head.b_out
        }

        for block in self.blocks:
            params['blocks'].append({
                'W_q': block['attn'].W_q,
                'W_k': block['attn'].W_k,
                'W_v': block['attn'].W_v,
                'W_o': block['attn'].W_o,
                'ln1_gamma': block['ln1'].gamma,
                'ln1_beta': block['ln1'].beta,
                'ffn_W1': block['ffn'].W1,
                'ffn_b1': block['ffn'].b1,
                'ffn_W2': block['ffn'].W2,
                'ffn_b2': block['ffn'].b2,
                'ln2_gamma': block['ln2'].gamma,
                'ln2_beta': block['ln2'].beta
            })

        return params

    def set_params(self, params):
        """设置所有参数"""
        self.input_emb.token_emb.weight = params['token_emb']
        self.input_emb.pos_emb.weight = params['pos_emb']

        for i, block in enumerate(self.blocks):
            block['attn'].W_q = params['blocks'][i]['W_q']
            block['attn'].W_k = params['blocks'][i]['W_k']
            block['attn'].W_v = params['blocks'][i]['W_v']
            block['attn'].W_o = params['blocks'][i]['W_o']
            block['ln1'].gamma = params['blocks'][i]['ln1_gamma']
            block['ln1'].beta = params['blocks'][i]['ln1_beta']
            block['ffn'].W1 = params['blocks'][i]['ffn_W1']
            block['ffn'].b1 = params['blocks'][i]['ffn_b1']
            block['ffn'].W2 = params['blocks'][i]['ffn_W2']
            block['ffn'].b2 = params['blocks'][i]['ffn_b2']
            block['ln2'].gamma = params['blocks'][i]['ln2_gamma']
            block['ln2'].beta = params['blocks'][i]['ln2_beta']

        self.lm_head.W_out = params['lm_head_W']
        self.lm_head.b_out = params['lm_head_b']


def save_checkpoint(model, epoch, step, loss, checkpoint_dir='checkpoints'):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    params = model.get_all_params()
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'params': params
    }

    # 保存最新检查点
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.npz')
    np.savez_compressed(checkpoint_path, **checkpoint)

    # 保存 epoch 检查点
    epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.npz')
    np.savez_compressed(epoch_path, **checkpoint)

    print(f"  [保存] Epoch {epoch}, Step {step}, Loss: {loss:.4f} -> {epoch_path}")

    return checkpoint_path


def load_checkpoint(checkpoint_path, model):
    """加载检查点"""
    print(f"  [加载] 从 {checkpoint_path} 恢复训练...")
    checkpoint = np.load(checkpoint_path, allow_pickle=True)

    model.set_params(checkpoint['params'].item())

    epoch = int(checkpoint['epoch'])
    step = int(checkpoint['step'])
    loss = float(checkpoint['loss'])

    print(f"  [恢复] Epoch {epoch}, Step {step}, Loss: {loss:.4f}")

    return epoch, step, loss


def sgd_step(model, lr=0.01, clip_val=1.0):
    """SGD 优化器"""
    model.input_emb.token_emb.weight -= lr * np.clip(model.input_emb.token_emb.grad_weight, -clip_val, clip_val)
    model.input_emb.pos_emb.weight -= lr * np.clip(model.input_emb.pos_emb.grad_weight, -clip_val, clip_val)

    for block in model.blocks:
        block['attn'].W_q -= lr * np.clip(block['attn'].grad_W_q, -clip_val, clip_val)
        block['attn'].W_k -= lr * np.clip(block['attn'].grad_W_k, -clip_val, clip_val)
        block['attn'].W_v -= lr * np.clip(block['attn'].grad_W_v, -clip_val, clip_val)
        block['attn'].W_o -= lr * np.clip(block['attn'].grad_W_o, -clip_val, clip_val)
        block['ln1'].gamma -= lr * np.clip(block['ln1'].grad_gamma, -clip_val, clip_val)
        block['ln1'].beta -= lr * np.clip(block['ln1'].grad_beta, -clip_val, clip_val)
        block['ffn'].W1 -= lr * np.clip(block['ffn'].grad_W1, -clip_val, clip_val)
        block['ffn'].b1 -= lr * np.clip(block['ffn'].grad_b1, -clip_val, clip_val)
        block['ffn'].W2 -= lr * np.clip(block['ffn'].grad_W2, -clip_val, clip_val)
        block['ffn'].b2 -= lr * np.clip(block['ffn'].grad_b2, -clip_val, clip_val)
        block['ln2'].gamma -= lr * np.clip(block['ln2'].grad_gamma, -clip_val, clip_val)
        block['ln2'].beta -= lr * np.clip(block['ln2'].grad_beta, -clip_val, clip_val)

    model.lm_head.W_out -= lr * np.clip(model.lm_head.grad_W_out, -clip_val, clip_val)
    model.lm_head.b_out -= lr * np.clip(model.lm_head.grad_b_out, -clip_val, clip_val)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 NanoGPT 预训练开始 (支持断点续训)")
    print("=" * 60)

    # ========== 1. 超参数配置 ==========
    DATA_FILE = './src/dataset/pre_data.parquet'
    CHECKPOINT_DIR = 'checkpoints'

    # 模型参数
    VOCAB_SIZE = 10000
    BLOCK_SIZE = 64
    EMBED_DIM = 256
    NUM_LAYERS = 6
    N_HEAD = 8

    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MAX_STEPS = 100000
    MAX_EPOCHS = 10
    CLIP_VAL = 1.0
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 1000  # 每1000步保存一次
    LOSS_STOP_THRESHOLD = 0.5  # loss 低于此值停止训练

    # ========== 2. 准备数据 ==========
    print("\n[1/5] 准备数据...")

    try:
        stoi, itos, vocab_size = step02_dataloader.load_vocab('vocab.json')
        print(f"词汇表加载成功: {vocab_size} 个词")
    except:
        vocab = step01_buildvocab.build_character_vocab(DATA_FILE, 'text', VOCAB_SIZE, 3)
        step01_buildvocab.save_vocab(vocab)
        stoi, itos, vocab_size = step02_dataloader.load_vocab('vocab.json')
        print(f"词汇表构建成功: {vocab_size} 个词")

    data_array = step02_dataloader.create_dataset(DATA_FILE, stoi, 'text', None)
    print(f"数据集大小: {len(data_array)} tokens")

    dataloader = step02_dataloader.DataLoader(data_array, BATCH_SIZE, BLOCK_SIZE)

    # 计算每个 epoch 的步数
    steps_per_epoch = len(data_array) // (BATCH_SIZE * BLOCK_SIZE)
    print(f"每个 Epoch 约 {steps_per_epoch} 步")

    # ========== 3. 初始化/加载模型 ==========
    print("\n[2/5] 初始化模型...")
    model = NanoGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        n_head=N_HEAD
    )

    # 计算参数量
    total_params = vocab_size * EMBED_DIM + BLOCK_SIZE * EMBED_DIM
    total_params += NUM_LAYERS * (4 * EMBED_DIM * EMBED_DIM + 2 * EMBED_DIM + 2 * EMBED_DIM * 4 * EMBED_DIM)
    total_params += EMBED_DIM * vocab_size + vocab_size
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # ========== 4. 检查断点续训 ==========
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    latest_checkpoint = os.path.join(CHECKPOINT_DIR, 'latest.npz')
    if os.path.exists(latest_checkpoint):
        print("\n[3/5] 检测到检查点，准备恢复训练...")
        start_epoch, global_step, best_loss = load_checkpoint(latest_checkpoint, model)
        start_epoch += 1  # 从下一个 epoch 继续
    else:
        print("\n[3/5] 未检测到检查点，从头开始训练")

    # ========== 5. 训练循环 ==========
    print("\n[4/5] 开始训练...")
    print("-" * 60)

    start_time = time.time()
    losses = []
    epoch_losses = []
    should_stop = False

    for epoch in range(start_epoch, MAX_EPOCHS):
        if should_stop:
            break

        epoch_loss_sum = 0
        epoch_steps = 0

        # 每次 epoch 打乱数据
        dataloader = step02_dataloader.DataLoader(data_array, BATCH_SIZE, BLOCK_SIZE)

        while True:
            # 获取 batch
            xb, yb = dataloader.get_batch()
            if xb is None:
                break

            # 前向传播
            logits, loss = model.forward(xb, yb)

            # 反向传播
            model.backward()

            # 更新参数
            sgd_step(model, lr=LEARNING_RATE, clip_val=CLIP_VAL)

            global_step += 1

            if loss is not None:
                losses.append(loss)
                epoch_loss_sum += loss
                epoch_steps += 1

            # 打印日志
            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0

                print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Step {global_step} | "
                      f"Loss: {avg_loss:.4f} | Speed: {steps_per_sec:.1f} steps/s | "
                      f"Time: {elapsed/60:.1f}min")

            # 保存检查点
            if global_step % SAVE_INTERVAL == 0:
                current_loss = np.mean(losses[-100:]) if losses else 0
                save_checkpoint(model, epoch, global_step, current_loss, CHECKPOINT_DIR)

            # 检查 loss 是否足够低
            if len(losses) > 0:
                current_loss = np.mean(losses[-100:])
                if current_loss < LOSS_STOP_THRESHOLD:
                    print(f"\n🎉 Loss 降至 {current_loss:.4f} < {LOSS_STOP_THRESHOLD}，停止训练！")
                    save_checkpoint(model, epoch, global_step, current_loss, CHECKPOINT_DIR)
                    should_stop = True
                    break

        # Epoch 结束
        avg_epoch_loss = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0
        epoch_losses.append(avg_epoch_loss)
        best_loss = min(best_loss, avg_epoch_loss)

        print(f"\n>>> Epoch {epoch+1} 完成 | Avg Loss: {avg_epoch_loss:.4f} | Best: {best_loss:.4f}")

        # 每个 epoch 保存一次
        save_checkpoint(model, epoch + 1, global_step, avg_epoch_loss, CHECKPOINT_DIR)

    # ========== 6. 保存最终模型 ==========
    print("\n[5/5] 训练完成!")
    print("=" * 60)
    print(f"总训练步数: {global_step}")
    print(f"最终 Loss: {losses[-1] if losses else 'N/A':.4f}")
    print(f"最佳 Loss: {best_loss:.4f}")
    print(f"总训练时间: {(time.time() - start_time) / 3600:.2f} 小时")
    print(f"检查点保存位置: {CHECKPOINT_DIR}/")
    print("=" * 60)
