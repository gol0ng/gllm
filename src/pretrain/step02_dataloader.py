import pandas as pd
import numpy as np
import json
import os

# 从本地加载build_vocab.py生成的vocab.json
def load_vocab(file_path="vocab.json"):
    print("开始加载词汇表")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到词表文件: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        vocab_list = json.load(f)
        
    stoi = {ch: i for i, ch in enumerate(vocab_list)}
    itos = {i: ch for i, ch in enumerate(vocab_list)}
    print("词汇表加载成功")
    return stoi, itos, len(vocab_list)

def encode_text(text, stoi):
    # print("词汇表编码器，将未知词用固定token:<unk>代替")
    unk_id = stoi.get("<unk>")
    return [stoi.get(c, unk_id) for c in text]

def create_dataset(parquet_path, stoi, text_column='text', max_rows=10000):
    '''
    刚开始做的时候我也在好奇为什么已经在build_vocab.py里面写过这几个参数了，这里还要写
    是因为我们在构建词汇表时不可能用整个数据集，我们只选取了前10000行
    而现在我们是为了构建预训练数据集，这里肯定是要全部数据的
    '''
    # 读取数据集
    df = pd.read_parquet(parquet_path)
    if max_rows is not None:
        df = df.head(max_rows)
        
    texts = df[text_column].dropna().tolist()
    eos_id = stoi.get("<eos>")
    
    all_tokens = [] 
    for text in texts:
        tokens = encode_text(text, stoi) # 把当前文章变成数字
        all_tokens.extend(tokens)        
        all_tokens.append(eos_id) # 将结束token放在text最后，告诉模型到这里就可以停止了
        
    # 把list变成np的array，效率更高
    data_array = np.array(all_tokens, dtype=np.int32)
    return data_array


class DataLoader:
    def __init__(self,data_array,batch_size = 64,block_size = 64):
        self.data = data_array
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_idx = len(data_array) - block_size -1 # 这个参数主要是因为我们在截取整个数据集最后一个输入时需要注意：要留出当前输入的输出，即id+1的长度
    
    def get_batch(self):
        ix = np.random.randint(0,self.max_idx,size=self.batch_size)

        x = np.empty((self.batch_size,self.block_size),dtype=np.int32) # 生成空的，高为batch_size，长为block_size的矩阵
        y = np.empty((self.batch_size,self.block_size),dtype=np.int32)

        for i,start_idx in enumerate(ix):
            x[i] = self.data[start_idx:start_idx+self.block_size] # 为刚刚的空矩阵填入数据
            y[i] = self.data[start_idx + 1 : start_idx + self.block_size + 1]
        
        return x,y
    

