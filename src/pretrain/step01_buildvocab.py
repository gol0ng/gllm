import pandas as pd
import numpy as np
import json
import os
from collections import Counter

# 利用数据集前n条数据构建词汇表
def build_character_vocab(file_path,text_column,max_rows=None,min_freq=3):
    '''
    file_path:数据集文件位置
    text_column:parquet文件的列名
    max_rows:要读取数据集文件的行数,None表示全部读取
    min_freq:用于词频统计,去掉所有低于min_freq的词
    '''
    print("读取数据集文件")
    try:
        df = pd.read_parquet(file_path,engine="pyarrow")
    except Exception as e:
        print("文件读取失败。错误信息:",e)
        return None
    
    if max_rows is not None:
        df = df.head(max_rows) # 读取max_rows行的数据

    text_list = df[text_column].dropna().tolist()
    print("成功读取了{}条数据".format(len(text_list)))
    
    # 使用Counter统计所有字符出现的次数
    char_counter = Counter()
    for text in text_list:
        char_counter.update(list(text))

    print("共发现{}个不同的字符".format(len(char_counter)))

    print("开始过滤出现次数小于{}的词".format(min_freq))
    valid_chars = [ch for ch,freq in char_counter.items() if freq >= min_freq]

    # 对挑选出来的字符进行排序，确保每次生成的字符ID固定
    chars = sorted(valid_chars)

    # 插入预训练必要的特殊token
    special_tokens = ["<pad>", "<unk>", "<eos>"]
    vocab_list = special_tokens + chars

    print("经过词频过滤后的最终的词表大小为{}".format(len(vocab_list)))

    return vocab_list

def save_vocab(vocab_list,file_path = "vocab.json"):
    print("开始保存词汇表")
    with open(file_path,"w",encoding="utf-8") as f:
        json.dump(vocab_list,f,ensure_ascii=False, indent=2)
    f.close()
    print("词汇表保存成功")




    



    