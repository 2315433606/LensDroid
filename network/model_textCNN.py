# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from Define import WORDLEN

class TextCNN(nn.Module):
    # filter_num*filter_sizes和embedding_dim吃显存
    def __init__(self,num_classes,embedding_dim,
                filter_num=32,filter_sizes=[2,3,4,5]):
        super(TextCNN, self).__init__()


        #构建卷积层
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        # self.dropout = nn.Dropout(dropout)
        #分类器层
        self.classifier = nn.Linear(len(filter_sizes)*filter_num, num_classes)

    def forward(self, x):
        # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
        
        x = x.unsqueeze(1)
        
        # 经过卷积运算,x中每个运算结果维度为(batch_size, filter_num, 1, w[i]=vocab_size-filter_sizes[i])
        # batch_size*filter_num*9/8/7/6(vocab_size-1~vocab_size-4)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # 经过最大池化层,维度变为(batch_size, filter_num)
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]

        # 将不同卷积核提取的特征组合起来,维度变为(batch, filter_num*len(filter_sizes))
        x = torch.cat(x, 1)
        # dropout层
        # x = self.dropout(x)

        # 全连接层
        logits = self.classifier(x)

        return logits

def create_textCNN(num_classes):
    with open('apkPreprocess/script_preprocess/word2id.json', "r") as f:
        Dict_word2id = json.load(f)
    vocab_size=len(Dict_word2id)
    model = TextCNN(num_classes=num_classes,embedding_dim=WORDLEN*vocab_size)
    return model