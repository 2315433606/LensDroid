from PIL import Image
import torch
from torch.utils.data import Dataset
import json 
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import dgl
import copy

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, apks_path: list, apks_class: list, transform=None):
        self.apks_path = apks_path
        self.apks_class = apks_class
        self.transform = transform

    def __len__(self):
        return len(self.apks_path)

    def __getitem__(self, item):
        apkPath = self.apks_path[item]
        label = self.apks_class[item]
        # print(apkPath)
        
        # image==>convnext
        img = Image.open(os.path.join(apkPath,"color.png"))
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(apkPath))
        if self.transform is not None:
            img = self.transform(img)

        # opcode==>textCnn
        code=torch.load(os.path.join(apkPath,"opcodeOneHot.pth"))
        #zjl：code = code.to(torch.float32)
        code=code.type(torch.FloatTensor)

        # api_graph、inputs==>GCN
        with open(os.path.join(apkPath,"apiLen.txt"), "r") as f:
            apiLen = f.read()
        srcList=np.load(os.path.join(apkPath,"srcNp.npy")).tolist()
        tgtList=np.load(os.path.join(apkPath,"tgtNp.npy")).tolist()
        Dict_senID_vec={}
        with open(os.path.join(apkPath,"Dict_senID_vec.json"), "r") as f:
            Dict_senID_vec = json.load(f)
        
        graph = dgl.graph((srcList,tgtList))
        #添加自环
        graph = dgl.add_self_loop(graph)
        LEVELNUM=15
        #int(apiLen) 是行数，LEVELNUM 是列数
        initEmbedding=torch.zeros(int(apiLen),LEVELNUM)
        
        for k,v in Dict_senID_vec.items():
            initEmbedding[int(k)][v]=1
        
        graph.ndata['feat']=initEmbedding

        return img, code, graph, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, codes, graphs, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)

        codes = pad_sequence(codes, batch_first = True, padding_value = 0 )

        graphs = dgl.batch(graphs)

        labels = torch.as_tensor(labels)
        return images, codes, graphs, labels
