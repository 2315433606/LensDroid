import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time

from Constant import WORDLEN

if __name__ == "__main__":
    print(f"WORDLEN为：{WORDLEN}") #4  WORDLEN：滑动窗口的长度
    #years=['data2018','data2019','data2020','data2021','data2022',"Drebin"]
    years=['data_Reflection']
    for year in years:
        rootName="apkPreprocess/TrainData/{}/ApkPreProcessRes".format(year)
        
        # print(time.ctime())
        for category in os.listdir(rootName):
            if category=='malware':
                pathName=os.path.join(rootName,category)
                for root, dirs, files in os.walk(pathName): #root下的子dics和files
                    for file in files:
                        if file=="IdSeq_Array.npy":
                            #IdSeqFn="apkPreprocess/TrainData/data2021/ApkPreProcessResTest/benign/agenciaeficaz.fibranetnovo/IdSeq_Array.npy"
                            IdSeqFn=os.path.join(root,file)
                            code=torch.tensor(np.load(IdSeqFn))#code 的值范围是从 0 到 7(表示八种不同的操作码)
                            print(time.ctime())
                            print(IdSeqFn)
                            if code.nelement() == 0:  # 检查code是否为空
                                print(f"{root}:code张量为空，跳过此文件。")
                            else:
                                tensor_x = F.one_hot(code, num_classes=8) # tensor_x的维度为(seqLen-wordLen+1, 8)  zjl:tensor_x的维度为(seqLen,8)
                                # overlapping:在序列数据中捕捉局部上下文信息，提高模型对序列中相邻元素之间关系的感知能力
                                row,col=tensor_x.size()#row是~.pcodeSeq.txt(IdSeq_Array.npy)的大小,col为8
                                
                                oneHotEmbedding=tensor_x[WORDLEN-1:]

                                #通过每次循环的操作，tmp 的长度始终是 seqLen-WORDLEN+1，保证了拼接后的 oneHotEmbedding 的长度不变，
                                #只在列方向上增加宽度，最终得到的 oneHotEmbedding 的形状为 (seqLen-WORDLEN+1, 8 * WORDLEN)

                                for i in range(WORDLEN):
                                    if i==0:
                                        # print(oneHotEmbedding.size())
                                        continue
                                    tmp=tensor_x[WORDLEN-i-1:-i]
                                    oneHotEmbedding=torch.cat((tmp,oneHotEmbedding),dim=1)
                                    #print(oneHotEmbedding.size())
                                torch.save(oneHotEmbedding,os.path.join(root,"opcodeOneHot.pth"))

