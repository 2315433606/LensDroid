import os
import itertools
import numpy as np
import pandas as pd
import logging
import json
import time

# 和Define对齐
NGRAM=3

#每个文件夹一个json,整体一个N-gram的csv
def getNgramVector(opcodeSeqFn,NgramDict):
    print(opcodeSeqFn)
    with open(opcodeSeqFn) as f:
        seqList=f.read().strip().split('|')
        for subSeq in seqList:
            length=len(subSeq)
            if length<NGRAM:
                continue
            for start in range(length-NGRAM+1):
                end=start+NGRAM
                tmp=subSeq[start:end]
                if tmp in NgramDict:
                    NgramDict[subSeq[start:end]]=1
                    # NgramDict[subSeq[start:end]]+=1
                else:
                    print("****Error****")  
    return NgramDict

if __name__ == "__main__":
    opcode7=["M","R","G","I","T","P","V"]
    cols=[]
    for item in itertools.product(opcode7,repeat=NGRAM):
        cols.append((''.join(item)))
    print(cols,len(cols))

    categoryDict={"benign":0,"malware":1}

    NgramDict={}
    L_NDict=[]
    fnList=[]
    ctgList=[]

    rootName="apkPreprocess/TrainData"
    years=['AndroZoo4']
    for year in years:
        dataPath=os.path.join(rootName,year,"ApkPreProcessRes")
        for category in os.listdir(dataPath):
            pathName=os.path.join(dataPath,category)
            print(time.ctime(),pathName)
            for root, dirs, files in os.walk(pathName): #root下的子dics和files
                for file in files:
                    if file.split('.')[-1]=="txt" and file.split('.')[-2]=="opcodeSeq":
                        NgramDict=dict.fromkeys(cols,0)
                        opcodeSeqFn=os.path.join(root,file)
                        NgramDict=getNgramVector(opcodeSeqFn,NgramDict)

                        # with open(os.path.join(root,"ngramEmbedding.json"),'w',encoding='utf8')as fp:
                        #     json.dump(NgramDict,fp,ensure_ascii=False)

                        L_NDict.append(NgramDict)
                        fnList.append(file)
                        ctgList.append(categoryDict[category])
                        # print(NgramDict)
    df=pd.DataFrame(L_NDict)
    df['fn']=fnList
    df['category']=ctgList
    df.to_csv("apkPreprocess/script_preprocess/{}Gram_EmbeddingAndroZoo4.csv".format(NGRAM))

    