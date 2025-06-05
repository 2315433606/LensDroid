import os
import itertools
import numpy as np
import pandas as pd
import json

#将opcodeSeq.txt转化为.npy

def opcodeSeq2IdSeq(opcodeSeqFn,Dict_word2id):
    print(opcodeSeqFn)
    # apkPreprocess/TrainData/data2021/ApkPreProcessResTest/benign/agenciaeficaz.fibranetnovo/agenciaeficaz.fibranetnovo.opcodeSeq.txt
    wordList=[]
    with open(opcodeSeqFn) as f:
        for word in f.read():
            #print(word) 
            wordList.append(Dict_word2id[word])
    return wordList

if __name__ == "__main__":
    opcode8=["M","R","G","I","T","P","V","|"]

    Dict_word2id=dict.fromkeys(opcode8,0)
    for idx,word in enumerate(opcode8):
        Dict_word2id[word]=idx

    with open("apkPreprocess/script_preprocess/word2id.json",'w',encoding='utf8') as fp:
        json.dump(Dict_word2id,fp,ensure_ascii=False)
    print(Dict_word2id)
    rootName="apkPreprocess/TrainData/data_Reflection/ApkPreProcessRes"
    wordList=[]

    for category in os.listdir(rootName):
        if category=='malware':
            pathName=os.path.join(rootName,category)
            print(pathName)
            for root, dirs, files in os.walk(pathName): #root下的子dics和files
                for file in files:
                    if file.split('.')[-2]=="opcodeSeq": 
                        opcodeSeqFn=os.path.join(root,file)
                        wordList=opcodeSeq2IdSeq(opcodeSeqFn,Dict_word2id)
                        wordList=np.array(wordList,dtype="int64")
                        
                        arrayFn=os.path.join(root,"IdSeq_Array")
                        #保存路径为：apkPreprocess/TrainData/data2021/ApkPreProcessResTest/benign/agenciaeficaz.fibranetnovo/IdSeq_Array.npy
                        np.save(arrayFn,wordList)
