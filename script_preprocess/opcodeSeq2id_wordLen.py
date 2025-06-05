import os
import itertools
import numpy as np
import pandas as pd
import json

from Constant import WORDLEN

def opcodeSeq2IdSeq(opcodeSeqFn,Dict_word2id):
    print(opcodeSeqFn)
    wordList=[]
    with open(opcodeSeqFn) as f:
        words=f.read()
        while len(words)%WORDLEN!=0:
            words+="#"
    
        for word in [words[i:i+WORDLEN] for i in range(0, len(words), WORDLEN)]:
            # print(word)
            wordList.append(Dict_word2id[word])
    return wordList

if __name__ == "__main__":
    opcode9=["#","|","M","R","G","I","T","P","V"]
    cols=[]
    for item in itertools.product(opcode9,repeat=WORDLEN):
        cols.append((''.join(item)))
    # print(cols,len(cols))

    # Dict_word2id=dict.fromkeys(cols,0)
    # for idx,word in enumerate(cols):
    #     Dict_word2id[word]=idx

    # with open("apkPreprocess/script_preprocess/word2id.json",'w',encoding='utf8') as fp:
    #     json.dump(Dict_word2id,fp,ensure_ascii=False)

    # print(Dict_word2id)

    # categoryDict={"malware":0,"benign":1}

    # rootName="apkPreprocess/TrainData/data2018/ApkPreProcessRes"

    # wordList=[]




    # for category in os.listdir(rootName):
    #     pathName=os.path.join(rootName,category)
    #     print(pathName)
    #     for root, dirs, files in os.walk(pathName): #root下的子dics和files
    #         for file in files:
    #             if file.split('.')[-2]=="opcodeSeq":
    #                 opcodeSeqFn=os.path.join(root,file)
    #                 wordList=opcodeSeq2IdSeq(opcodeSeqFn,Dict_word2id)
    #                 wordList=np.array(wordList,dtype="int64")
                    
    #                 arrayFn=os.path.join(root,"IdSeq_Array")
    #                 np.save(arrayFn,wordList)
