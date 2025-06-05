import os
import numpy as np
import pandas as pd
import json
import re
import logging
import time
import copy

def isSensitive(str,nameList,Dict_caller_level,Dict_level_id):
    levelVec=[]
    if re.match("new",str):
        return levelVec

    package=""
    method=""
    matchObj=re.match(r"<(.*): .* (.*)\(.*\)>",str,re.I)
    if matchObj:
        package=matchObj.group(1)
        method=matchObj.group(2)

    for line in nameList:
        # "query.android.content": "signature",
        words=line.split('.',1)
        if re.match(words[1],package,re.I)!=None and words[0].lower()==method.lower():
        # if re.search(words[0],package,re.I)!=None and re.search(words[1],package,re.I)!=None and re.search(words[2],method,re.I)!=None:
            tmpList=Dict_caller_level[line].split('|')
            for it in tmpList:
                if it!="normal":
                    levelVec.append(Dict_level_id[it])
            # print(str)
            # print(package)
            # print(words[1])
            # print(method)
            # print(words[0])
            # print(line)
            # print(Dict_caller_level[line])
            break
    return levelVec


# demo
if __name__ == "__main__":
    Dict_caller_level={}
    with open("FlowDroid/Manifest_script/Dict_caller_level.json", "r") as f:
        Dict_caller_level = json.load(f)
    nameList=list(Dict_caller_level.keys())
    nameList.sort()

    Dict_level_id={}
    with open("FlowDroid/Manifest_script/Dict_level_id.json", "r") as f:
        Dict_level_id = json.load(f)

    # demo    
    # cgPath="FlowDroid/call_graph.txt"
    ApkRootName="apkPreprocess/TrainData/data2021/ApkTrainSet"
    cgRootName="apkPreprocess/TrainData/data2021/ApkPreProcessRes"
    cnt=0
    for category in os.listdir(ApkRootName):
        pathName=os.path.join(ApkRootName,category)
        for fileName in os.listdir(pathName):
            cnt+=1
            print(cnt)
            apiList=[]
            srcList=[]
            tgtList=[]
            # senIdList=[]
            Dict_senID_vec={}

            apkName='.'.join(fileName.split('.')[:-1])
            curPath=os.path.join(cgRootName,category,apkName)
            cgPath=os.path.join(curPath,"call_graph.txt")
            print(cgPath)

            if not os.path.exists(cgPath):#如果路径不存在
                print("Not exists\n")
                continue

            if os.path.exists(os.path.join(curPath,"apiLen.txt")):
                continue
            
            print(time.ctime())
            with open(cgPath) as f:
                for line in f:
                    [src,tgt]=line.strip().split('——>')
                    # 下标就是对应id
                    srcVisitFlag=1
                    tgtVisitFlag=1
                    if src not in apiList:
                        srcVisitFlag=0
                        apiList.append(src) 
                    if tgt not in apiList:
                        tgtVisitFlag=0
                        apiList.append(tgt) 
                    srcId=apiList.index(src)
                    tgtId=apiList.index(tgt)
                    srcList.append(srcId)
                    tgtList.append(tgtId)

                    if srcVisitFlag==0:
                        levelVec=isSensitive(src,nameList,Dict_caller_level,Dict_level_id)
                        if len(levelVec)!=0:
                            # senIdList.append(srcId)
                            if srcId not in Dict_senID_vec.keys():
                                Dict_senID_vec[srcId]=levelVec
                            else:
                                logging.info("error")

                    if tgtVisitFlag==0:
                        levelVec=isSensitive(tgt,nameList,Dict_caller_level,Dict_level_id)
                        if len(levelVec)!=0:
                            # senIdList.append(tgtId)
                            if tgtId not in Dict_senID_vec.keys():
                                Dict_senID_vec[tgtId]=levelVec
                            else:
                                logging.info("error")
            # print(srcList)
            # print(tgtList)
            # print(Dict_senID_vec)
            # np.save(os.path.join(curPath,'apiNp.npy'),np.array(apiList))
            with open(os.path.join(curPath,'apiLen.txt'),'w') as f:
                f.write(str(len(apiList)))

            np.save(os.path.join(curPath,'srcNp.npy'),np.array(srcList))
            np.save(os.path.join(curPath,'tgtNp.npy'),np.array(tgtList))

            json_str=json.dumps(Dict_senID_vec, indent=1, sort_keys=True)
            with open(os.path.join(curPath,"Dict_senID_vec.json"), 'w') as json_file:
                json_file.write(json_str)

            # print(np.load(os.path.join(curPath,'srcNp.npy')))
            # print(np.load(os.path.join(curPath,'tgtNp.npy')))

            # print(len(apiList))
            sensitiveSet=list(map(int, Dict_senID_vec.keys()))
            # apiSenIdxList=[]
            oldId2newId={}
            # newId2oldId=[]
            srcSenList=[]
            tgtSenList=[]
            # nodeSet=set()
            for i in range(len(srcList)):
                if srcList[i] in sensitiveSet or tgtList[i] in sensitiveSet:
                    # apiSenIdxList.append(i)
                    srcSenList.append(srcList[i])
                    tgtSenList.append(tgtList[i])
                    if srcList[i] not in oldId2newId.keys():
                        oldId2newId[srcList[i]]=len(oldId2newId)
                        # newId2oldId.append(srcList[i])
                    if tgtList[i] not in oldId2newId.keys():
                        oldId2newId[tgtList[i]]=len(oldId2newId)
                        # newId2oldId.append(tgtList[i])
            srcNewList=copy.deepcopy(srcSenList)
            tgtNewList=copy.deepcopy(tgtSenList)
            for i in range(len(srcSenList)):
                srcNewList[i]=oldId2newId[srcNewList[i]]    
                tgtNewList[i]=oldId2newId[tgtNewList[i]] 
                
            np.save(os.path.join(curPath,'srcSenNp.npy'),np.array(srcNewList))
            np.save(os.path.join(curPath,'tgtSenNp.npy'),np.array(srcNewList))




