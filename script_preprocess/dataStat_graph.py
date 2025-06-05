import os
import shutil
import re
import numpy as np
import time


if __name__ == "__main__":
    years=['Drebin']
    for year in years:
        print("year:",year)
        ApkRootName="apkPreprocess/TrainData/{}/ApkTrainSet".format(year)
        cgRootName="apkPreprocess/TrainData/{}/ApkPreProcessRes".format(year)
        smaliRootName="apkPreprocess/TrainData/{}/ApkSmaliFiles".format(year)
        for category in os.listdir(cgRootName):
            if year=="Drebin" and category=="malware":
                continue
            cnt=0
            sumNode=0
            sumEdge=0
            pathName=os.path.join(cgRootName,category)
            for fileName in os.listdir(pathName):
                cnt+=1
                # print("cnt:",cnt,time.ctime())
                cgPath=os.path.join(cgRootName,category,fileName)
                with open(os.path.join(cgPath,"apiLen.txt"), "r") as f:
                    apiLen = f.read()
                    sumNode+=(int)(apiLen)

                srcList=np.load(os.path.join(cgPath,"srcNp.npy")).tolist()
                sumEdge+=len(srcList)
            print(year,category,cnt,sumNode,sumEdge,sumNode*1.0/cnt,sumEdge*1.0/cnt)