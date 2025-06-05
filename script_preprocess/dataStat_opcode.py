import os
import shutil
import re
import numpy as np
import time

if __name__ == "__main__":
    # years=['data2018','data2019','data2020','data2021','data2022','Drebin']
    years=['Drebin','AndroZoo2010-2012','AndroZoo2','AndroZoo3','AndroZoo4']
    for year in years:
        print("year:",year)
        ApkRootName="apkPreprocess/TrainData/{}/ApkTrainSet".format(year)
        cgRootName="apkPreprocess/TrainData/{}/ApkPreProcessRes".format(year)
        smaliRootName="apkPreprocess/TrainData/{}/ApkSmaliFiles".format(year)
        for category in os.listdir(cgRootName):
            # if year=="Drebin" and category=="benign":
            #     continue
            if year=="Drebin" and category=="malware":
                continue
            cnt=0
            sumFz=0
            pathName=os.path.join(cgRootName,category)
            for fileName in os.listdir(pathName):
                cnt+=1
                # print("cnt:",cnt,time.ctime())
                cgPath=os.path.join(cgRootName,category,fileName)
                # print(cgPath)
                opcodeFn=os.path.join(cgPath,fileName+".opcodeSeq.txt")
                if not os.path.exists(opcodeFn):
                    print(opcodeFn)
                fz=os.path.getsize(opcodeFn)
                # print(fz)
                sumFz+=fz
            print(year,category,sumFz,cnt,sumFz*1.0/cnt)