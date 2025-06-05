import os
import numpy as np
from androguard.core.bytecodes.apk import APK
from PIL import Image
import time
# fileName=apkName.type  +  前面是pathName ==> fn 


def get_bytes(apk: APK, file_type: str) -> bytes:
    assert file_type in {".dex", ".so", ".xml"}
    for f in apk.get_files():
        if f.endswith(file_type):
            yield apk.get_file(f)

def generate_png(apk: APK, apkName: str, file_type: str,category):
    assert file_type in {".dex", ".so", ".xml"}
    stream = bytes()
    for s in get_bytes(apk, file_type):
        stream += s
    current_len = len(stream)
    return current_len

def generateSize(apk, apkName,category):
    dexSize = generate_png(apk, apkName, '.dex',category)
    xmlSize = generate_png(apk, apkName, '.xml',category)
    soSize  = generate_png(apk, apkName, '.so',category)
    return dexSize,xmlSize,soSize
    

if __name__ == "__main__":
    # years=['data2018','data2019','data2020','data2021','data2022','Drebin']
    years=['Drebin','AndroZoo2010-2012','AndroZoo2','AndroZoo3','AndroZoo4']
    for year in years:
        rootName="apkPreprocess/TrainData/{}/ApkTrainSet".format(year)
        for category in os.listdir(rootName):
            if year=="Drebin" and category=="malware":
                continue
            pathName=os.path.join(rootName,category)
            sumDex=0
            sumXml=0
            sumSo=0
            cnt=0
            for fileName in os.listdir(pathName):
                cnt+=1
                print("cnt:",cnt,time.ctime())
                fn=os.path.join(pathName,fileName)
                # print(fn)
                try:
                    apk = APK(fn)
                    apkName=""
                    if year=="Drebin":
                        # apkName=fileName 
                        apkName='.'.join(fileName.split('.')[:-1])
                    else:
                        apkName='.'.join(fileName.split('.')[:-1])

                    dexSize,xmlSize,soSize=generateSize(apk, apkName,category)
                    sumDex+=dexSize
                    sumXml+=xmlSize
                    sumSo+=soSize
                except:
                    print("Error****",fn)
            print(year,category,"sumDex,sumXml,sumSo",sumDex,sumXml,sumSo,"cnt",cnt)
            print("avgDex,avgXml,avgSo",sumDex*1.0/cnt,sumXml*1.0/cnt,sumSo*1.0/cnt)
