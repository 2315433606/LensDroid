import os
import subprocess
import time
# fileName=apkName.type  +  前面是pathName ==> fn 

def DecodeByApkTool(fileName:str,apkName:str,category):
    #from fileName to dir
    pathName="apkPreprocess/TrainData/data_Reflection/ApkSmaliFiles"
    dir=os.path.join(pathName,category,apkName)
    if not os.path.exists(dir):
        os.makedirs(dir)
    apkToolDir="/usr/local/bin"
    apkToolCmd = "{0}/apktool d -f {1} -o {2} >/dev/null 2>&1".format(apkToolDir, fileName, dir)
    print(apkToolCmd)
    ps = subprocess.Popen(apkToolCmd,shell=True,stdout=0)
    ps.wait()
    print(ps.returncode)
    if ps.returncode:
        print("***Error:",apkToolCmd)

    # res=os.system(apkToolCmd)
    # if res!=0:
        # print("Error:",apkToolCmd)

if __name__ == "__main__":
    rootName="apkPreprocess/TrainData/data_Reflection/ApkTrainSet"
    for category in os.listdir(rootName):
        if category=="malware":
            cnt=0
            pathName=os.path.join(rootName,category)
            # print(pathName)
            # if category=="malware":
            #     continue
            for fileName in os.listdir(pathName):
                fn=os.path.join(pathName,fileName)
                apkName='.'.join(fileName.split('.')[:-1])
                print(cnt,time.ctime())
                cnt+=1
                print(fn,apkName,category)
                DecodeByApkTool(fn, apkName,category)
        
